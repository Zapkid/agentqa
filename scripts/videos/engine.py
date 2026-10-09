"""A tiny scene engine for coded videos: Pillow draws each frame, ffmpeg encodes H.264.

A video is a list of scenes; a scene is (seconds, draw) where draw(canvas, t, p) paints frame at
local time t (seconds) with progress p in [0, 1]. Scenes crossfade into each other.
"""

from __future__ import annotations

import math
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1280, 720, 30
FADE_S = 0.45

BG = (11, 16, 32)
PANEL = (22, 29, 51)
TEXT = (245, 245, 244)
MUTED = (161, 161, 170)
BLUE = (42, 120, 214)
ORANGE = (235, 104, 52)
GREEN = (27, 175, 122)

_FONT_DIR = Path(matplotlib.__file__).parent / "mpl-data/fonts/ttf"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(_FONT_DIR / name), size)


def ease(p: float) -> float:
    """Ease-in-out (smoothstep), clamped."""
    p = max(0.0, min(1.0, p))
    return p * p * (3 - 2 * p)


def window(t: float, start: float, length: float) -> float:
    """Eased progress of an animation that starts at `start` and lasts `length` seconds."""
    return ease((t - start) / length) if length > 0 else float(t >= start)


def blend(a: tuple[int, int, int], b: tuple[int, int, int], k: float) -> tuple[int, int, int]:
    return tuple(round(x + (y - x) * k) for x, y in zip(a, b, strict=True))  # type: ignore[return-value]


@dataclass
class Canvas:
    img: Image.Image
    draw: ImageDraw.ImageDraw

    @classmethod
    def new(cls) -> Canvas:
        img = Image.new("RGB", (W, H), BG)
        return cls(img, ImageDraw.Draw(img))

    def text(
        self,
        xy: tuple[float, float],
        s: str,
        size: int,
        color: tuple[int, int, int] = TEXT,
        *,
        bold: bool = False,
        alpha: float = 1.0,
        anchor: str = "la",
    ) -> None:
        if alpha <= 0:
            return
        self.draw.text(xy, s, font=font(size, bold), fill=blend(BG, color, alpha), anchor=anchor)

    def wrapped(
        self,
        xy: tuple[float, float],
        s: str,
        size: int,
        width: int,
        color: tuple[int, int, int] = TEXT,
        *,
        bold: bool = False,
        alpha: float = 1.0,
        spacing: float = 1.35,
    ) -> float:
        """Draw word-wrapped text; returns the y below the last line."""
        f = font(size, bold)
        x, y = xy
        line = ""
        for word in s.split():
            trial = f"{line} {word}".strip()
            if f.getlength(trial) <= width:
                line = trial
            else:
                self.text((x, y), line, size, color, bold=bold, alpha=alpha)
                y += size * spacing
                line = word
        if line:
            self.text((x, y), line, size, color, bold=bold, alpha=alpha)
            y += size * spacing
        return y

    def rect(
        self,
        box: tuple[float, float, float, float],
        color: tuple[int, int, int],
        radius: int = 12,
        alpha: float = 1.0,
    ) -> None:
        if alpha > 0 and box[2] > box[0]:
            self.draw.rounded_rectangle(box, radius=radius, fill=blend(BG, color, alpha))

    def paste(self, im: Image.Image, xy: tuple[int, int], alpha: float = 1.0) -> None:
        if alpha >= 1:
            self.img.paste(im, xy)
        elif alpha > 0:
            region = self.img.crop((xy[0], xy[1], xy[0] + im.width, xy[1] + im.height))
            self.img.paste(Image.blend(region, im.convert("RGB"), alpha), xy)


Draw = Callable[[Canvas, float, float], None]


def overline(c: Canvas, s: str, alpha: float = 1.0) -> None:
    c.text((96, 84), s.upper(), 20, ORANGE, bold=True, alpha=alpha)


def footer(c: Canvas, s: str, alpha: float = 1.0) -> None:
    c.text((96, H - 56), s, 16, MUTED, alpha=alpha)


def ken_burns(
    im: Image.Image,
    box: tuple[int, int, int, int],
    p: float,
    zoom: float = 1.06,
    focus: tuple[float, float] = (0.5, 0.35),
) -> Image.Image:
    """Scale an image to cover `box` size, zooming slowly toward `focus` as p goes 0 -> 1."""
    bw, bh = box[2] - box[0], box[3] - box[1]
    scale = max(bw / im.width, bh / im.height) * (1 + (zoom - 1) * ease(p))
    sw, sh = int(im.width * scale), int(im.height * scale)
    big = im.resize((sw, sh), Image.LANCZOS)
    left = int((sw - bw) * focus[0])
    top = int((sh - bh) * focus[1])
    return big.crop((left, top, left + bw, top + bh))


def render(path: Path, scenes: list[tuple[float, Draw]]) -> Path:
    """Encode the scenes to an H.264 MP4 (yuv420p, faststart) with ffmpeg."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-c:v", "libx264", "-preset", "slow", "-crf", "24", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(path),
    ]  # fmt: skip
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None

    def frame_at(i: int, t: float) -> Image.Image:
        dur, draw = scenes[i]
        c = Canvas.new()
        draw(c, t, min(1.0, t / dur))
        return c.img

    for i, (dur, _) in enumerate(scenes):
        n = math.ceil(dur * FPS)
        for k in range(n):
            t = k / FPS
            img = frame_at(i, t)
            fade_in = i > 0 and t < FADE_S
            if fade_in:  # crossfade from the previous scene's last frame
                prev = frame_at(i - 1, scenes[i - 1][0])
                img = Image.blend(prev, img, ease(t / FADE_S))
            proc.stdin.write(img.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {path}")
    return path
