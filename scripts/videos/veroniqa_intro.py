"""VeroniQA's narrated introduction (1080p, voice-over, music, captions).

    uv run --with kokoro-onnx==0.6.1 --with soundfile python scripts/videos/veroniqa_intro.py

Inputs: media/veroniqa/ (real screenshots and the run they show; capture_veroniqa.py) and
results/*.json for the benchmark numbers. The narration is generated from that data.
Voice: Kokoro-82M (Apache-2.0), voice af_heart, run locally; model files go to .cache/tts/
(downloaded from the kokoro-onnx GitHub release if missing). Music: synthesised by audio.py.
Outputs media/videos/veroniqa-intro.mp4 (H.264 + AAC + a soft English subtitle track) and
media/videos/veroniqa-intro.srt.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import audio
from engine import ease, window

from agentqa.config import REPO_ROOT
from agentqa.evals.report import latest_results

W, H, FPS = 1920, 1080, 30
CACHE = REPO_ROOT / ".cache/veroniqa-intro"
TTS = REPO_ROOT / ".cache/tts"
SHOTS = REPO_ROOT / "media/veroniqa"
OUT = REPO_ROOT / "media/videos/veroniqa-intro.mp4"
MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
VOICE, SPEED = "af_heart", 0.96
# How the voice should say things (captions keep the real spelling).
SAY = [(r"\bVeroniQA\b", "Veronica"), (r"\bveroniqa\b", "Veronica"), (r"\bAgentQA\b", "Agent QA"),
       (r"(\d+)%", r"\1 percent"), (r"cloud-metadata", "cloud metadata")]  # fmt: skip

NAVY, NAVY2 = (9, 13, 28), (17, 24, 49)
INK, MUTED, FAINT = (246, 246, 244), (170, 176, 196), (110, 118, 140)
ORANGE, AMBER, BLUE = (235, 104, 52), (245, 166, 90), (64, 132, 230)


# ---------------------------------------------------------------- fonts


def _font_path(weight: str) -> str:
    inter = Path(f"/usr/share/fonts/opentype/inter/Inter-{weight}.otf")
    if inter.exists():
        return str(inter)
    import matplotlib

    bold = weight in ("Bold", "SemiBold", "ExtraBold")
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return str(Path(matplotlib.__file__).parent / "mpl-data/fonts/ttf" / name)


_FONTS: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def F(size: int, weight: str = "Regular") -> ImageFont.FreeTypeFont:
    key = (weight, size)
    if key not in _FONTS:
        _FONTS[key] = ImageFont.truetype(_font_path(weight), size)
    return _FONTS[key]


def MONO(size: int) -> ImageFont.FreeTypeFont:
    import matplotlib

    path = Path(matplotlib.__file__).parent / "mpl-data/fonts/ttf/DejaVuSansMono.ttf"
    return ImageFont.truetype(str(path), size)


# ---------------------------------------------------------------- script


@dataclass
class Scene:
    key: str
    lines: list[str]
    lead: float = 0.7
    tail: float = 1.0
    voice: list[np.ndarray] = field(default_factory=list)
    start: float = 0.0
    cues: list[tuple[float, float, str]] = field(default_factory=list)

    @property
    def duration(self) -> float:
        speech = sum(v.size for v in self.voice) / audio.SR + 0.32 * max(0, len(self.voice) - 1)
        return self.lead + speech + self.tail


def narration() -> tuple[list[Scene], dict[str, Any]]:
    facts = json.loads((SHOTS / "run.json").read_text())
    func_path, _ = latest_results()
    assert func_path, "no results/*.json"
    st = json.loads(func_path.read_text())["strategies"]
    s0, s3 = st["S0"]["aggregate"], st["S3"]["aggregate"]
    recall = round(s3["recall"]["mean"] * 100)
    ratio = round(
        100 * s3["cost_usd_list_equivalent"]["mean"] / s0["cost_usd_list_equivalent"]["mean"]
    )
    found, seeded = len(facts["product_bugs"]), len(facts["seeded_bugs"])
    which = "both" if found == seeded == 2 else ("all" if found == seeded else f"{found} of the")
    data = {"recall": recall, "ratio": ratio, "n": s3["n"], "found": found, "seeded": seeded,
            "passed": facts["outcomes"].get("passed", 0), "facts": facts}  # fmt: skip
    scenes = [
        Scene("intro", ["Meet VeroniQA, your AI assistant for API testing.",
                        "She turns your requirements into tests that run, and results you can trust."],
              lead=1.8),
        Scene("knowledge", ["Start a project, and give her what your team already has: requirement "
                            "documents, PDF files, or links to your docs.",
                            "Every source is indexed, and scanned for prompt injection before she "
                            "ever reads it."]),
        Scene("ask", ["Ask a question in plain English.",
                      "VeroniQA answers from your own documents, and shows the passages she used.",
                      "If the answer isn't there, she tells you, instead of guessing."]),
        Scene("links", ["Links are fetched safely.",
                        "Internal and cloud-metadata addresses are refused, so your knowledge base "
                        "can't be turned against your network."]),
        Scene("run", ["When you're ready, just say: run the tests.",
                      "VeroniQA plans, writes and runs API tests against your service, then "
                      "triages every failure, with evidence.",
                      f"In this demo, she found {which} planted bugs, with {data['passed']} tests "
                      "passing."]),
        Scene("proof", ["Behind her, the AgentQA engine gives easy work to cheaper models, and "
                        "checks every "
                        "result.",
                        f"On our simulated benchmark, it found {recall}% of the planted bugs, at "
                        f"{ratio}% of the cost."]),
        Scene("outro", ["VeroniQA. Ask, add knowledge, and run the tests."], tail=4.2),
    ]  # fmt: skip
    return scenes, data


def spoken(text: str) -> str:
    for pattern, repl in SAY:
        text = re.sub(pattern, repl, text)
    return text


# ---------------------------------------------------------------- voice


def _ensure_model() -> None:
    TTS.mkdir(parents=True, exist_ok=True)
    for name in ("kokoro-v1.0.onnx", "voices-v1.0.bin"):
        if not (TTS / name).exists():
            print(f"downloading {name} ...")
            urllib.request.urlretrieve(MODEL_URL + name, TTS / name)  # noqa: S310 - fixed https URL


def synthesize(scenes: list[Scene]) -> None:
    """Kokoro TTS per line (cached by text and voice), resampled to 48 kHz, silence trimmed."""
    voice_dir = CACHE / "voice"
    voice_dir.mkdir(parents=True, exist_ok=True)
    kokoro = None
    for sc in scenes:
        sc.voice = []
        for line in sc.lines:
            say = spoken(line)
            key = hashlib.sha256(f"{VOICE}|{SPEED}|{say}".encode()).hexdigest()[:16]
            wav48 = voice_dir / f"{key}.wav"
            if not wav48.exists():
                if kokoro is None:
                    _ensure_model()
                    from kokoro_onnx import Kokoro

                    kokoro = Kokoro(str(TTS / "kokoro-v1.0.onnx"), str(TTS / "voices-v1.0.bin"))
                samples, sr = kokoro.create(say, voice=VOICE, speed=SPEED, lang="en-us")
                raw = voice_dir / f"{key}-raw.wav"
                import soundfile as sf

                sf.write(raw, samples, sr)
                subprocess.run(
                    ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-af",
                     "aresample=48000:resampler=soxr", "-ac", "1", "-sample_fmt", "s16", str(wav48)],
                    check=True,
                )  # fmt: skip
                raw.unlink()
            x = audio.read_wav(wav48)
            loud = np.flatnonzero(np.abs(x) > 0.01)
            sc.voice.append(x[max(0, loud[0] - 960) : loud[-1] + 2400] if loud.size else x)


def timeline(scenes: list[Scene]) -> float:
    t = 0.0
    for sc in scenes:
        sc.start = t
        cursor = t + sc.lead
        sc.cues = []
        for line, v in zip(sc.lines, sc.voice, strict=True):
            sc.cues.append((cursor, cursor + v.size / audio.SR, line))
            cursor += v.size / audio.SR + 0.32
        t += sc.duration
    return t


# ---------------------------------------------------------------- drawing helpers


def _background() -> Image.Image:
    y = np.linspace(0, 1, H)[:, None, None]
    x = np.linspace(0, 1, W)[None, :, None]
    base = np.array(NAVY) * (1 - y) + np.array(NAVY2) * y + 6 * x
    img = Image.fromarray(np.clip(base, 0, 255).astype("uint8"), "RGB").convert("RGBA")
    dots = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(dots)
    for gx in range(0, W, 48):
        for gy in range(0, H, 48):
            d.ellipse((gx - 1, gy - 1, gx + 1, gy + 1), fill=(255, 255, 255, 14))
    return Image.alpha_composite(img, dots)


def _glow(color: tuple[int, int, int], radius: int, alpha: int) -> Image.Image:
    size = radius * 4
    g = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(g).ellipse((radius, radius, 3 * radius, 3 * radius), fill=(*color, alpha))
    return g.filter(ImageFilter.GaussianBlur(radius * 0.55))


BG = None
GLOWS: list[Image.Image] = []


def frame_base(t: float) -> Image.Image:
    global BG
    if BG is None:
        BG = _background()
        GLOWS.extend([_glow(ORANGE, 380, 70), _glow(BLUE, 460, 60)])
    img = BG.copy()
    for i, g in enumerate(GLOWS):
        cx = W * (0.78 if i == 0 else 0.18) + 120 * math.sin(t * 0.21 + i * 2)
        cy = H * (0.22 if i == 0 else 0.8) + 80 * math.cos(t * 0.17 + i)
        img.alpha_composite(g, (int(cx - g.width / 2), int(cy - g.height / 2)))
    return img


def text(d: ImageDraw.ImageDraw, xy: tuple[float, float], s: str, font: ImageFont.FreeTypeFont,
         color: tuple[int, int, int], alpha: float = 1.0, anchor: str = "la") -> None:  # fmt: skip
    if alpha > 0.01:
        d.text(xy, s, font=font, fill=(*color, int(255 * min(1, alpha))), anchor=anchor)


def wrap(s: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    lines, line = [], ""
    for word in s.split():
        trial = f"{line} {word}".strip()
        if font.getlength(trial) <= width:
            line = trial
        else:
            lines.append(line)
            line = word
    return [*lines, line] if line else lines


def logo(size: int) -> Image.Image:
    """VeroniQA's mark: a gradient speech bubble with a check-mark 'V'."""
    s = size * 2  # supersample
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    grad = np.zeros((s, s, 4), dtype="uint8")
    k = np.linspace(0, 1, s)[None, :] * 0.6 + np.linspace(0, 1, s)[:, None] * 0.4
    for c in range(3):
        grad[..., c] = (ORANGE[c] * (1 - k) + AMBER[c] * k).astype("uint8")
    grad[..., 3] = 255
    mask = Image.new("L", (s, s), 0)
    md = ImageDraw.Draw(mask)
    pad = s * 0.06
    md.rounded_rectangle((pad, pad, s - pad, s * 0.84), radius=int(s * 0.24), fill=255)
    md.polygon([(s * 0.28, s * 0.78), (s * 0.24, s * 0.97), (s * 0.48, s * 0.8)], fill=255)
    img.paste(Image.fromarray(grad, "RGBA"), (0, 0), mask)
    d = ImageDraw.Draw(img)
    w = int(s * 0.075)
    d.line([(s * 0.3, s * 0.42), (s * 0.45, s * 0.62), (s * 0.72, s * 0.24)],
           fill=(255, 255, 255, 255), width=w, joint="curve")  # fmt: skip
    for p in [(s * 0.3, s * 0.42), (s * 0.72, s * 0.24)]:
        d.ellipse(
            (p[0] - w / 2, p[1] - w / 2, p[0] + w / 2, p[1] + w / 2), fill=(255, 255, 255, 255)
        )
    return img.resize((size, size), Image.LANCZOS)


LOGO_CACHE: dict[int, Image.Image] = {}


def paste_logo(img: Image.Image, center: tuple[float, float], size: int, alpha: float) -> None:
    if size < 4 or alpha <= 0:
        return
    if size not in LOGO_CACHE:
        LOGO_CACHE[size] = logo(size)
    lg = LOGO_CACHE[size]
    if alpha < 1:
        lg = lg.copy()
        lg.putalpha(lg.getchannel("A").point(lambda a: int(a * alpha)))
    img.alpha_composite(lg, (int(center[0] - size / 2), int(center[1] - size / 2)))


# ---------------------------------------------------------------- scene renderers


SHOT_CACHE: dict[str, Image.Image] = {}


def browser(img: Image.Image, shot: str, box: tuple[int, int, int, int], p: float, t: float,
            highlight: list[float] | None, scale: float, appear: float) -> None:  # fmt: skip
    """A browser window with the screenshot, slowly zooming toward the highlight."""
    if shot not in SHOT_CACHE:
        SHOT_CACHE[shot] = Image.open(SHOTS / f"{shot}.png").convert("RGB")
    src = SHOT_CACHE[shot]
    x0, y0, x1, y1 = box
    bar = 46
    iw, ih = x1 - x0, y1 - y0 - bar
    rise = (1 - ease(appear)) * 40
    y0r = y0 + rise
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle((x0 + 10, y0r + 24, x1 + 10, y1 + rise + 30),
                                             radius=22, fill=(0, 0, 0, 150))  # fmt: skip
    layer.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(26)))
    ld.rounded_rectangle((x0, y0r, x1, y1 + rise), radius=18, fill=(236, 238, 242, 255))
    for i, c in enumerate([(237, 106, 94), (245, 191, 79), (98, 197, 84)]):
        ld.ellipse((x0 + 22 + i * 24, y0r + 17, x0 + 36 + i * 24, y0r + 31), fill=(*c, 255))
    ld.rounded_rectangle(
        (x0 + 120, y0r + 10, x1 - 120, y0r + 37), radius=13, fill=(255, 255, 255, 255)
    )
    text(ld, ((x0 + x1) / 2, y0r + 24), "127.0.0.1:8501  ·  VeroniQA", F(17, "Medium"),
         (120, 126, 140), anchor="mm")  # fmt: skip
    # zoom toward the highlight
    hx, hy = (0.5, 0.45)
    if highlight:
        hx = (highlight[0] + highlight[2] / 2) / 1440
        hy = (highlight[1] + highlight[3] / 2) / 900
    z = 1.0 + 0.16 * ease(min(1, max(0, (t - 0.8) / 4.0)))
    cw, ch = src.width / z, src.height / z
    cx = min(max(hx * src.width, cw / 2), src.width - cw / 2)
    cy = min(max(hy * src.height, ch / 2), src.height - ch / 2)
    crop = (cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2)
    view = src.resize((iw, ih), Image.BICUBIC, box=crop, reducing_gap=2.0).convert("RGBA")
    mask = Image.new("L", (iw, ih), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, -20, iw, ih), radius=18, fill=255)
    layer.paste(view, (x0, int(y0r + bar)), mask)
    if highlight and t > 1.2:
        k = ease(min(1, (t - 1.2) / 0.5))
        sx, sy = iw / cw, ih / ch
        hx0 = x0 + (highlight[0] * scale - crop[0]) * sx
        hy0 = y0r + bar + (highlight[1] * scale - crop[1]) * sy
        hx1 = hx0 + highlight[2] * scale * sx
        hy1 = hy0 + highlight[3] * scale * sy
        pulse = 0.5 + 0.5 * math.sin(t * 3.2)
        glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(glow).rounded_rectangle((hx0 - 6, hy0 - 6, hx1 + 6, hy1 + 6), radius=16,
                                               outline=(*ORANGE, int(150 * k * (0.6 + 0.4 * pulse))),
                                               width=14)  # fmt: skip
        layer.alpha_composite(glow.filter(ImageFilter.GaussianBlur(8)))
        ImageDraw.Draw(layer).rounded_rectangle((hx0, hy0, hx1, hy1), radius=12,
                                                outline=(*ORANGE, int(255 * k)), width=5)  # fmt: skip
    if appear < 1:
        layer.putalpha(layer.getchannel("A").point(lambda a: int(a * ease(appear))))
    img.alpha_composite(layer)


FEATURES = {
    "knowledge": ("01", "KNOWLEDGE", "Your requirements, indexed and checked",
                  ["Upload Markdown, text, HTML or PDF", "Add links to your docs",
                   "Prompt-injection scan on every source"], "knowledge"),
    "ask": ("02", "ASK", "Answers from your documents, with sources",
            ["Hybrid vector + keyword retrieval", "Every claim cites a passage",
             "Says so when it doesn't know"], "chat"),
    "links": ("03", "SAFE LINKS", "Your network stays off-limits",
              ["http(s) only, public addresses only", "Re-checked after every redirect",
               "Size and content-type limits"], "ssrf"),
    "run": ("04", "RUN THE TESTS", "From chat to triaged findings",
            ["Plans, writes and runs API tests", "Triage with evidence and a curl repro",
             "Full report, one click away"], "runs"),
}  # fmt: skip


def draw_feature(img: Image.Image, key: str, t: float, data: dict[str, Any]) -> None:
    num, label, headline, bullets, shot = FEATURES[key]
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    a = window(t, 0.1, 0.5)
    text(d, (120, 236), num, F(26, "Bold"), ORANGE, a)
    d.line((166, 250, 206, 250), fill=(*ORANGE, int(255 * a)), width=3)
    text(d, (220, 236), label, F(22, "SemiBold"), ORANGE, a)
    y = 300.0
    for i, line in enumerate(wrap(headline, F(54, "Bold"), 600)):
        text(d, (120 - (1 - window(t, 0.25 + 0.1 * i, 0.6)) * 30, y), line, F(54, "Bold"), INK,
             window(t, 0.25 + 0.1 * i, 0.6))  # fmt: skip
        y += 66
    y += 30
    for i, b in enumerate(bullets):
        k = window(t, 1.0 + 0.45 * i, 0.5)
        d.ellipse((122, y + 9, 140, y + 27), fill=(*ORANGE, int(255 * k)))
        d.line([(127, y + 18), (130, y + 22), (136, y + 13)], fill=(255, 255, 255, int(255 * k)),
               width=3)  # fmt: skip
        text(d, (158, y + 2), b, F(27, "Medium"), MUTED, k)
        y += 52
    img.alpha_composite(layer)
    facts = data["facts"]
    browser(img, shot, (770, 200, 1800, 200 + 46 + 640), 0, t,
            facts.get("highlights", {}).get(shot), facts.get("screenshot_scale", 1),
            window(t, 0.2, 0.7))  # fmt: skip


def draw_intro(img: Image.Image, t: float, data: dict[str, Any]) -> None:
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    pop = ease(min(1, t / 0.9))
    cy = 300
    if 0.2 < t < 2.0:  # ripple
        k = (t - 0.2) / 1.8
        r = 110 + 240 * ease(k)
        d.ellipse((W / 2 - r, cy - r, W / 2 + r, cy + r), outline=(*ORANGE, int(110 * (1 - k))),
                  width=3)  # fmt: skip
    img.alpha_composite(layer)
    paste_logo(img, (W / 2, cy), int(180 * (0.6 + 0.4 * pop)), pop)
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    name = "VeroniQA"
    f = F(128, "Bold")
    x = W / 2 - f.getlength(name) / 2
    for i, ch in enumerate(name):
        k = window(t, 0.7 + 0.06 * i, 0.5)
        text(d, (x, 610 + (1 - k) * 24), ch, f, INK, k, anchor="ls")
        x += f.getlength(ch)
    text(d, (W / 2, 705), "Your AI assistant for API testing", F(38, "Medium"), MUTED,
         window(t, 1.5, 0.6), anchor="ms")  # fmt: skip
    text(d, (W / 2, 770), "BUILT ON AGENTQA", F(20, "SemiBold"), ORANGE, window(t, 2.0, 0.6),
         anchor="ms")  # fmt: skip
    img.alpha_composite(layer)


def draw_proof(img: Image.Image, t: float, data: dict[str, Any]) -> None:
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    a = window(t, 0.1, 0.5)
    text(d, (W / 2, 210), "THE ENGINE BEHIND HER", F(24, "SemiBold"), ORANGE, a, anchor="mm")
    text(d, (W / 2, 280), "Cheapest model that gets it right, every result checked",
         F(48, "Bold"), INK, window(t, 0.25, 0.6), anchor="mm")  # fmt: skip
    cards = [(data["recall"], "%", "of planted bugs found"),
             (data["ratio"], "%", "of the cost of using the strongest model for everything")]  # fmt: skip
    for i, (value, unit, label) in enumerate(cards):
        cx = W / 2 + (i - 0.5) * 620
        k = window(t, 0.8 + 0.5 * i, 0.6)
        box = (cx - 270, 380, cx + 270, 720)
        d.rounded_rectangle(box, radius=28, fill=(255, 255, 255, int(16 * k)),
                            outline=(255, 255, 255, int(40 * k)), width=2)  # fmt: skip
        shown = round(value * ease(min(1, max(0, (t - 0.9 - 0.5 * i) / 1.6))))
        text(d, (cx, 520), f"{shown}{unit}", F(150, "Bold"), ORANGE if i == 0 else AMBER, k,
             anchor="mm")  # fmt: skip
        for j, line in enumerate(wrap(label, F(28, "Medium"), 440)):
            text(d, (cx, 630 + j * 38), line, F(28, "Medium"), MUTED, k, anchor="mm")
    text(d, (W / 2, 800), f"Bundled benchmark, simulated models, {data['n']} cold runs per "
         "strategy. Real-model runs: make eval-live.", F(22), FAINT, window(t, 2.2, 0.6),
         anchor="mm")  # fmt: skip
    img.alpha_composite(layer)


def draw_outro(img: Image.Image, t: float, data: dict[str, Any]) -> None:
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    pop = ease(min(1, t / 0.8))
    paste_logo(img, (W / 2, 300), int(150 * (0.7 + 0.3 * pop)), pop)
    text(d, (W / 2, 470), "VeroniQA", F(110, "Bold"), INK, window(t, 0.3, 0.6), anchor="mm")
    text(d, (W / 2, 565), "Ask.  Add knowledge.  Run the tests.", F(40, "Medium"), ORANGE,
         window(t, 0.7, 0.6), anchor="mm")  # fmt: skip
    k = window(t, 1.3, 0.6)
    cmd = "make veroniqa"
    f = MONO(34)
    w = f.getlength(cmd) + 70
    d.rounded_rectangle((W / 2 - w / 2, 630, W / 2 + w / 2, 700), radius=35,
                        fill=(255, 255, 255, int(22 * k)), outline=(*ORANGE, int(200 * k)), width=2)  # fmt: skip
    text(d, (W / 2, 666), cmd, f, INK, k, anchor="mm")
    text(d, (W / 2, 760), "github.com/Zapkid/agentqa", F(26, "Medium"), MUTED, window(t, 1.7, 0.6),
         anchor="mm")  # fmt: skip
    text(d, (W / 2, H - 60), "Demo: simulated models on the bundled Orders API. "
         "Screens are the real app.", F(20), FAINT, window(t, 2.0, 0.6), anchor="mm")  # fmt: skip
    img.alpha_composite(layer)


def draw_caption(img: Image.Image, now: float, scenes: list[Scene]) -> None:
    for sc in scenes:
        for start, end, line in sc.cues:
            if start - 0.12 <= now <= end + 0.25:
                k = min(1.0, (now - start + 0.12) / 0.18, (end + 0.25 - now) / 0.2)
                f = F(31, "Medium")
                rows = wrap(line, f, 1300)
                h = 22 + 44 * len(rows)
                y0 = H - 70 - h
                wmax = max(f.getlength(r) for r in rows) + 56
                layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                ld = ImageDraw.Draw(layer)
                ld.rounded_rectangle((W / 2 - wmax / 2, y0, W / 2 + wmax / 2, y0 + h), radius=16,
                                     fill=(6, 9, 20, int(190 * k)))  # fmt: skip
                for i, r in enumerate(rows):
                    text(ld, (W / 2, y0 + 33 + i * 44), r, f, INK, k, anchor="mm")
                img.alpha_composite(layer)
                return


def scene_frame(sc: Scene, t: float, now: float, scenes: list[Scene], data: dict[str, Any],
                captions: bool = True) -> Image.Image:  # fmt: skip
    img = frame_base(now)
    if sc.key == "intro":
        draw_intro(img, t, data)
    elif sc.key == "proof":
        draw_proof(img, t, data)
    elif sc.key == "outro":
        draw_outro(img, t, data)
    else:
        draw_feature(img, sc.key, t, data)
    if captions:
        draw_caption(img, now, scenes)
    return img


# ---------------------------------------------------------------- assembly


def srt(scenes: list[Scene], path: Path) -> Path:
    def ts(x: float) -> str:
        ms = round(x * 1000)
        return (
            f"{ms // 3_600_000:02d}:{ms // 60_000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"
        )

    cues = [c for sc in scenes for c in sc.cues]
    body = "\n".join(
        f"{i}\n{ts(a)} --> {ts(b + 0.2)}\n{s}\n" for i, (a, b, s) in enumerate(cues, 1)
    )
    path.write_text(body, encoding="utf-8")
    return path


def render_video(scenes: list[Scene], total: float, data: dict[str, Any], path: Path) -> None:
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow",
           "-crf", "20", "-pix_fmt", "yuv420p", "-tune", "animation", str(path)]  # fmt: skip
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    fade = 0.5
    n = math.ceil(total * FPS)
    for i in range(n):
        now = i / FPS
        idx = max(j for j, sc in enumerate(scenes) if sc.start <= now + 1e-9)
        sc = scenes[idx]
        t = now - sc.start
        img = scene_frame(sc, t, now, scenes, data, captions=False)
        if idx > 0 and t < fade:
            prev = scenes[idx - 1]
            img = Image.blend(scene_frame(prev, prev.duration, now, scenes, data, False), img,
                              ease(t / fade))  # fmt: skip
        draw_caption(img, now, scenes)
        proc.stdin.write(img.convert("RGB").tobytes())
        if i % (FPS * 10) == 0:
            print(f"  video {now:5.1f}/{total:.1f}s")
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg (video) failed")


def render_audio(scenes: list[Scene], total: float, path: Path) -> None:
    n = int(total * audio.SR)
    vo = np.zeros(n)
    for sc in scenes:
        for (start, _, _), v in zip(sc.cues, sc.voice, strict=True):
            i = int(start * audio.SR)
            vo[i : i + v.size] += v[: max(0, n - i)]
    music = audio.duck(audio.music_bed(total), vo)
    vo_path = audio.write_wav(CACHE / "voice.wav", vo * 0.9)
    mu_path = audio.write_wav(CACHE / "music.wav", music * 0.42)
    graph = (
        "[0:a]highpass=f=75,equalizer=f=220:t=q:w=1.0:g=-2,equalizer=f=3200:t=q:w=1.2:g=2.5,"
        "acompressor=threshold=-20dB:ratio=3:attack=5:release=120:makeup=2,"
        "aformat=channel_layouts=stereo[v];"
        "[1:a]equalizer=f=2800:t=q:w=1.5:g=-3[m];"
        "[v][m]amix=inputs=2:normalize=0,"
        "loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000[out]"
    )
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(vo_path), "-i", str(mu_path),
         "-filter_complex", graph, "-map", "[out]", "-c:a", "aac", "-b:a", "192k", str(path)],
        check=True,
    )  # fmt: skip


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    scenes, data = narration()
    synthesize(scenes)
    total = timeline(scenes)
    print(f"timeline {total:.1f}s: " + ", ".join(f"{s.key} {s.duration:.1f}s" for s in scenes))
    subs = srt(scenes, OUT.with_suffix(".srt"))
    render_audio(scenes, total, CACHE / "audio.m4a")
    render_video(scenes, total, data, CACHE / "video.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(CACHE / "video.mp4"),
         "-i", str(CACHE / "audio.m4a"), "-i", str(subs), "-map", "0:v", "-map", "1:a",
         "-map", "2:s", "-c:v", "copy", "-c:a", "copy", "-c:s", "mov_text",
         "-metadata:s:s:0", "language=eng", "-metadata:s:a:0", "language=eng",
         "-metadata", "title=Meet VeroniQA", "-movflags", "+faststart", str(OUT)],
        check=True,
    )  # fmt: skip
    print(f"wrote {OUT.relative_to(REPO_ROOT)} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
