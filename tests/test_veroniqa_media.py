"""The media the site ships stays small: Vercel stores every deployment's image and serves the video
through a function, so each file the page loads has a hard ceiling."""

from __future__ import annotations

import re
from pathlib import Path

from agentqa.veroniqa import site

ROOT = Path(__file__).resolve().parent.parent
MAX_VIDEO_BYTES = 3_000_000  # the intro video, in decimal megabytes like a download counter
MAX_SHIPPED_BYTES = 3_200_000  # everything Dockerfile.vercel copies from media/


def _end_of_last_caption() -> float:
    cues = re.findall(r"-->\s+(\d+):(\d+):(\d+),(\d+)", site.SUBTITLES.read_text(encoding="utf-8"))
    h, m, s, ms = (int(x) for x in cues[-1])
    return h * 3600 + m * 60 + s + ms / 1000


def test_the_intro_video_is_short_and_under_3_mb() -> None:
    size = site.INTRO.stat().st_size
    assert 0 < size <= MAX_VIDEO_BYTES, f"{size} bytes"
    assert site.VIDEO_SECONDS <= 60  # a short introduction


def test_the_stated_length_matches_the_captions() -> None:
    # The captions end a moment before the closing card does (its tail is under 3 seconds).
    end = _end_of_last_caption()
    assert end <= site.VIDEO_SECONDS <= end + 3.5, (end, site.VIDEO_SECONDS)


def test_everything_the_image_ships_from_media_is_small() -> None:
    copied = re.findall(
        r"^COPY (media/\S+(?: media/\S+)*) ", (ROOT / "Dockerfile.vercel").read_text(), re.M
    )
    files = [ROOT / f for line in copied for f in line.split()]
    assert files, "Dockerfile.vercel no longer copies the media"
    total = sum(f.stat().st_size for f in files)
    assert total <= MAX_SHIPPED_BYTES, (
        f"{total} bytes: {[(f.name, f.stat().st_size) for f in files]}"
    )
    assert site.INTRO.with_suffix(".jpg").stat().st_size < 100_000  # the poster


def test_the_page_uses_the_video_length_it_states() -> None:
    page, md = site.home_page(), site.home_markdown()
    assert f"in {site.VIDEO_SECONDS} seconds" in page and f"{site.VIDEO_SECONDS} s video" in md
    assert 'property="og:video:width" content="1280"' in page
