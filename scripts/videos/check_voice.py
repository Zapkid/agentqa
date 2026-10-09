"""Check the narrated video's soundtrack: an independent speech recogniser (Whisper base.en via
sherpa-onnx) transcribes each narration line from the final mix, music included, and the word
error rate against the script is reported. Model files go to .cache/asr/ (GitHub release).

    uv run --with kokoro-onnx==0.6.1 --with soundfile --with sherpa-onnx \
        python scripts/videos/check_voice.py
"""

from __future__ import annotations

import re
import subprocess
import sys
import tarfile
import urllib.request
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import veroniqa_intro as intro

from agentqa.config import REPO_ROOT

ASR = REPO_ROOT / ".cache/asr"
MODEL = "sherpa-onnx-whisper-base.en"
URL = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{MODEL}.tar.bz2"


def words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", s.lower().replace("%", " percent"))


def edits(ref: list[str], hyp: list[str]) -> int:
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[:], i
        for j, h in enumerate(hyp, 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (r != h))
    return d[-1]


def main() -> int:
    import sherpa_onnx

    if not (ASR / MODEL).exists():
        ASR.mkdir(parents=True, exist_ok=True)
        archive = ASR / "model.tar.bz2"
        urllib.request.urlretrieve(URL, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(ASR, filter="data")
        archive.unlink()
    m = ASR / MODEL
    rec = sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=str(m / "base.en-encoder.onnx"), decoder=str(m / "base.en-decoder.onnx"),
        tokens=str(m / "base.en-tokens.txt"), num_threads=4,
    )  # fmt: skip
    wav = ASR / "mix16k.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(intro.OUT), "-map", "0:a", "-ac", "1",
         "-ar", "16000", str(wav)],
        check=True,
    )  # fmt: skip
    with wave.open(str(wav)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768
    scenes, _ = intro.narration()
    intro.synthesize(scenes)
    intro.timeline(scenes)
    errors = total = 0
    for sc in scenes:
        for start, end, line in sc.cues:
            stream = rec.create_stream()
            stream.accept_waveform(16000, x[int((start - 0.15) * 16000) : int((end + 0.3) * 16000)])
            rec.decode_stream(stream)
            ref, hyp = words(intro.spoken(line)), words(stream.result.text)
            e = edits(ref, hyp)
            errors, total = errors + e, total + len(ref)
            print(f"[{start:5.1f}s] {'ok ' if e == 0 else f'{e} x'} {stream.result.text.strip()}")
    print(f"word error rate: {errors}/{total} = {errors / total:.1%}")
    return 0 if errors / total < 0.05 else 1


if __name__ == "__main__":
    raise SystemExit(main())
