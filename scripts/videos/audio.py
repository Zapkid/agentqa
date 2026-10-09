"""Audio for coded videos: an original music bed synthesised from scratch, voice-ducking, WAV I/O.

The music is generated here (additive pads, a plucked arpeggio, sub bass, soft kick and shaker,
FFT convolution reverb), so it needs no samples and carries no licence. numpy only.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

SR = 48_000
BPM = 96
BEAT = 60 / BPM
BAR = 4 * BEAT

# D major: Dmaj9 - Bm7(add11) - Gmaj7 - Asus2, two bars each.
CHORDS = [
    [50, 57, 61, 64, 66],
    [47, 54, 57, 62, 64],
    [43, 50, 54, 59, 62],
    [45, 52, 59, 61, 64],
]
ROOTS = [38, 35, 43, 33]


def hz(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def _t(n: int) -> np.ndarray:
    return np.arange(n) / SR


def lowpass(x: np.ndarray, cutoff: float, slope: float = 2.0) -> np.ndarray:
    """Zero-phase FFT low-pass with a smooth roll-off (works on mono or (n, 2))."""
    n = x.shape[0]
    f = np.fft.rfftfreq(n, 1 / SR)
    gain = 1 / np.sqrt(1 + (f / cutoff) ** (2 * slope))
    spec = np.fft.rfft(x, axis=0)
    return np.fft.irfft(spec * (gain[:, None] if x.ndim == 2 else gain), n=n, axis=0)


def highpass(x: np.ndarray, cutoff: float) -> np.ndarray:
    return x - lowpass(x, cutoff, slope=2.0)


def pad_chord(notes: list[int], seconds: float) -> np.ndarray:
    """Warm detuned additive pad (band-limited saw, 3 voices) with slow attack and release."""
    n = int(seconds * SR)
    t = _t(n)
    out = np.zeros(n)
    for note in notes:
        for cents in (-7.0, 0.0, 7.0):
            f0 = hz(note) * 2 ** (cents / 1200)
            phase = np.random.default_rng(note * 7 + int(cents)).uniform(0, 2 * np.pi)
            for k in range(1, 9):
                if f0 * k > 9000:
                    break
                out += np.sin(2 * np.pi * f0 * k * t + phase * k) / (k**1.15)
    env = np.minimum(1, t / 1.4) * np.minimum(1, (seconds - t) / 1.6).clip(0, 1)
    return out * env / (len(notes) * 3 * 2.4)


def pluck(freq: float, seconds: float = 1.2) -> np.ndarray:
    n = int(seconds * SR)
    t = _t(n)
    tone = sum(
        amp * np.sin(2 * np.pi * freq * k * t) * np.exp(-t * k / 0.9)
        for k, amp in ((1, 1.0), (2, 0.34), (3, 0.14), (4, 0.08), (6, 0.04))
    )
    return tone * np.minimum(1, t / 0.003) * np.exp(-t / 0.3)


def shimmer(notes: list[int], seconds: float) -> np.ndarray:
    """A quiet, high, slowly breathing layer two octaves above the chord's top notes."""
    n = int(seconds * SR)
    t = _t(n)
    out = np.zeros(n)
    for i, note in enumerate(sorted(notes)[-2:]):
        f = hz(note + 24)
        out += np.sin(2 * np.pi * f * t) * (0.6 + 0.4 * np.sin(2 * np.pi * (0.17 + 0.05 * i) * t))
    env = np.minimum(1, t / 2.0) * np.minimum(1, (seconds - t) / 2.0).clip(0, 1)
    return out * env / 2


def kick() -> np.ndarray:
    n = int(0.45 * SR)
    t = _t(n)
    freq = 45 + 75 * np.exp(-t / 0.035)
    phase = 2 * np.pi * np.cumsum(freq) / SR
    return np.sin(phase) * np.exp(-t / 0.16)


def shaker(rng: np.random.Generator) -> np.ndarray:
    n = int(0.09 * SR)
    t = _t(n)
    noise = highpass(rng.standard_normal(n), 6000)
    return noise * np.minimum(1, t / 0.006) * np.exp(-t / 0.03)


def reverb(
    x: np.ndarray, seconds: float = 2.6, decay: float = 0.55, wet: float = 0.28
) -> np.ndarray:
    """Stereo FFT convolution with decaying, decorrelated noise (a soft hall)."""
    rng = np.random.default_rng(7)
    n_ir = int(seconds * SR)
    t = _t(n_ir)
    ir = rng.standard_normal((n_ir, 2)) * np.exp(-t / decay)[:, None]
    ir = lowpass(ir, 7000)
    ir /= np.sqrt((ir**2).sum(axis=0))
    n = x.shape[0] + n_ir
    size = 1 << (n - 1).bit_length()
    wet_sig = np.fft.irfft(
        np.fft.rfft(x, size, axis=0) * np.fft.rfft(ir, size, axis=0), size, axis=0
    )[: x.shape[0]]
    return (1 - wet) * x + wet * wet_sig


def place(track: np.ndarray, sound: np.ndarray, at: float, gain: float = 1.0) -> None:
    i = int(at * SR)
    if i >= track.shape[0]:
        return
    j = min(track.shape[0], i + sound.shape[0])
    if track.ndim == 2 and sound.ndim == 1:
        track[i:j] += gain * sound[: j - i, None]
    else:
        track[i:j] += gain * sound[: j - i]


def music_bed(total: float, *, groove_from: float = 7.5, outro: float = 6.0) -> np.ndarray:
    """Stereo music bed of `total` seconds: pads throughout, arpeggio and bass after the intro,
    a light groove in the middle, and a final chord that rings out."""
    n = int(total * SR)
    pads = np.zeros((n, 2))
    arp = np.zeros((n, 2))
    low = np.zeros((n, 2))
    drums = np.zeros((n, 2))
    air = np.zeros((n, 2))
    rng = np.random.default_rng(3)
    chord_len = 2 * BAR
    t0, idx = 0.0, 0
    while t0 < total:
        notes = CHORDS[idx % 4]
        last = t0 + chord_len >= total - 0.01
        length = (total - t0) if last else chord_len + 1.2  # overlap for smooth changes
        p = pad_chord(notes, max(2.0, length))
        place(pads, p, t0)
        sh = shimmer(notes, max(2.0, length))
        place(air, np.stack([sh, np.roll(sh, int(0.011 * SR))], axis=1), t0, 0.07)
        if t0 >= BAR and t0 < total - outro:  # bass enters after the first bar
            for b in range(2):
                bt = t0 + b * BAR
                sub = np.sin(2 * np.pi * hz(ROOTS[idx % 4]) * _t(int(BAR * SR)))
                sub *= np.minimum(1, _t(sub.size) / 0.05) * np.exp(-_t(sub.size) / 1.8)
                place(low, sub, bt, 0.09)
        if t0 >= BAR:
            tones = sorted(notes[1:])
            pattern = [0, 2, 1, 3, 2, 1, 3, 2]
            for step in range(16):
                st = t0 + step * BEAT / 2
                if st >= total - 2.5:
                    break
                note = tones[pattern[step % 8] % len(tones)] + 12
                g = 0.11 if step % 2 == 0 else 0.07
                pan = 0.35 if step % 2 else -0.35
                pl = pluck(hz(note))
                place(arp, np.stack([pl * (1 - pan), pl * (1 + pan)], axis=1), st, g)
        if groove_from <= t0 < total - outro:
            for beat in range(8):
                bt = t0 + beat * BEAT
                if beat % 4 in (0, 2):
                    place(drums, kick(), bt, 0.10)
                for half in (0, 1):
                    place(drums, shaker(rng), bt + half * BEAT / 2, 0.035 if half else 0.02)
        t0 += chord_len
        idx += 1
    # ping-pong delay on the arpeggio (dotted eighth)
    d = int(0.75 * BEAT * SR)
    echo = np.zeros_like(arp)
    echo[d:, 0] = arp[:-d, 1] * 0.38
    echo[d:, 1] = arp[:-d, 0] * 0.38
    echo[2 * d :] += arp[: -2 * d] * 0.14
    arp = arp + echo
    pads = lowpass(pads, 4200, slope=1.5)
    mix = highpass(reverb(pads * 0.9 + arp + air, wet=0.32) + low + drums, 30)
    fade = np.minimum(1, _t(n) / 1.5) * np.minimum(1, (total - _t(n)) / 3.0).clip(0, 1)
    mix *= fade[:, None]
    return mix / (np.abs(mix).max() + 1e-9) * 0.5  # -6 dBFS peak


def duck(music: np.ndarray, voice: np.ndarray, depth_db: float = -11.0) -> np.ndarray:
    """Lower the music while the voice speaks (attack 60 ms, release 450 ms)."""
    hop = SR // 100
    frames = voice.shape[0] // hop
    rms = np.sqrt((voice[: frames * hop].reshape(frames, hop) ** 2).mean(axis=1) + 1e-12)
    active = (20 * np.log10(rms) > -42).astype(float)
    env = np.zeros(frames)
    level = 0.0
    a, r = 1 - np.exp(-1 / 6), 1 - np.exp(-1 / 45)
    for i, target in enumerate(active):
        level += (a if target > level else r) * (target - level)
        env[i] = level
    gain_db = depth_db * np.interp(np.arange(music.shape[0]) / hop, np.arange(frames), env)
    return music * (10 ** (gain_db / 20))[:, None]


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        assert w.getframerate() == SR and w.getsampwidth() == 2, path
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(float) / 32768
        return data.reshape(-1, w.getnchannels()).mean(axis=1)


def write_wav(path: Path, x: np.ndarray) -> Path:
    x = np.clip(x, -1, 1)
    channels = 1 if x.ndim == 1 else x.shape[1]
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((x * 32767).astype("<i2").tobytes())
    return path
