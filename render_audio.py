"""
render_audio.py
===============

Render ArcaNet MIDI samples to WAV and MP3 using only numpy + ffmpeg.

This is a minimal pure-Python Standard-MIDI-File parser plus a tiny
additive/AM piano-like synthesizer.  It is not a high-fidelity
soundfont renderer, but it produces pleasant and musically faithful
audio previews that can be embedded directly in a static web page.

Usage:
    python render_audio.py

Dependencies:
    - numpy              (stdlib pip package)
    - ffmpeg             (for WAV -> MP3 transcoding)
"""

from __future__ import annotations

import os
import struct
import subprocess
import wave

import numpy as np

SR = 44100                      # sample rate
OUT_DIR = os.path.join(os.path.dirname(__file__), "docs", "audio")
MIDI_DIR = os.path.join(os.path.dirname(__file__), "examples")


# ---------------------------------------------------------------------------
# Minimal SMF parser (Format 0, single track)
# ---------------------------------------------------------------------------

def _read_vlq(buf: bytes, i: int) -> tuple[int, int]:
    value = 0
    while True:
        byte = buf[i]
        i += 1
        value = (value << 7) | (byte & 0x7F)
        if not (byte & 0x80):
            return value, i


def parse_smf(path: str) -> tuple[list[tuple[float, float, int, int]], int]:
    """Return (notes, tempo_us_per_qn) where notes is a list of
    (onset_seconds, duration_seconds, pitch, velocity)."""
    with open(path, "rb") as fh:
        data = fh.read()

    assert data[:4] == b"MThd"
    ppq = struct.unpack(">H", data[12:14])[0]

    # Track
    assert data[14:18] == b"MTrk"
    track_len = struct.unpack(">I", data[18:22])[0]
    track = data[22:22 + track_len]

    tempo_us = 500_000  # default 120 bpm
    i = 0
    abs_tick = 0
    running_status = 0
    notes_on: dict[int, tuple[int, int]] = {}  # pitch -> (tick, vel)
    raw_notes = []  # (on_tick, off_tick, pitch, vel)

    while i < len(track):
        delta, i = _read_vlq(track, i)
        abs_tick += delta
        status = track[i]
        if status & 0x80:
            running_status = status
            i += 1
        else:
            status = running_status

        if status == 0xFF:  # meta
            meta_type = track[i]; i += 1
            length, i = _read_vlq(track, i)
            payload = track[i:i + length]
            i += length
            if meta_type == 0x51 and length == 3:
                tempo_us = int.from_bytes(payload, "big")
            continue

        if status in (0xF0, 0xF7):
            length, i = _read_vlq(track, i)
            i += length
            continue

        high = status & 0xF0
        if high == 0x90:
            pitch = track[i]; vel = track[i + 1]; i += 2
            if vel == 0:
                if pitch in notes_on:
                    on_tick, on_vel = notes_on.pop(pitch)
                    raw_notes.append((on_tick, abs_tick, pitch, on_vel))
            else:
                notes_on[pitch] = (abs_tick, vel)
        elif high == 0x80:
            pitch = track[i]; i += 2
            if pitch in notes_on:
                on_tick, on_vel = notes_on.pop(pitch)
                raw_notes.append((on_tick, abs_tick, pitch, on_vel))
        elif high in (0xA0, 0xB0, 0xE0):
            i += 2
        elif high in (0xC0, 0xD0):
            i += 1

    # Convert ticks -> seconds with the (last seen) tempo
    sec_per_tick = tempo_us / 1_000_000.0 / ppq
    notes = []
    for on_t, off_t, pitch, vel in raw_notes:
        on_s = on_t * sec_per_tick
        dur_s = max(0.05, (off_t - on_t) * sec_per_tick)
        notes.append((on_s, dur_s, pitch, vel))
    return notes, tempo_us


# ---------------------------------------------------------------------------
# Tiny synth: piano-ish additive tone with exponential decay
# ---------------------------------------------------------------------------

def midi_to_hz(m: int) -> float:
    return 440.0 * 2.0 ** ((m - 69) / 12.0)


def piano_voice(freq: float, dur: float, vel: int) -> np.ndarray:
    n = int(SR * dur)
    t = np.arange(n) / SR
    # Short attack, long-ish exponential decay characteristic of a
    # damped piano string; harmonics weaken with partial number.
    amps  = [1.0, 0.55, 0.30, 0.16, 0.09, 0.05, 0.03]
    decay = [3.8, 5.2, 6.8, 8.5, 10.5, 13.0, 16.0]
    sig = np.zeros(n, dtype=np.float32)
    for k, (a, d) in enumerate(zip(amps, decay), start=1):
        phase = np.random.uniform(0, 2 * np.pi)   # inharmonic phase
        env   = np.exp(-d * t)
        sig  += a * env * np.sin(2 * np.pi * k * freq * t + phase)
    # Attack shaping: raise-cosine 8 ms
    attack = int(SR * 0.008)
    if n > attack:
        sig[:attack] *= 0.5 * (1 - np.cos(np.pi * np.arange(attack) / attack))
    # Release shaping: last 20 ms fade
    release = int(SR * 0.020)
    if n > release:
        sig[-release:] *= np.linspace(1.0, 0.0, release)
    return sig * (vel / 127.0) * 0.25


def render_to_wav(notes, out_wav: str, tail: float = 1.5) -> None:
    total_sec = max(n[0] + n[1] for n in notes) + tail
    total_samp = int(SR * total_sec)
    mix = np.zeros(total_samp, dtype=np.float32)
    for (on_s, dur_s, pitch, vel) in notes:
        freq = midi_to_hz(pitch)
        voice = piano_voice(freq, dur_s + 0.5, vel)
        start = int(SR * on_s)
        end   = start + len(voice)
        if end > total_samp:
            end = total_samp
            voice = voice[:end - start]
        mix[start:end] += voice

    # Normalise to -1 dB
    peak = np.max(np.abs(mix))
    if peak > 0:
        mix = mix * (0.891 / peak)

    # Convert to int16 and write
    pcm = np.clip(mix * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(out_wav, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())


def wav_to_mp3(wav_path: str, mp3_path: str) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-i", wav_path, "-codec:a", "libmp3lame",
         "-qscale:a", "4", mp3_path],
        check=True,
    )


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    midis = sorted(f for f in os.listdir(MIDI_DIR) if f.endswith(".mid"))
    for name in midis:
        src  = os.path.join(MIDI_DIR, name)
        stem = os.path.splitext(name)[0]
        wav  = os.path.join(OUT_DIR, stem + ".wav")
        mp3  = os.path.join(OUT_DIR, stem + ".mp3")
        notes, _ = parse_smf(src)
        print(f"[render] {name}: {len(notes)} notes")
        render_to_wav(notes, wav)
        wav_to_mp3(wav, mp3)
        try:
            os.remove(wav)
        except OSError:
            pass
        print(f"         -> {mp3}")


if __name__ == "__main__":
    main()
