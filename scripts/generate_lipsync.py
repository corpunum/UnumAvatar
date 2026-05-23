"""
Generate lip-sync animation data from text.
Pipeline: Text → Piper TTS (WAV) → Audio analysis → Viseme timeline (JSON)

Viseme mapping (simplified Oculus/ARKit style):
  - silence: mouth closed
  - AA: jaw open wide (vowels a, o)
  - EE: mouth wide narrow (vowels e, i)
  - OO: mouth round (vowels u, o)
  - FF: lower lip under upper teeth (f, v)
  - TH: tongue between teeth (th)
  - PP: lips together (p, b, m)
  - SS: teeth together (s, z, c)
  - CH: rounded open (sh, ch, j)

Maps to blendshapes: jawOpen, mouthOpen, mouthFunnel, mouthPucker,
  mouthSmileLeft, mouthSmileRight, mouthClose
"""

import json
import os
import struct
import subprocess
import sys
import wave

import numpy as np


PIPER_MODEL = "data/voices/en_US-amy-medium.onnx"
OUTPUT_DIR = "output"
FRAME_MS = 33  # ~30fps


def text_to_wav(text, wav_path):
    proc = subprocess.run(
        ["python3", "-m", "piper", "--model", PIPER_MODEL, "--output-file", wav_path],
        input=text.encode(), capture_output=True
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Piper failed: {proc.stderr.decode()}")
    return wav_path


def read_wav(wav_path):
    with wave.open(wav_path, 'rb') as w:
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, sr


def analyze_audio(samples, sr, frame_ms=FRAME_MS):
    """Extract per-frame energy and spectral features for viseme estimation."""
    frame_size = int(sr * frame_ms / 1000)
    n_frames = len(samples) // frame_size

    frames = []
    for i in range(n_frames):
        chunk = samples[i * frame_size:(i + 1) * frame_size]
        energy = np.sqrt(np.mean(chunk ** 2))

        # Zero crossing rate - distinguishes voiced vs unvoiced
        zcr = np.sum(np.abs(np.diff(np.sign(chunk)))) / (2 * len(chunk))

        # Simple spectral centroid approximation
        fft = np.abs(np.fft.rfft(chunk))
        freqs = np.fft.rfftfreq(len(chunk), 1.0/sr)
        if fft.sum() > 0:
            centroid = np.sum(freqs * fft) / np.sum(fft)
        else:
            centroid = 0

        frames.append({
            "time": i * frame_ms / 1000.0,
            "energy": float(energy),
            "zcr": float(zcr),
            "centroid": float(centroid),
        })

    return frames


def frames_to_blendshapes(frames):
    """Convert audio analysis frames to blendshape weights."""
    if not frames:
        return []

    max_energy = max(f["energy"] for f in frames) or 1.0

    timeline = []
    for f in frames:
        e = f["energy"] / max_energy
        zcr = f["zcr"]
        cent = f["centroid"]

        # Silence threshold
        if e < 0.05:
            timeline.append({
                "time": f["time"],
                "jawOpen": 0, "mouthOpen": 0, "mouthFunnel": 0,
                "mouthPucker": 0, "mouthSmileLeft": 0, "mouthSmileRight": 0,
                "mouthClose": 0.3,
            })
            continue

        jaw = min(1.0, e * 1.2)
        mouth_open = min(1.0, e * 0.9)

        # High centroid + high zcr = sibilants (s, z, sh) → teeth together, small opening
        # Low centroid + low zcr = vowels → jaw open
        # Medium + medium = consonants
        funnel = 0
        pucker = 0
        smile = 0

        if cent > 3000 and zcr > 0.15:
            # Sibilant: small mouth, teeth visible
            jaw *= 0.3
            mouth_open *= 0.2
            smile = 0.2
        elif cent < 1500 and zcr < 0.1:
            # Open vowel: wide open
            jaw *= 1.0
            mouth_open *= 1.0
            if cent < 800:
                # Low vowel (ah, oh) → rounder
                funnel = e * 0.4
            else:
                # Higher vowel (ee) → wider
                smile = e * 0.3
        elif cent > 2000:
            # Mid consonant
            jaw *= 0.5
            pucker = e * 0.3
        else:
            # Nasal/liquid
            jaw *= 0.6
            mouth_open *= 0.5

        timeline.append({
            "time": f["time"],
            "jawOpen": round(jaw, 3),
            "mouthOpen": round(mouth_open, 3),
            "mouthFunnel": round(funnel, 3),
            "mouthPucker": round(pucker, 3),
            "mouthSmileLeft": round(smile, 3),
            "mouthSmileRight": round(smile, 3),
            "mouthClose": 0,
        })

    return timeline


def generate(text, output_name="speech"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    wav_path = os.path.join(OUTPUT_DIR, f"{output_name}.wav")
    json_path = os.path.join(OUTPUT_DIR, f"{output_name}_lipsync.json")

    print(f"[1/3] Generating speech: '{text[:60]}...'")
    text_to_wav(text, wav_path)

    print(f"[2/3] Analyzing audio...")
    samples, sr = read_wav(wav_path)
    frames = analyze_audio(samples, sr)

    print(f"[3/3] Mapping to blendshapes...")
    timeline = frames_to_blendshapes(frames)

    result = {
        "audio": f"{output_name}.wav",
        "duration": len(samples) / sr,
        "fps": 1000 / FRAME_MS,
        "frames": timeline
    }

    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"  WAV: {wav_path} ({len(samples)/sr:.1f}s)")
    print(f"  Lipsync: {json_path} ({len(timeline)} frames)")
    return result


if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else \
        "Hello, I am Unum, your intelligent companion. I can help you think, create, and solve problems. What would you like to explore together?"
    generate(text)
