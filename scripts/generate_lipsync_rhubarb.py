"""
Rhubarb-based lip-sync generator.
Flow: text → Piper TTS (WAV) → Rhubarb (phoneme cues) → blendshape JSON

Usage:
  python generate_lipsync_rhubarb.py "Hello, I am Unum."
  python generate_lipsync_rhubarb.py --file input.txt
"""

import sys, os, json, subprocess, argparse
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent.parent
OUTPUT = BASE / "output"
VOICES_DIR = BASE / "data" / "voices"
RHUBARB = BASE / "tools" / "rhubarb" / "Rhubarb-Lip-Sync-1.13.0-Linux" / "rhubarb"
PIPER   = BASE / "venv" / "bin" / "piper"

VOICE_MODEL = VOICES_DIR / "en_US-amy-medium.onnx"
VOICE_JSON  = VOICES_DIR / "en_US-amy-medium.onnx.json"

WAV_OUT     = OUTPUT / "speech.wav"
RHUBARB_OUT = OUTPUT / "speech_rhubarb.json"
LIPSYNC_OUT = OUTPUT / "speech_lipsync.json"

# ── Rhubarb mouth cue → blendshape weights ────────────────────────────────
# Based on Preston Blair phoneme groups
RHUBARB_MAP = {
    "A": {"mouthClose": 0.9, "jawOpen": 0.0,  "mouthOpen": 0.0},
    "B": {"mouthPucker": 0.75, "mouthFunnel": 0.35, "jawOpen": 0.05},
    "C": {"mouthOpen": 0.35, "jawOpen": 0.15},
    "D": {"mouthOpen": 0.50, "jawOpen": 0.25},
    "E": {"mouthOpen": 0.75, "jawOpen": 0.45},
    "F": {"mouthPucker": 0.40, "mouthFunnel": 0.75, "jawOpen": 0.05},
    "G": {"mouthSmileLeft": 0.30, "mouthSmileRight": 0.30, "mouthOpen": 0.25, "jawOpen": 0.10},
    "H": {"mouthOpen": 0.40, "jawOpen": 0.12},
    "X": {"mouthClose": 0.65, "jawOpen": 0.0},
}

ALL_SHAPES = [
    "jawOpen", "mouthClose", "mouthFunnel", "mouthPucker",
    "mouthSmileLeft", "mouthSmileRight", "mouthOpen",
    "eyeBlinkLeft", "eyeBlinkRight",
    "browInnerUp", "browDownLeft", "browDownRight",
]

FPS = 60
ATTACK_MS  = 40    # ms to blend in a new shape
RELEASE_MS = 110   # ms to fade out a shape


def text_to_wav(text: str) -> bool:
    if not VOICE_MODEL.exists():
        print(f"ERROR: Piper voice not found at {VOICE_MODEL}")
        return False
    WAV_OUT.parent.mkdir(exist_ok=True)
    cmd = [
        str(PIPER), "--model", str(VOICE_MODEL),
        "--output_file", str(WAV_OUT),
        "--sentence_silence", "0.3",
        "--length_scale", "1.05",
    ]
    result = subprocess.run(cmd, input=text.encode(), capture_output=True)
    if result.returncode != 0:
        print(f"Piper error: {result.stderr.decode()}")
        return False
    print(f"  WAV: {WAV_OUT} ({WAV_OUT.stat().st_size // 1024}KB)")
    return True


def run_rhubarb() -> bool:
    if not RHUBARB.exists():
        print(f"ERROR: Rhubarb not found at {RHUBARB}")
        return False
    cmd = [
        str(RHUBARB), "--machineReadable",
        "--exportFormat", "json",
        "--output", str(RHUBARB_OUT),
        "--recognizer", "phonetic",
        str(WAV_OUT)
    ]
    print("  Running Rhubarb phoneme analysis...")
    result = subprocess.run(cmd, capture_output=True, timeout=120)
    if result.returncode != 0:
        print(f"Rhubarb error: {result.stderr.decode()[:400]}")
        return False
    print(f"  Rhubarb output: {RHUBARB_OUT}")
    return True


def rhubarb_to_blendshapes() -> dict:
    with open(RHUBARB_OUT) as f:
        data = json.load(f)

    cues = data.get("mouthCues", [])
    if not cues:
        print("No mouth cues from Rhubarb")
        return None

    # Total duration
    duration = cues[-1]["end"]
    n_frames = int(duration * FPS) + 1

    # Per-frame blendshape weights (raw from cues)
    raw = {s: np.zeros(n_frames) for s in ALL_SHAPES}

    for cue in cues:
        key = cue["value"]
        weights = RHUBARB_MAP.get(key, {})
        start_f = int(cue["start"] * FPS)
        end_f   = min(int(cue["end"] * FPS), n_frames - 1)
        for shape, w in weights.items():
            if shape in raw:
                raw[shape][start_f:end_f+1] = w

    # Temporal smoothing: attack + release
    attack_frames  = max(1, int(ATTACK_MS  / 1000 * FPS))
    release_frames = max(1, int(RELEASE_MS / 1000 * FPS))

    smoothed = {}
    for shape in ALL_SHAPES:
        v = raw[shape]
        s = np.zeros_like(v)
        s[0] = v[0]
        for i in range(1, len(v)):
            if v[i] > s[i-1]:
                alpha = 1.0 / attack_frames
            else:
                alpha = 1.0 / release_frames
            s[i] = s[i-1] + alpha * (v[i] - s[i-1])
        smoothed[shape] = s

    # Build frames list
    frames = []
    for fi in range(n_frames):
        frame = {"time": round(fi / FPS, 4)}
        for shape in ALL_SHAPES:
            frame[shape] = round(float(smoothed[shape][fi]), 4)
        frames.append(frame)

    return {
        "fps": FPS,
        "duration": round(duration, 4),
        "frames": frames,
        "source": "rhubarb",
    }


def save_lipsync(data: dict):
    with open(LIPSYNC_OUT, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Lip-sync: {LIPSYNC_OUT} ({len(data['frames'])} frames, {data['duration']:.1f}s)")


def run(text: str):
    print("=== Rhubarb Lip-Sync Pipeline ===\n")

    print("[1/3] Generating speech (Piper TTS)...")
    if not text_to_wav(text):
        sys.exit(1)

    print("[2/3] Phoneme analysis (Rhubarb)...")
    if not run_rhubarb():
        print("  Falling back to audio-analysis lip-sync...")
        # Import fallback
        sys.path.insert(0, str(BASE / "scripts"))
        import generate_lipsync as fallback
        fallback.main_with_wav(str(WAV_OUT), str(LIPSYNC_OUT))
        return

    print("[3/3] Building blendshape timeline...")
    data = rhubarb_to_blendshapes()
    if data:
        save_lipsync(data)
        print(f"\n✓ Done. WAV + lipsync ready in output/")
    else:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="?", default="Hello, I am Unum, your AI companion.")
    parser.add_argument("--file", help="Read text from file")
    args = parser.parse_args()

    if args.file:
        with open(args.file) as f:
            text = f.read().strip()
    else:
        text = args.text

    os.chdir(BASE)
    run(text)


if __name__ == "__main__":
    main()
