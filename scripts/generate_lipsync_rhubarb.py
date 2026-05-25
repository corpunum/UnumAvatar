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

# ── Rhubarb mouth cue → viseme + blendshape weights ───────────────────────
# Rhubarb cues (A..H,X) are first mapped to an Oculus-style viseme layer.
RHUBARB_VISEME_MAP = {
    "A": {"viseme_sil": 1.0, "viseme_PP": 0.45},                            # closed/rest
    "B": {"viseme_PP": 1.0},                                                # M/B/P
    "C": {"viseme_E": 0.55, "viseme_I": 0.35, "viseme_DD": 0.25},          # EH/AE-ish
    "D": {"viseme_aa": 0.75, "viseme_DD": 0.35, "viseme_nn": 0.20},        # AA
    "E": {"viseme_aa": 1.0, "viseme_E": 0.35},                              # wide open
    "F": {"viseme_O": 0.85, "viseme_U": 0.55, "viseme_FF": 0.50},          # O/U/WQ
    "G": {"viseme_FF": 1.0, "viseme_SS": 0.35, "viseme_CH": 0.20},         # FV
    "H": {"viseme_DD": 0.75, "viseme_nn": 0.40, "viseme_TH": 0.20},        # L / tongue-forward
    "X": {"viseme_sil": 1.0},
}

ALL_SHAPES = [
    "viseme_sil", "viseme_PP", "viseme_FF", "viseme_TH", "viseme_DD",
    "viseme_kk", "viseme_CH", "viseme_SS", "viseme_nn", "viseme_RR",
    "viseme_aa", "viseme_E", "viseme_I", "viseme_O", "viseme_U",
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
        "--sentence_silence", "0.18",
        "--length_scale", "0.93",
        "--noise_scale", "0.52",
        "--noise_w_scale", "0.62",
        "--volume", "1.12",
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
        visemes = RHUBARB_VISEME_MAP.get(key, {"viseme_sil": 1.0})
        start_f = int(cue["start"] * FPS)
        end_f   = min(int(cue["end"] * FPS), n_frames - 1)
        for shape, w in visemes.items():
            if shape in raw:
                raw[shape][start_f:end_f+1] = w

    # Convert visemes into the currently available runtime blendshape controls.
    # This keeps compatibility with the existing GLB while giving better
    # phoneme separation and stronger closed/open dominance behavior.
    for fi in range(n_frames):
        vis = {k: raw[k][fi] for k in ALL_SHAPES if k.startswith("viseme_")}
        pp = vis.get("viseme_PP", 0.0)
        sil = vis.get("viseme_sil", 0.0)
        aa = vis.get("viseme_aa", 0.0)
        ee = max(vis.get("viseme_E", 0.0), vis.get("viseme_I", 0.0))
        oo = max(vis.get("viseme_O", 0.0), vis.get("viseme_U", 0.0))
        ff = vis.get("viseme_FF", 0.0)
        dd = max(vis.get("viseme_DD", 0.0), vis.get("viseme_TH", 0.0), vis.get("viseme_nn", 0.0))
        ss = max(vis.get("viseme_SS", 0.0), vis.get("viseme_CH", 0.0), vis.get("viseme_kk", 0.0))

        jaw_open = min(1.0, aa * 0.68 + ee * 0.46 + dd * 0.28)
        mouth_open = min(1.0, aa * 0.76 + ee * 0.52 + ss * 0.26 + dd * 0.22)
        mouth_funnel = min(1.0, oo * 0.88 + ff * 0.30)
        mouth_pucker = min(1.0, oo * 0.56 + pp * 0.45)
        smile = min(1.0, ss * 0.32 + ee * 0.20)
        mouth_close = min(1.0, sil * 0.82 + pp * 0.95)

        # Dominance rules: bilabials close strongly; open vowels suppress close.
        if pp > 0.58:
            jaw_open *= 0.25
            mouth_open *= 0.18
            mouth_close = max(mouth_close, 0.88)
        if max(aa, ee) > 0.45:
            mouth_close *= 0.18

        raw["jawOpen"][fi] = jaw_open
        raw["mouthOpen"][fi] = mouth_open
        raw["mouthFunnel"][fi] = mouth_funnel
        raw["mouthPucker"][fi] = mouth_pucker
        raw["mouthSmileLeft"][fi] = smile
        raw["mouthSmileRight"][fi] = smile
        raw["mouthClose"][fi] = mouth_close

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
