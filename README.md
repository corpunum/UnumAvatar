# UnumAvatar

A photorealistic talking 3D avatar built from a single front-facing photo using MediaPipe landmarks, Blender Python, Three.js, and Rhubarb Lip Sync.

## Features

- **Photo-realistic face** — MediaPipe 468-landmark mesh with UV-mapped photo texture
- **Animated jaw/mouth** — 12 blendshape morph targets (jawOpen, mouthOpen, mouthFunnel, mouthSmileLeft/Right, eyeBlink, browUp/Down)
- **Photo eyes** — Cropped eye patches from source photo, rendered as transparent quads
- **Hair cap** — Ellipsoid hair geometry with dark color matching photo
- **Ears, neck, cranium** — Complete head geometry
- **Rhubarb Lip Sync** — Phoneme-level lipsync from WAV audio
- **Deterministic 30fps recording** — Frame-by-frame render via `canvas.toDataURL()` → ffmpeg MP4

## Stack

- **MediaPipe FaceLandmarker** — 478 landmarks + 52 blendshapes
- **InsightFace / buffalo_l** — 3D depth from 68-point landmarks
- **Blender 4.2** (Python API) — GLB builder with subdivision + shape keys
- **Three.js r170** — WebGL renderer (MeshBasicMaterial for unlit photo quality)
- **Rhubarb Lip Sync** — phoneme → blendshape timeline
- **Playwright** — headless Chromium for deterministic frame capture
- **ffmpeg** — PNG sequence → MP4 assembly

## Quickstart

```bash
# 1. Build the avatar GLB
tools/blender-4.2.9-linux-x64/blender --background --python scripts/build_avatar_blender.py

# 2. Start local server
python3 -m http.server 8899 --directory output

# 3. View in browser
open http://localhost:8899/talking.html

# 4. Record a video (requires speech.wav + speech_lipsync.json in output/)
python3 scripts/record_speaking.py
```

## Key Files

| File | Description |
|------|-------------|
| `scripts/build_avatar_blender.py` | GLB builder — landmarks, texture bake, shape keys, hair/ears/neck |
| `scripts/record_speaking.py` | Deterministic 30fps video recorder |
| `output/talking.html` | Three.js viewer with lipsync playback |
| `output/viewer.html` | Interactive blendshape viewer |
| `output/unum_avatar_v6.glb` | Built avatar (not tracked — rebuild from script) |
| `output/speech_lipsync.json` | Rhubarb lipsync data (200 frames, 3.33s) |

## Architecture

```
Photo → MediaPipe landmarks → Blender GLB (face mesh + morph targets)
Audio → Rhubarb → lipsync JSON → Three.js animation loop → canvas frames → ffmpeg MP4
```

## Lipsync Pipeline

The lipsync JSON has per-frame blendshape weights. A **viseme resolver** in `talking.html` prevents `mouthClose` from canceling `jawOpen`:

```javascript
const openSignal = Math.max(rawJaw, rawOpen, rawFunnel * 0.5);
const closeSignal = rawClose * Math.max(0, 1.0 - openSignal * 1.5);
```
