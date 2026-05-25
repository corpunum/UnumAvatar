# Avatar V2 Implementation Plan (Local First)

Date: 2026-05-25
Status: Execution started

## Objective
Build a realistic local-first 3D OpenUnum default avatar that can speak with no internet dependency at runtime.

## Non-Goals
- Do not keep iterating on the current photo-overlay eye/mouth hack as final architecture.
- Do not require cloud APIs for normal end-user runtime.

## Architecture Decision

Use a rigged 3D head with ARKit-compatible blendshapes and real facial components:
- real eyes (no eye patch overlays)
- real mouth interior (teeth, tongue, cavity)
- hair cards / bun volume
- ARKit/FACS-like blendshape contract

Runtime path:
1. local TTS audio
2. local blendshape timeline generation
3. local GLB playback in Three.js

## Secure NVIDIA Usage (authoring/acceleration only)

NVIDIA NIM can be used to accelerate authoring and benchmark quality, but never hardcode API keys.

Required pattern:
```bash
export NVIDIA_API_KEY='...'
```

Rules:
- never commit keys
- never print keys in logs/docs
- pass via environment variables only

## Work Phases

### Phase 0 - Baseline Freeze
- Keep current prototype outputs for A/B comparisons.
- Preserve `output/avatar_speaking.mp4` as V1 baseline.

### Phase 1 - Reference Curation (in progress)
- Completed: dataset inventory and initial audit docs.
- Next: extract/curate viseme and motion keyframes from videos.

### Phase 2 - Runtime Contract
Create schemas for blendshape clips:
- `schemas/avatar_blendshape_frame.schema.json`
- `schemas/avatar_clip.schema.json`

Frame format:
```json
{
  "time": 0.033,
  "blendshapes": {
    "jawOpen": 0.12,
    "mouthFunnel": 0.21,
    "eyeBlinkLeft": 0.0,
    "eyeBlinkRight": 0.0
  }
}
```

### Phase 3 - Eye System Replacement
- remove photo eye patches and duplicate iris overlays
- add single coherent eye rig
- drive blinks through eyelid blendshapes

### Phase 4 - Mouth System Replacement
- remove dependency on baked mouth-hole + aperture hack as final system
- build real lip/teeth/tongue/cavity geometry
- implement viseme-driven blendshapes for `A/E/O/M/F/L/S` plus neutral/blink/smile

### Phase 5 - Hair
- keep cap only as fallback mass
- prioritize strand cards around hairline + temples + bun silhouette
- ensure no helmet-like front edge

### Phase 6 - Audio-to-Blendshape Backends
Tiered local-first strategy:
1. high-quality local backend (GPU path)
2. lightweight offline viseme fallback (CPU path)

### Phase 7 - Viewer/QA
Add deterministic QA tools:
- `debug_blendshapes.html` slider page
- frame-contact-sheet generation
- fixed timestamp A/B captures

## First Execution Commands For Codex 5.3

```bash
cd /home/corp-unum/3d-avatar

# 1) curate video keyframes
mkdir -p references/avatar_v2/video_keyframes
ffmpeg -hide_banner -loglevel error -y \
  -i "/home/corp-unum/UNUM VIDEOS/kling_20260428_作品_shot_1_5s__4404_0 (1).mp4" \
  -vf "fps=8" references/avatar_v2/video_keyframes/square_%04d.png
ffmpeg -hide_banner -loglevel error -y \
  -i "/home/corp-unum/UNUM VIDEOS/kling_20260429_作品_Close_up_s_5101_0.mp4" \
  -vf "fps=10" references/avatar_v2/video_keyframes/closeup_%04d.png

# 2) start schema contract
mkdir -p schemas
```

## Current Repo State Warning

Working tree is intentionally dirty with prior prototype changes in:
- `scripts/build_avatar_blender.py`
- `output/talking.html`
- `output/unum_avatar_v6.glb`
- `output/avatar_speaking.mp4`

Do not reset/clean unrelated changes. Build V2 incrementally and verify each step with deterministic captures.
