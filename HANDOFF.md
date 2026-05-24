# UnumAvatar — Full Technical Handoff

> This document contains everything needed to continue improving the 3D avatar.
> Written for Codex / GPT-5.5 / any AI agent to pick up exactly where Claude left off.

---

## Repository

- **GitHub**: https://github.com/corpunum/UnumAvatar.git
- **Local path**: `/home/corp-unum/3d-avatar/`
- **Branch**: `master` (latest commit: `e481a5a`)
- **Python venv**: `/home/corp-unum/3d-avatar/venv/` (activate: `source venv/bin/activate`)

---

## Source Photos (Character Identity)

All source photos are in `/home/corp-unum/3d-avatar/input/`:

| File | Description |
|------|-------------|
| `unum-clean-front-neutral.png` | **Primary input** — front-facing neutral expression, used for texture baking |
| `unum-clean-front-v2.png` | Front face variant 2 |
| `unum-clean-front-v3.png` | Front face variant 3 |
| `unum-clean-left-profile.png` | Left profile view |
| `unum-clean-left-profile-v2.png` | Left profile variant 2 |
| `unum-clean-right-profile.png` | Right profile view |
| `unum-clean-back-head.png` | Back of head view |
| `unum-cyber-eye-closeup.png` | High-res eye closeup |
| `unum-cyber-markings-map.png` | Facial markings reference |

These photos can be used for:
- Multi-view reconstruction (left/right/back views available)
- Higher-res eye/iris textures from the closeup
- GPT image generation of additional angles, expressions, or open-mouth views

---

## Architecture & Pipeline

```
Photo (input/unum-clean-front-neutral.png)
    ↓
MediaPipe FaceLandmarker (478 landmarks + 52 blendshapes)
    ↓
Blender Python GLB builder (scripts/build_avatar_blender.py)
    → Face mesh with 12 morph targets
    → UV-mapped photo texture with mouth alpha hole
    → Photo-cropped eye patches (EyeballL, EyeballR)
    → Cranium, HairCap, HairCards, Ears, Neck, Shoulders
    → MouthCavity + Teeth meshes
    ↓
Output: output/unum_avatar_v6.glb (7.4 MB)
    ↓
Three.js r170 viewer (output/talking.html)
    + Rhubarb lipsync JSON (output/speech_lipsync.json)
    + Audio (output/speech.wav)
    ↓
Deterministic 30fps recording (scripts/record_speaking.py)
    → Playwright headless Chromium
    → canvas.toDataURL() per frame
    → ffmpeg PNG → MP4
    ↓
Output: output/avatar_speaking.mp4
```

---

## Key Files — What Each Does

### Scripts (in `scripts/`)

| File | Lines | Purpose |
|------|-------|---------|
| `build_avatar_blender.py` | 1380 | **Main GLB builder** — the core file. Reads photo, runs MediaPipe, builds face mesh with morph targets, bakes UV texture with mouth alpha hole, adds cranium/hair/ears/neck/shoulders/mouth cavity/teeth, exports GLB |
| `record_speaking.py` | 92 | Deterministic 30fps video recorder using Playwright + ffmpeg |
| `serve_avatar.py` | 79 | Simple HTTP server for local dev (port 8899) |
| `generate_lipsync_rhubarb.py` | 198 | Generates lipsync JSON from WAV using Rhubarb |
| `generate_lipsync.py` | 190 | Fallback lipsync generator |
| `build_head_v5.py` | 889 | Older mesh builder (superseded by build_avatar_blender.py) |
| `build_pipeline.sh` | — | Shell pipeline orchestrator |

### Output (in `output/`)

| File | Purpose |
|------|---------|
| `talking.html` | **Three.js viewer** — loads GLB, applies materials, viseme resolver, lipsync playback, gaze drift, head sway, deterministic render function |
| `viewer.html` | Interactive blendshape slider viewer |
| `unum_avatar_v6.glb` | Latest built GLB (face + all geometry + morph targets) |
| `speech_lipsync.json` | Rhubarb lipsync data (200 frames @ 60fps → 3.33s) |
| `speech.wav` | Speech audio file |
| `avatar_speaking.mp4` | Latest recorded video |
| `unum_skin_v6.png` | Baked face texture (UV-mapped photo with alpha mouth hole) |
| `eye_patch_L.png` / `eye_patch_R.png` | Cropped photo eye patches |
| `hair_strand_texture.png` | Procedural hair strand texture (256x512 RGBA) |
| `face_data.json` | Cached MediaPipe landmark + blendshape data |

### Tools (in `tools/`, gitignored)

| Tool | Path |
|------|------|
| Blender 4.2.9 | `tools/blender-4.2.9-linux-x64/blender` |
| Rhubarb Lip Sync | `tools/rhubarb/rhubarb` |

### Data (in `data/`, gitignored)

| File | Purpose |
|------|---------|
| `face_landmarker.task` | MediaPipe FaceLandmarker model |
| `voices/` | TTS voice data |

---

## How to Run

### 1. Rebuild the avatar GLB
```bash
cd /home/corp-unum/3d-avatar
source venv/bin/activate
tools/blender-4.2.9-linux-x64/blender --background --python scripts/build_avatar_blender.py
```

### 2. Start local server
```bash
python3 -m http.server 8899 --directory output
```

### 3. View in browser
```
http://localhost:8899/talking.html
```

### 4. Record video
```bash
python3 scripts/record_speaking.py
# Requires: server running on :8899, speech.wav + speech_lipsync.json in output/
# Output: output/avatar_speaking.mp4
```

### 5. Generate lipsync from new audio
```bash
python3 scripts/generate_lipsync_rhubarb.py output/speech.wav output/speech_lipsync.json
```

---

## Live Pages & Ports

| URL | Purpose |
|-----|---------|
| `http://localhost:8899/talking.html` | Main avatar viewer with lipsync playback |
| `http://localhost:8899/viewer.html` | Interactive blendshape slider debug viewer |
| `http://localhost:8899/` | File listing of output/ |

Server: `python3 -m http.server 8899 --directory /home/corp-unum/3d-avatar/output`

---

## GLB Structure (unum_avatar_v6.glb)

Meshes in the GLB and their roles:

| Mesh Name | Material | Purpose |
|-----------|----------|---------|
| `Face` | Photo UV texture (MeshBasicMaterial, unlit) | Main face with 12 morph targets |
| `EyeballL` / `EyeballR` | Photo eye patch (MeshBasicMaterial, transparent) | Cropped eye regions from photo |
| `MouthCavity` | Dark interior (#1A0804, MeshBasicMaterial) | Dark mouth interior visible when jaw opens |
| `Teeth` | Off-white (#EBE8D8, MeshBasicMaterial) | Teeth strip behind upper lip |
| `Cranium` | Skin tone (#C37D5D, MeshStandardMaterial) | Ellipsoidal back-of-head |
| `HairCap` | Dark hair (#29241F, MeshStandardMaterial) | Smooth scalp cap (currently depthTest:false hack) |
| `HairCards` | Hair strand texture (MeshStandardMaterial, alphaTest) | Alpha-blended hair cards around hairline |
| `EarL` / `EarR` | Skin tone (#A86A50, MeshStandardMaterial) | Ear geometry with inner bowl |
| `Neck` | Desaturated skin (#8E4A3B, MeshStandardMaterial) | Tapered neck cylinder |
| `Shoulders` | Dark clothing (#2D2A30, MeshStandardMaterial) | Body silhouette grounding |

### Morph Targets (12 shape keys on Face mesh)

```
jawOpen, mouthClose, mouthFunnel, mouthPucker,
mouthSmileLeft, mouthSmileRight, mouthOpen,
eyeBlinkLeft, eyeBlinkRight,
browInnerUp, browDownLeft, browDownRight
```

Blendshape scale factor: `sm = 0.65` (in build_avatar_blender.py)

---

## Render Stack in talking.html

### Render Order
```
-1  Shoulders
 0  Neck
 1  Cranium, Ears
 3  HairCap (depthTest:false — paints over cranium)
 4  HairCards (alphaTest:0.35)
 5  MouthCavity (visible when openAmount > 0.08)
 6  Teeth (visible when openAmount > 0.16)
10  Face (transparent, depthWrite:false)
12  EyeballL, EyeballR (transparent, depthWrite:false)
```

### Lighting
```javascript
HemisphereLight(0xffffff, 0x445066, 1.2)
DirectionalLight(0xffffff, 1.0) @ (20, 25, 100)   // key
DirectionalLight(0x88aaff, 0.45) @ (-25, 10, -30)  // rim
```

### Viseme Resolver (speech lipsync → morph weights)
```javascript
jawOpen  = clamp(pow(rawJaw, 0.8) * 2.4, 0, 1)   // gamma-lifted gain
mouthOpen = clamp(pow(rawOpen, 0.8) * 2.0, 0, 1)
closeSignal = rawClose * max(0, 1 - openSignal * 1.5)  // suppressed during open
```

### Micro-animations
- **Gaze drift**: Target-based smooth saccades, retarget every 0.4-1.4s, per-eye asymmetry
- **Head sway**: Sub-1-degree yaw (0.7°), roll (0.35°), pitch (0.25°) sinusoidal
- **Blink**: Every 3-5s, fast close (40%) + slow open (60%)
- **Breathing**: Tiny jawOpen oscillation during idle

### Deterministic Render
```javascript
window.renderFrameAt(t)  // renders one frame at time t, used by record_speaking.py
```

---

## Current Known Issues & What to Fix Next

### PRIORITY: Simplify to head-only (remove neck/shoulders)

The user wants to focus on **head quality only**. Remove:
- Neck mesh and `add_neck()` function
- Shoulders mesh and `add_shoulders()` function
- Their material handlers in talking.html

Keep: Face, Cranium, HairCap, HairCards, Ears, MouthCavity, Teeth, Eyeballs

### Remaining Quality Issues (in priority order)

1. **Hair cards not visible** — currently blend into the dark hair cap. Cards need to extend beyond the cap silhouette edge, especially at the hairline and temples, to break the smooth helmet look. The procedural strand texture exists at `output/hair_strand_texture.png` (256x512 RGBA, 80% coverage).

2. **Mouth texture tearing** — at high jaw-open, the photo lip texture stretches into vertical artifacts. Need a stable inner-mouth alpha mask that doesn't distort. Consider a separate lower-jaw texture strip or a more aggressive alpha cutoff at the lip crease.

3. **Teeth not convincingly staged** — teeth mesh exists behind upper lip but doesn't read as natural. Needs better placement (behind upper lip, slightly forward of cavity), possibly separate upper/lower teeth rows, and proper reveal at different jaw-open thresholds.

4. **HairCap depthTest:false hack** — causes compositing weirdness. Should use proper depth with the hair cap geometry fully outside the cranium. Fix by scaling the cap slightly larger than cranium and enabling normal depthTest.

5. **Face/cranium seam at temples** — the flat cranium color (#C37D5D) still differs from the photo texture at the boundary. A vertex color gradient on the cranium's first ring sampling the adjacent face boundary would fix this.

6. **Ears could use more detail** — current is outer rim + inner bowl. A helix ridge and tragus would make them more convincing.

7. **Static photo eyes** — gaze drift moves the whole patch. Better: extract iris-only sublayer from photo, animate just the iris, keep eyelids/lashes static from the photo.

### GPT Image Generation Opportunities

Codex can use GPT image 2.0 to generate:
- **Open-mouth expression** from the front-neutral photo → use as a separate jaw-open texture to avoid stretch artifacts
- **Closed-eye expression** → better blink texture instead of squishing via morph targets
- **Side-angle views** from the front photo → better ear/temple textures
- **Hair texture detail** → realistic strand textures instead of procedural noise
- **Teeth reference** → generate a teeth-showing smile view for teeth texture

---

## build_avatar_blender.py — Key Functions

| Function | Line | Purpose |
|----------|------|---------|
| `load_face_data()` | ~50 | Load/cache MediaPipe landmarks + blendshapes |
| `build_face_mesh()` | ~120 | Build face mesh from 468 landmarks with UV mapping |
| `add_shape_keys()` | ~280 | Create 12 morph target shape keys from blendshape deltas |
| `bake_texture()` | ~400 | UV-map photo to face mesh, punch transparent mouth alpha hole |
| `add_mouth_cavity()` | ~568 | Dark interior mesh behind lips |
| `add_teeth()` | ~628 | Teeth strip behind upper lip |
| `add_ears()` | ~671 | Ear geometry at landmarks 234/454 |
| `add_hair_cap()` | ~743 | Ellipsoidal scalp cap with per-longitude hairline |
| `add_hair_cards()` | ~855 | Alpha-textured hair cards around hairline |
| `add_cranium()` | ~960 | Ellipsoidal cranium stitched to face boundary |
| `add_neck()` | ~1020 | Tapered neck (TO BE REMOVED) |
| `add_shoulders()` | ~1100 | Body silhouette (TO BE REMOVED) |
| `add_eyeball()` | ~500 | Photo eye patch quad at iris landmarks |
| `place_eyeballs()` | ~540 | Position eyeballs using MediaPipe iris centers (468, 473) |
| `get_avg_skin_color()` | ~1090 | Sample average skin color from texture |
| `main()` | ~1200 | Orchestrator: calls all functions in order |

### Key Parameters
```python
sm = 0.65                    # Blendshape morph scale factor
INNER_UP = [78,191,80,81,82,13,312,311,310,415,308]  # Inner upper lip landmarks
INNER_LO = [78,95,88,178,87,14,317,402,318,324]      # Inner lower lip landmarks
mh_t = 14.0                 # Mouth alpha hole height (28px total)
```

### MediaPipe Landmark Reference
```
10   = forehead top
152  = chin bottom
234  = left cheek (face width left)
454  = right cheek (face width right)
468  = left iris center
473  = right iris center
33,133   = left eye corners
263,362  = right eye corners
```

---

## Dependencies

```
Python packages (in venv):
  mediapipe, opencv-python, numpy, Pillow, playwright

System tools:
  Blender 4.2.9 (tools/blender-4.2.9-linux-x64/blender)
  Rhubarb Lip Sync (tools/rhubarb/rhubarb)
  ffmpeg (system)
  Chromium (installed via playwright)

Three.js r170 (CDN in talking.html):
  https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js
  https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/controls/OrbitControls.js
  https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/loaders/GLTFLoader.js
```

---

## Git History (most recent first)

```
e481a5a v8 major overhaul: anatomy, lighting, micro-animation per Codex review
38ca296 Add shoulders, gaze drift, head sway; shrink neck; fix morphTargets warning
afec870 Polish mouth opening, neck shape, hairline, and cranium seam
921be6e Fix off-center neck, improve hair framing and neck color match
fd41582 Improve jaw opening, hair cap, ears, mouth cavity, and rendering fixes
566041d Fix rest-pose lip gap, thin mouth slit, expose scene/mesh to window
c9a3fa4 Fix mouth opening, add hair/ears/neck, landmark-based mouth hole
bd08cb9 Initial commit: photorealistic talking 3D avatar
```

---

## Quick Verification After Changes

```bash
# 1. Rebuild
tools/blender-4.2.9-linux-x64/blender --background --python scripts/build_avatar_blender.py

# 2. Start server (if not running)
python3 -m http.server 8899 --directory output &

# 3. Record video
python3 scripts/record_speaking.py

# 4. Extract frames for visual inspection
ffmpeg -y -i output/avatar_speaking.mp4 -frames:v 1 /tmp/rest.png
ffmpeg -y -i output/avatar_speaking.mp4 -vf "select='eq(n\,14)'" -vsync 0 -frames:v 1 /tmp/peak.png

# 5. View in browser
# http://localhost:8899/talking.html
```
