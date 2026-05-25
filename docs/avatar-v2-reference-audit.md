# Avatar V2 Reference Audit

Date: 2026-05-25
Project root: `/home/corp-unum/3d-avatar`

## Source Datasets
- Videos: `/home/corp-unum/UNUM VIDEOS`
- Photos: `/home/corp-unum/Pictures/Character for Unum`

## Inventory Summary
- Video files: `10`
- Photo/image files: `30`

## Video Inventory (resolution, duration)

| File | Resolution | FPS | Duration (s) | Size (MB) |
|---|---:|---:|---:|---:|
| Unum_Agent_AI.mp4 | 856x1072 | 24 | 10.083 | 15.3 |
| Unum_Agent_AI (1).mp4 | 856x1072 | 24 | 10.083 | 15.3 |
| Unum_Agent_AI (2).mp4 | 856x1072 | 24 | 10.083 | 15.3 |
| kling_20260428_作品_shot_1_5s__4404_0 (1).mp4 | 1440x1440 | 24 | 15.042 | 32.3 |
| kling_20260428_作品_form__her__4545_0 (2).mp4 | 784x1176 | 24 | 15.042 | 15.1 |
| kling_20260428_作品_form__her__4590_0 (1).mp4 | 784x1176 | 24 | 15.042 | 13.6 |
| kling_20260429_作品_Close_up_s_5101_0.mp4 | 784x1176 | 24 | 5.042 | 6.5 |
| kling_20260429_作品_Dancing_se_1027_0.mp4 | 856x1072 | 24 | 10.042 | 10.2 |
| kling_20260502_作品_一位身穿白色连衣裙和_139_0.mp4 | 856x1072 | 24 | 11.042 | 9.6 |
| kling_20260428_Build_Avatar_Smiling_na_4482_0 (1).mp4 | 960x960 | 30 | 7.200 | 4.2 |

Notes:
- `Unum_Agent_AI*.mp4` are likely duplicates (same duration/size/resolution).
- The highest-detail reference clip is `kling_20260428_作品_shot_1_5s__4404_0 (1).mp4` at 1440x1440.

## Image Inventory Highlights

Primary curated identity set (recommended for base asset build):
- `unum-clean-front-neutral.png` (1254x1254)
- `unum-clean-left-profile.png` (1254x1254)
- `unum-clean-right-profile.png` (1254x1254)
- `unum-clean-back-head.png` (1254x1254)
- `unum-cyber-eye-closeup.png` (1254x1254)
- `unum-cyber-markings-map.png` (1254x1254)

Additional generated references exist across:
- 1491x1055 landscape
- 1122x1402 portrait
- 1024x1536 portrait

## Gaps Against V2 Requirements

Current local dataset is strong for:
- identity likeness
- eye color/shape
- hairline/back hair bun reference
- side profile geometry

Current local dataset is weak for:
- explicit viseme reference set (`A/E/O/M/F/L/S`)
- teeth/tongue clarity during speech
- consistent close-up mouth-inside frames

## Required Extraction Pass (before modeling changes)

Run this pass to generate deterministic keyframes into
`references/avatar_v2/video_keyframes`:

```bash
cd /home/corp-unum/3d-avatar
mkdir -p references/avatar_v2/video_keyframes

# High-detail square clip
ffmpeg -hide_banner -loglevel error -y \
  -i "/home/corp-unum/UNUM VIDEOS/kling_20260428_作品_shot_1_5s__4404_0 (1).mp4" \
  -vf "fps=8" references/avatar_v2/video_keyframes/square_%04d.png

# Close-up portrait clip
ffmpeg -hide_banner -loglevel error -y \
  -i "/home/corp-unum/UNUM VIDEOS/kling_20260429_作品_Close_up_s_5101_0.mp4" \
  -vf "fps=10" references/avatar_v2/video_keyframes/closeup_%04d.png
```

Then manually curate and copy selected frames into:
- `references/avatar_v2/neutral`
- `references/avatar_v2/eyes`
- `references/avatar_v2/mouth_visemes`
- `references/avatar_v2/hair`
- `references/avatar_v2/profile`

## Immediate Recommendation

Use the `unum-clean-*` images as the canonical geometry/texture identity set,
then use video keyframes strictly for motion realism calibration (blink timing,
mouth range, teeth visibility, head micro-motion).

Do not continue patching eye/mouth overlays as a final architecture.
