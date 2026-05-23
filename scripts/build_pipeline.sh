#!/usr/bin/env bash
# Full Unum Avatar Build Pipeline v6
# Runs: COLMAP → Blender → Rhubarb demo speech → gltf-transform optimize
set -e
cd "$(dirname "$0")/.."

BASE="$(pwd)"
BLENDER="$(which blender 2>/dev/null || echo '')"
RHUBARB="$BASE/tools/rhubarb/Rhubarb-Lip-Sync-1.13.0-Linux/rhubarb"
PYTHON="$BASE/venv/bin/python"
NODE_MODULES="$BASE/tools/node_modules"

echo "=== Unum Avatar Full Build Pipeline ==="
echo ""

# ── Step 1: Multi-view COLMAP depth (optional) ────────────────────────────
echo "[1/5] Multi-view depth estimation..."
$PYTHON scripts/multiview_depth.py && echo "  ✓ COLMAP done" || echo "  ⚠ COLMAP skipped (will use InsightFace depth)"

# ── Step 2: Blender build ─────────────────────────────────────────────────
echo ""
echo "[2/5] Blender avatar build..."
if [ -n "$BLENDER" ]; then
    $BLENDER --background --python scripts/build_avatar_blender.py
    GLB_PATH="output/unum_avatar_v6.glb"
    echo "  ✓ Blender build done → $GLB_PATH"
else
    echo "  ⚠ Blender not found — falling back to Python builder v5..."
    $PYTHON scripts/build_head_v5.py
    GLB_PATH="output/unum_head_v5.glb"
fi

# ── Step 3: Optimize GLB with gltf-transform ─────────────────────────────
echo ""
echo "[3/5] Optimizing GLB..."
if [ -f "$NODE_MODULES/.bin/gltf-transform" ] || command -v npx &>/dev/null; then
    IN_GLB="$GLB_PATH"
    OUT_GLB="${GLB_PATH%.glb}_opt.glb"
    npx --prefix "$BASE/tools" @gltf-transform/cli optimize "$IN_GLB" "$OUT_GLB" \
        --texture-size 2048 2>/dev/null && \
    echo "  ✓ Optimized → $OUT_GLB" && \
    cp "$OUT_GLB" "$IN_GLB" && \
    echo "  ✓ Replaced original with optimized" || \
    echo "  ⚠ gltf-transform failed, using unoptimized"
else
    echo "  ⚠ gltf-transform not installed, skipping"
fi

# ── Step 4: Demo speech + Rhubarb lipsync ─────────────────────────────────
echo ""
echo "[4/5] Generating demo speech + Rhubarb lip-sync..."
DEMO_TEXT="Hello. I am Unum, your intelligent AI companion. I can see you, hear you, and understand you."
$PYTHON scripts/generate_lipsync_rhubarb.py "$DEMO_TEXT" && \
echo "  ✓ Rhubarb lip-sync done" || \
echo "  ⚠ Rhubarb failed, falling back to audio analysis..."
$PYTHON scripts/generate_lipsync.py "$DEMO_TEXT" 2>/dev/null || true

# ── Step 5: Update viewer to use v6 ───────────────────────────────────────
echo ""
echo "[5/5] Updating viewers..."
if [ -f "output/unum_avatar_v6.glb" ]; then
    TARGET_GLB="unum_avatar_v6.glb"
else
    TARGET_GLB="unum_head_v5.glb"
fi

for viewer in output/viewer.html output/talking.html; do
    if [ -f "$viewer" ]; then
        # Replace GLB filename reference
        sed -i "s|unum_head_v5\.glb|$TARGET_GLB|g; s|unum_avatar_v6\.glb|$TARGET_GLB|g" "$viewer"
        echo "  ✓ Updated $viewer → $TARGET_GLB"
    fi
done

echo ""
echo "=== Build complete! ==="
echo "  Viewer: http://localhost:8765/viewer.html"
echo "  Talking: http://localhost:8765/talking.html"
