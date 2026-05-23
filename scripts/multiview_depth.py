"""
Multi-view depth estimation using pycolmap + multiple face photos.
Produces per-vertex depth corrections for the face mesh.
Uses: front, left profile, right profile, left+right 3/4 if available.
"""
import os, sys, shutil, json
import numpy as np
import cv2
import pycolmap
from pathlib import Path

INPUT_DIR = Path("input")
COLMAP_WS = Path("tools/colmap_workspace")
OUTPUT_DIR = Path("output")

PHOTOS = {
    "front":         "unum-clean-front-v2.png",
    "left_profile":  "unum-clean-left-profile.png",
    "right_profile": "unum-clean-right-profile.png",
    "left_34":       "unum-clean-left-34.png",      # optional
    "right_34":      "unum-clean-right-34.png",     # optional
    "front_orig":    "unum-clean-front-neutral.png",
}

def run():
    ws = COLMAP_WS
    img_dir = ws / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Copy available photos
    available = []
    for name, fname in PHOTOS.items():
        src = INPUT_DIR / fname
        if src.exists():
            dst = img_dir / f"{name}.png"
            shutil.copy(src, dst)
            available.append(name)
            print(f"  + {name}")
        else:
            print(f"  - {name} (missing)")

    if len(available) < 2:
        print("Need at least 2 photos for multi-view. Skipping COLMAP.")
        return False

    db_path = ws / "database.db"
    if db_path.exists():
        db_path.unlink()

    print("\n[COLMAP] Extracting features...")
    reader_opts = pycolmap.ImageReaderOptions()
    reader_opts.camera_model = "SIMPLE_PINHOLE"
    pycolmap.extract_features(
        database_path=str(db_path),
        image_path=str(img_dir),
        reader_options=reader_opts,
    )

    print("[COLMAP] Matching features...")
    pycolmap.match_exhaustive(database_path=str(db_path))

    sparse_dir = ws / "sparse"
    sparse_dir.mkdir(exist_ok=True)
    print("[COLMAP] Sparse reconstruction...")
    maps = pycolmap.incremental_mapping(
        database_path=str(db_path),
        image_path=str(img_dir),
        output_path=str(sparse_dir),
    )

    if not maps:
        print("COLMAP reconstruction failed — no maps produced.")
        return False

    print(f"  {len(maps)} reconstruction(s)")
    recon = maps[0]

    # Export sparse point cloud for depth reference
    pts = []
    for pt3d in recon.points3D.values():
        pts.append([pt3d.xyz[0], pt3d.xyz[1], pt3d.xyz[2], pt3d.error])

    pts = np.array(pts, dtype=np.float32)
    out = {"points": pts.tolist(), "n_images": len(recon.images)}
    with open(OUTPUT_DIR / "colmap_points.json", "w") as f:
        json.dump(out, f)

    print(f"\n  Saved {len(pts)} 3D points → output/colmap_points.json")
    return True


if __name__ == "__main__":
    os.chdir(Path(__file__).parent.parent)
    success = run()
    if success:
        print("\n✓ Multi-view depth estimation done.")
    else:
        print("\n⚠ COLMAP failed — will use InsightFace depth only.")
