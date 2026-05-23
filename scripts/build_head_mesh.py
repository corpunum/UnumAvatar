"""
Build a 3D head mesh from mediapipe face landmarks.
Uses the 478 landmarks + canonical face mesh topology to create
a textured, blendshape-ready GLB head model.
"""
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
import numpy as np
import json
import struct
import os

INPUT_IMAGE = "input/unum-clean-front-neutral.png"
MODEL_PATH = "data/face_landmarker.task"
OUTPUT_DIR = "output"

# MediaPipe canonical face mesh triangulation (468 base + 10 iris = 478 landmarks)
# We use the standard 468-point topology
FACE_MESH_TESSELATION = None

def get_face_mesh_triangles():
    """Extract triangle indices from mediapipe's face mesh connections."""
    from mediapipe.tasks.python.vision import FaceLandmarksConnections
    edges = set()
    for conn in FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION:
        edges.add((conn.start, conn.end))

    # Build adjacency and find triangles
    adj = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    triangles = set()
    for a, b in edges:
        common = adj.get(a, set()) & adj.get(b, set())
        for c in common:
            tri = tuple(sorted([a, b, c]))
            triangles.add(tri)

    return np.array(list(triangles), dtype=np.int32)


def detect_landmarks(image_path):
    """Detect face landmarks and blendshapes."""
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)
    image = mp.Image.create_from_file(image_path)
    result = landmarker.detect(image)
    landmarker.close()

    if not result.face_landmarks:
        raise RuntimeError("No face detected")

    lm = result.face_landmarks[0]
    pts = np.array([[l.x, l.y, l.z] for l in lm])

    blendshapes = {}
    if result.face_blendshapes:
        for b in result.face_blendshapes[0]:
            blendshapes[b.category_name] = b.score

    transform = None
    if result.facial_transformation_matrixes:
        transform = np.array(result.facial_transformation_matrixes[0])

    return pts, blendshapes, transform, image


def landmarks_to_3d(pts, image_width, image_height, scale=100.0):
    """Convert normalized landmarks to 3D coordinates (in mm-ish units).

    MediaPipe landmarks are normalized [0,1] for x,y and relative depth for z.
    We center the mesh and scale to reasonable 3D units.
    """
    verts = np.zeros_like(pts, dtype=np.float32)
    verts[:, 0] = (pts[:, 0] - 0.5) * scale  # X: left-right
    verts[:, 1] = -(pts[:, 1] - 0.5) * scale  # Y: up-down (flip for 3D)
    verts[:, 2] = -pts[:, 2] * scale * 0.5     # Z: depth (scale down, flip)
    return verts


def compute_uvs(pts):
    """Use the normalized x,y landmark positions as UV coordinates."""
    uvs = np.zeros((len(pts), 2), dtype=np.float32)
    uvs[:, 0] = pts[:, 0]       # U = normalized X
    uvs[:, 1] = 1.0 - pts[:, 1] # V = 1 - normalized Y (flip for UV space)
    return uvs


def compute_normals(verts, faces):
    """Compute per-vertex normals from face normals."""
    normals = np.zeros_like(verts)
    for face in faces:
        v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
        n = np.cross(v1 - v0, v2 - v0)
        norm = np.linalg.norm(n)
        if norm > 0:
            n /= norm
        normals[face[0]] += n
        normals[face[1]] += n
        normals[face[2]] += n

    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normals /= norms
    return normals.astype(np.float32)


def write_obj(filepath, verts, faces, uvs=None, normals=None):
    """Write OBJ file."""
    with open(filepath, 'w') as f:
        f.write("# Unum Avatar Head Mesh\n")
        f.write(f"# {len(verts)} vertices, {len(faces)} faces\n\n")

        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        if uvs is not None:
            for uv in uvs:
                f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")

        if normals is not None:
            for n in normals:
                f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")

        for face in faces:
            if uvs is not None and normals is not None:
                f.write(f"f {face[0]+1}/{face[0]+1}/{face[0]+1} {face[1]+1}/{face[1]+1}/{face[1]+1} {face[2]+1}/{face[2]+1}/{face[2]+1}\n")
            else:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def write_glb(filepath, verts, faces, uvs, normals, texture_path=None):
    """Write a binary glTF (.glb) file."""
    import cv2

    # Prepare binary buffer
    verts_f32 = verts.astype(np.float32)
    normals_f32 = normals.astype(np.float32)
    uvs_f32 = uvs.astype(np.float32)
    indices_u16 = faces.flatten().astype(np.uint16)

    verts_bytes = verts_f32.tobytes()
    normals_bytes = normals_f32.tobytes()
    uvs_bytes = uvs_f32.tobytes()
    indices_bytes = indices_u16.tobytes()

    # Load texture image as PNG bytes
    texture_bytes = b""
    if texture_path and os.path.exists(texture_path):
        with open(texture_path, 'rb') as f:
            texture_bytes = f.read()

    # Build buffer
    buffer_data = bytearray()

    # Accessor 0: indices
    indices_offset = len(buffer_data)
    buffer_data.extend(indices_bytes)
    # Pad to 4-byte boundary
    while len(buffer_data) % 4 != 0:
        buffer_data.extend(b'\x00')

    # Accessor 1: positions
    positions_offset = len(buffer_data)
    buffer_data.extend(verts_bytes)

    # Accessor 2: normals
    normals_offset = len(buffer_data)
    buffer_data.extend(normals_bytes)

    # Accessor 3: texcoords
    uvs_offset = len(buffer_data)
    buffer_data.extend(uvs_bytes)

    # Image data
    image_offset = len(buffer_data)
    buffer_data.extend(texture_bytes)

    total_buffer_len = len(buffer_data)

    # Build glTF JSON
    gltf = {
        "asset": {"version": "2.0", "generator": "unum-avatar-builder"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "UNumHead"}],
        "meshes": [{
            "primitives": [{
                "attributes": {
                    "POSITION": 1,
                    "NORMAL": 2,
                    "TEXCOORD_0": 3
                },
                "indices": 0,
                "material": 0
            }]
        }],
        "accessors": [
            {  # 0: indices
                "bufferView": 0,
                "componentType": 5123,  # UNSIGNED_SHORT
                "count": len(indices_u16),
                "type": "SCALAR",
                "max": [int(indices_u16.max())],
                "min": [int(indices_u16.min())]
            },
            {  # 1: positions
                "bufferView": 1,
                "componentType": 5126,  # FLOAT
                "count": len(verts_f32),
                "type": "VEC3",
                "max": verts_f32.max(axis=0).tolist(),
                "min": verts_f32.min(axis=0).tolist()
            },
            {  # 2: normals
                "bufferView": 2,
                "componentType": 5126,
                "count": len(normals_f32),
                "type": "VEC3"
            },
            {  # 3: texcoords
                "bufferView": 3,
                "componentType": 5126,
                "count": len(uvs_f32),
                "type": "VEC2"
            }
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": indices_offset, "byteLength": len(indices_bytes), "target": 34963},
            {"buffer": 0, "byteOffset": positions_offset, "byteLength": len(verts_bytes), "target": 34962, "byteStride": 12},
            {"buffer": 0, "byteOffset": normals_offset, "byteLength": len(normals_bytes), "target": 34962, "byteStride": 12},
            {"buffer": 0, "byteOffset": uvs_offset, "byteLength": len(uvs_bytes), "target": 34962, "byteStride": 8},
        ],
        "materials": [{
            "pbrMetallicRoughness": {
                "metallicFactor": 0.0,
                "roughnessFactor": 0.7,
            },
            "name": "skin",
            "doubleSided": True
        }],
        "buffers": [{"byteLength": total_buffer_len}]
    }

    # Add texture if available
    if texture_bytes:
        gltf["bufferViews"].append({
            "buffer": 0,
            "byteOffset": image_offset,
            "byteLength": len(texture_bytes)
        })
        gltf["images"] = [{"bufferView": 4, "mimeType": "image/png"}]
        gltf["textures"] = [{"source": 0}]
        gltf["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"] = {"index": 0}

    # Serialize JSON
    json_str = json.dumps(gltf, separators=(',', ':'))
    json_bytes = json_str.encode('utf-8')
    # Pad to 4-byte boundary
    while len(json_bytes) % 4 != 0:
        json_bytes += b' '

    # Pad buffer to 4-byte boundary
    while len(buffer_data) % 4 != 0:
        buffer_data.extend(b'\x00')

    # Write GLB
    with open(filepath, 'wb') as f:
        # Header
        f.write(struct.pack('<III', 0x46546C67, 2, 12 + 8 + len(json_bytes) + 8 + len(buffer_data)))
        # JSON chunk
        f.write(struct.pack('<II', len(json_bytes), 0x4E4F534A))
        f.write(json_bytes)
        # Binary chunk
        f.write(struct.pack('<II', len(buffer_data), 0x004E4942))
        f.write(buffer_data)


def generate_blendshape_targets(base_verts, faces):
    """Generate standard blendshape morph targets for lip sync.

    These are offsets from the base mesh for key expressions.
    Uses mediapipe landmark indices for mouth, eyes, brows.
    """
    # Key landmark indices (mediapipe 478-point face mesh)
    # Mouth
    UPPER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
    LOWER_LIP = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 61]
    MOUTH_CORNERS = [61, 291]

    # Eyes
    LEFT_EYE_UPPER = [159, 145, 133, 173, 157, 158]
    LEFT_EYE_LOWER = [145, 153, 154, 155, 133]
    RIGHT_EYE_UPPER = [386, 374, 362, 398, 384, 385]
    RIGHT_EYE_LOWER = [374, 380, 381, 382, 362]

    # Brows
    LEFT_BROW = [70, 63, 105, 66, 107]
    RIGHT_BROW = [300, 293, 334, 296, 336]

    # Jaw
    JAW = [152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234,
           377, 400, 378, 379, 365, 397, 288, 361, 323, 454]

    targets = {}

    # jawOpen - move lower lip and jaw down
    delta = np.zeros_like(base_verts)
    for i in LOWER_LIP + JAW:
        if i < len(delta):
            delta[i, 1] -= 3.0  # move down
    targets["jawOpen"] = delta.copy()

    # mouthSmileLeft
    delta = np.zeros_like(base_verts)
    if MOUTH_CORNERS[0] < len(delta):
        delta[MOUTH_CORNERS[0], 0] -= 1.5  # left corner moves left
        delta[MOUTH_CORNERS[0], 1] += 1.0  # and up
    targets["mouthSmileLeft"] = delta.copy()

    # mouthSmileRight
    delta = np.zeros_like(base_verts)
    if MOUTH_CORNERS[1] < len(delta):
        delta[MOUTH_CORNERS[1], 0] += 1.5
        delta[MOUTH_CORNERS[1], 1] += 1.0
    targets["mouthSmileRight"] = delta.copy()

    # mouthPucker
    delta = np.zeros_like(base_verts)
    for i in UPPER_LIP + LOWER_LIP:
        if i < len(delta):
            delta[i, 2] += 1.5  # push forward
    targets["mouthPucker"] = delta.copy()

    # eyeBlinkLeft
    delta = np.zeros_like(base_verts)
    for i in LEFT_EYE_UPPER:
        if i < len(delta):
            delta[i, 1] -= 1.0  # upper lid down
    for i in LEFT_EYE_LOWER:
        if i < len(delta):
            delta[i, 1] += 0.5  # lower lid up
    targets["eyeBlinkLeft"] = delta.copy()

    # eyeBlinkRight
    delta = np.zeros_like(base_verts)
    for i in RIGHT_EYE_UPPER:
        if i < len(delta):
            delta[i, 1] -= 1.0
    for i in RIGHT_EYE_LOWER:
        if i < len(delta):
            delta[i, 1] += 0.5
    targets["eyeBlinkRight"] = delta.copy()

    # browUpLeft
    delta = np.zeros_like(base_verts)
    for i in LEFT_BROW:
        if i < len(delta):
            delta[i, 1] += 1.5
    targets["browInnerUp"] = delta.copy()

    # browDown
    delta = np.zeros_like(base_verts)
    for i in LEFT_BROW + RIGHT_BROW:
        if i < len(delta):
            delta[i, 1] -= 1.0
    targets["browDownLeft"] = delta.copy()

    # Viseme targets for lip sync (ARKit-compatible names)
    # viseme_aa (open mouth, wide)
    delta = np.zeros_like(base_verts)
    for i in LOWER_LIP + JAW:
        if i < len(delta):
            delta[i, 1] -= 4.0
    for i in MOUTH_CORNERS:
        if i < len(delta):
            delta[i, 0] += (1.0 if i == 291 else -1.0) * 1.0
    targets["jawOpen_wide"] = delta.copy()  # viseme AA

    # viseme_oh (round mouth)
    delta = np.zeros_like(base_verts)
    for i in LOWER_LIP + JAW:
        if i < len(delta):
            delta[i, 1] -= 2.5
    for i in UPPER_LIP + LOWER_LIP:
        if i < len(delta):
            delta[i, 2] += 1.0
    targets["mouthFunnel"] = delta.copy()  # viseme O

    return targets


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== Unum Avatar Head Builder ===\n")

    # Step 1: Detect landmarks
    print("[1/5] Detecting face landmarks...")
    pts, blendshapes, transform, image = detect_landmarks(INPUT_IMAGE)
    print(f"  {len(pts)} landmarks detected")
    print(f"  {len(blendshapes)} blendshapes detected")

    # Step 2: Get face mesh triangulation
    print("[2/5] Building mesh topology...")
    faces = get_face_mesh_triangles()
    # Filter faces that reference iris landmarks (468-477) for cleaner mesh
    max_base = 468
    base_mask = np.all(faces < max_base, axis=1)
    faces_base = faces[base_mask]
    print(f"  {len(faces_base)} triangles (base face mesh)")

    # Step 3: Convert to 3D
    print("[3/5] Converting to 3D coordinates...")
    verts = landmarks_to_3d(pts[:max_base], image.width, image.height, scale=150.0)
    uvs = compute_uvs(pts[:max_base])
    normals = compute_normals(verts, faces_base)
    print(f"  {len(verts)} vertices")
    print(f"  X: [{verts[:,0].min():.1f}, {verts[:,0].max():.1f}]")
    print(f"  Y: [{verts[:,1].min():.1f}, {verts[:,1].max():.1f}]")
    print(f"  Z: [{verts[:,2].min():.1f}, {verts[:,2].max():.1f}]")

    # Step 4: Generate blendshape targets
    print("[4/5] Generating blendshape morph targets...")
    targets = generate_blendshape_targets(verts, faces_base)
    print(f"  {len(targets)} morph targets generated")
    for name in targets:
        print(f"    - {name}")

    # Step 5: Export
    print("[5/5] Exporting mesh files...")

    # OBJ (for viewing/debugging)
    obj_path = os.path.join(OUTPUT_DIR, "unum_head.obj")
    write_obj(obj_path, verts, faces_base, uvs, normals)
    print(f"  OBJ: {obj_path}")

    # GLB (for Three.js)
    glb_path = os.path.join(OUTPUT_DIR, "unum_head.glb")
    write_glb(glb_path, verts, faces_base, uvs, normals, INPUT_IMAGE)
    print(f"  GLB: {glb_path}")

    # Save blendshape targets as NPZ
    targets_path = os.path.join(OUTPUT_DIR, "blendshape_targets.npz")
    np.savez(targets_path, **{k: v.astype(np.float32) for k, v in targets.items()})
    print(f"  Blendshapes: {targets_path}")

    # Save landmark data
    data_path = os.path.join(OUTPUT_DIR, "face_data.json")
    with open(data_path, 'w') as f:
        json.dump({
            "landmark_count": len(pts),
            "vertex_count": len(verts),
            "face_count": len(faces_base),
            "blendshape_names": list(targets.keys()),
            "detected_blendshapes": blendshapes,
            "image_size": [image.width, image.height]
        }, f, indent=2)
    print(f"  Metadata: {data_path}")

    print(f"\n=== Done! Check {OUTPUT_DIR}/ ===")
    print(f"  Total: {len(verts)} verts, {len(faces_base)} tris, {len(targets)} blendshapes")


if __name__ == "__main__":
    main()
