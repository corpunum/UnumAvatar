"""
Build a 3D head mesh from mediapipe face landmarks — V2.
Fixed UV mapping: uses direct landmark x,y as texture coordinates
to project the face photo cleanly onto the mesh.
"""
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
import numpy as np
import json
import struct
import os
import cv2

INPUT_IMAGE = "input/unum-clean-front-neutral.png"
MODEL_PATH = "data/face_landmarker.task"
OUTPUT_DIR = "output"


def get_face_mesh_triangles():
    """Extract triangle indices from mediapipe's face mesh connections."""
    from mediapipe.tasks.python.vision import FaceLandmarksConnections
    edges = set()
    for conn in FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION:
        edges.add((conn.start, conn.end))

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

    return pts, blendshapes, image


def create_face_texture(image_path, pts, size=1024):
    """Create a clean face texture by cropping the face region
    and mapping UV coordinates to this crop."""
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    # The face landmarks are normalized [0,1]
    # The texture IS the full image — UVs map directly to landmark x,y positions
    # Just resize to power-of-2 for GPU efficiency
    texture = cv2.resize(img, (size, size))
    texture_path = os.path.join(OUTPUT_DIR, "unum_head_texture.png")
    cv2.imwrite(texture_path, texture)
    return texture_path


def landmarks_to_3d(pts, scale=150.0):
    """Convert normalized landmarks to 3D coordinates."""
    verts = np.zeros((len(pts), 3), dtype=np.float32)
    # Center on face
    cx = (pts[:, 0].min() + pts[:, 0].max()) / 2
    cy = (pts[:, 1].min() + pts[:, 1].max()) / 2

    verts[:, 0] = (pts[:, 0] - cx) * scale
    verts[:, 1] = -(pts[:, 1] - cy) * scale  # flip Y
    verts[:, 2] = -pts[:, 2] * scale * 0.6    # depth

    return verts


def compute_normals(verts, faces):
    """Compute per-vertex normals."""
    normals = np.zeros_like(verts)
    for face in faces:
        v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
        n = np.cross(v1 - v0, v2 - v0)
        norm = np.linalg.norm(n)
        if norm > 1e-8:
            n /= norm
        normals[face[0]] += n
        normals[face[1]] += n
        normals[face[2]] += n

    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1
    normals /= norms
    return normals.astype(np.float32)


def write_glb(filepath, verts, faces, uvs, normals, texture_path,
              morph_targets=None, morph_names=None):
    """Write GLB with optional morph targets."""
    verts_f32 = verts.astype(np.float32)
    normals_f32 = normals.astype(np.float32)
    uvs_f32 = uvs.astype(np.float32)
    indices_u16 = faces.flatten().astype(np.uint16)

    # Build binary buffer
    buffer_data = bytearray()

    # 0: indices
    idx_offset = len(buffer_data)
    buffer_data.extend(indices_u16.tobytes())
    while len(buffer_data) % 4:
        buffer_data.extend(b'\x00')

    # 1: positions
    pos_offset = len(buffer_data)
    buffer_data.extend(verts_f32.tobytes())

    # 2: normals
    norm_offset = len(buffer_data)
    buffer_data.extend(normals_f32.tobytes())

    # 3: texcoords
    uv_offset = len(buffer_data)
    buffer_data.extend(uvs_f32.tobytes())

    # Morph target accessors
    morph_accessor_start = 4
    morph_bv_start = 4
    morph_data = []
    if morph_targets:
        for name, delta in morph_targets.items():
            delta_f32 = delta.astype(np.float32)
            offset = len(buffer_data)
            buffer_data.extend(delta_f32.tobytes())
            morph_data.append({
                "offset": offset,
                "length": len(delta_f32.tobytes()),
                "max": delta_f32.max(axis=0).tolist(),
                "min": delta_f32.min(axis=0).tolist(),
                "count": len(delta_f32)
            })

    # Image
    img_offset = len(buffer_data)
    with open(texture_path, 'rb') as f:
        img_bytes = f.read()
    buffer_data.extend(img_bytes)

    total_len = len(buffer_data)

    # Build glTF structure
    buffer_views = [
        {"buffer": 0, "byteOffset": idx_offset, "byteLength": len(indices_u16) * 2, "target": 34963},
        {"buffer": 0, "byteOffset": pos_offset, "byteLength": len(verts_f32) * 4 * 3, "target": 34962, "byteStride": 12},
        {"buffer": 0, "byteOffset": norm_offset, "byteLength": len(normals_f32) * 4 * 3, "target": 34962, "byteStride": 12},
        {"buffer": 0, "byteOffset": uv_offset, "byteLength": len(uvs_f32) * 4 * 2, "target": 34962, "byteStride": 8},
    ]

    accessors = [
        {"bufferView": 0, "componentType": 5123, "count": len(indices_u16), "type": "SCALAR",
         "max": [int(indices_u16.max())], "min": [int(indices_u16.min())]},
        {"bufferView": 1, "componentType": 5126, "count": len(verts_f32), "type": "VEC3",
         "max": verts_f32.max(axis=0).tolist(), "min": verts_f32.min(axis=0).tolist()},
        {"bufferView": 2, "componentType": 5126, "count": len(normals_f32), "type": "VEC3"},
        {"bufferView": 3, "componentType": 5126, "count": len(uvs_f32), "type": "VEC2"},
    ]

    # Add morph target buffer views and accessors
    morph_targets_gltf = []
    for i, md in enumerate(morph_data):
        bv_idx = len(buffer_views)
        buffer_views.append({
            "buffer": 0, "byteOffset": md["offset"],
            "byteLength": md["length"], "byteStride": 12
        })
        acc_idx = len(accessors)
        accessors.append({
            "bufferView": bv_idx, "componentType": 5126,
            "count": md["count"], "type": "VEC3",
            "max": md["max"], "min": md["min"]
        })
        morph_targets_gltf.append({"POSITION": acc_idx})

    # Image buffer view
    img_bv_idx = len(buffer_views)
    buffer_views.append({"buffer": 0, "byteOffset": img_offset, "byteLength": len(img_bytes)})

    primitive = {
        "attributes": {"POSITION": 1, "NORMAL": 2, "TEXCOORD_0": 3},
        "indices": 0,
        "material": 0
    }
    if morph_targets_gltf:
        primitive["targets"] = morph_targets_gltf

    mesh_def = {"primitives": [primitive]}
    if morph_names:
        mesh_def["extras"] = {"targetNames": morph_names}
        mesh_def["weights"] = [0.0] * len(morph_names)

    gltf = {
        "asset": {"version": "2.0", "generator": "unum-avatar-builder-v2"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "UNumHead"}],
        "meshes": [mesh_def],
        "accessors": accessors,
        "bufferViews": buffer_views,
        "materials": [{
            "pbrMetallicRoughness": {
                "metallicFactor": 0.0,
                "roughnessFactor": 0.6,
                "baseColorTexture": {"index": 0}
            },
            "name": "skin",
            "doubleSided": True
        }],
        "textures": [{"source": 0, "sampler": 0}],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],
        "images": [{"bufferView": img_bv_idx, "mimeType": "image/png"}],
        "buffers": [{"byteLength": total_len}]
    }

    json_str = json.dumps(gltf, separators=(',', ':'))
    json_bytes = json_str.encode('utf-8')
    while len(json_bytes) % 4:
        json_bytes += b' '
    while len(buffer_data) % 4:
        buffer_data.extend(b'\x00')

    with open(filepath, 'wb') as f:
        f.write(struct.pack('<III', 0x46546C67, 2, 12 + 8 + len(json_bytes) + 8 + len(buffer_data)))
        f.write(struct.pack('<II', len(json_bytes), 0x4E4F534A))
        f.write(json_bytes)
        f.write(struct.pack('<II', len(buffer_data), 0x004E4942))
        f.write(buffer_data)


def generate_blendshapes(base_verts):
    """Generate morph target deltas for lip sync and expressions."""
    # MediaPipe face mesh landmark regions
    UPPER_LIP_TOP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
    LOWER_LIP_BOT = [146, 91, 181, 84, 17, 314, 405, 321, 375]
    UPPER_LIP_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
    LOWER_LIP_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
    MOUTH_LEFT = [61, 78]
    MOUTH_RIGHT = [291, 308]
    JAW_OUTLINE = [152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
                   377, 400, 378, 379, 365, 397, 288, 361, 323]
    LEFT_EYE_TOP = [159, 158, 157, 173, 133, 246, 161, 160]
    LEFT_EYE_BOT = [145, 153, 154, 155, 133, 144, 163]
    RIGHT_EYE_TOP = [386, 385, 384, 398, 362, 466, 388, 387]
    RIGHT_EYE_BOT = [374, 380, 381, 382, 362, 373, 390]
    LEFT_BROW = [70, 63, 105, 66, 107, 55, 65]
    RIGHT_BROW = [300, 293, 334, 296, 336, 285, 295]

    n = len(base_verts)
    targets = {}

    def delta_for(indices, dx=0, dy=0, dz=0, smooth_radius=0):
        d = np.zeros((n, 3), dtype=np.float32)
        for i in indices:
            if i < n:
                d[i] = [dx, dy, dz]
        return d

    # jawOpen
    targets["jawOpen"] = (
        delta_for(LOWER_LIP_BOT + LOWER_LIP_INNER, dy=-4.0) +
        delta_for(JAW_OUTLINE, dy=-2.5)
    )

    # mouthClose (lips together, jaw stays)
    targets["mouthClose"] = (
        delta_for(LOWER_LIP_BOT, dy=1.5) +
        delta_for(UPPER_LIP_TOP, dy=-0.5)
    )

    # mouthFunnel (O shape)
    targets["mouthFunnel"] = (
        delta_for(LOWER_LIP_BOT + LOWER_LIP_INNER, dy=-2.0, dz=2.0) +
        delta_for(UPPER_LIP_TOP + UPPER_LIP_INNER, dy=1.0, dz=2.0) +
        delta_for(MOUTH_LEFT, dx=1.0) +
        delta_for(MOUTH_RIGHT, dx=-1.0)
    )

    # mouthPucker
    targets["mouthPucker"] = (
        delta_for(UPPER_LIP_TOP + UPPER_LIP_INNER + LOWER_LIP_BOT + LOWER_LIP_INNER, dz=2.5) +
        delta_for(MOUTH_LEFT, dx=2.0) +
        delta_for(MOUTH_RIGHT, dx=-2.0)
    )

    # mouthSmileLeft / Right
    targets["mouthSmileLeft"] = delta_for(MOUTH_LEFT, dx=-2.0, dy=1.5)
    targets["mouthSmileRight"] = delta_for(MOUTH_RIGHT, dx=2.0, dy=1.5)

    # mouthOpen (wide AA)
    targets["mouthOpen"] = (
        delta_for(LOWER_LIP_BOT + LOWER_LIP_INNER + JAW_OUTLINE, dy=-5.0) +
        delta_for(MOUTH_LEFT, dx=-1.5) +
        delta_for(MOUTH_RIGHT, dx=1.5)
    )

    # eyeBlinkLeft / Right
    targets["eyeBlinkLeft"] = (
        delta_for(LEFT_EYE_TOP, dy=-1.5) +
        delta_for(LEFT_EYE_BOT, dy=0.8)
    )
    targets["eyeBlinkRight"] = (
        delta_for(RIGHT_EYE_TOP, dy=-1.5) +
        delta_for(RIGHT_EYE_BOT, dy=0.8)
    )

    # browInnerUp
    targets["browInnerUp"] = delta_for(LEFT_BROW + RIGHT_BROW, dy=2.0)

    # browDownLeft / Right
    targets["browDownLeft"] = delta_for(LEFT_BROW, dy=-1.5)
    targets["browDownRight"] = delta_for(RIGHT_BROW, dy=-1.5)

    return targets


def write_obj(filepath, verts, faces, uvs, normals):
    with open(filepath, 'w') as f:
        f.write("# Unum Avatar Head v2\n")
        f.write(f"mtllib unum_head.mtl\n")
        f.write(f"usemtl skin\n\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for uv in uvs:
            f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
        for n in normals:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        for face in faces:
            i0, i1, i2 = face[0]+1, face[1]+1, face[2]+1
            f.write(f"f {i0}/{i0}/{i0} {i1}/{i1}/{i1} {i2}/{i2}/{i2}\n")

    # Write MTL
    mtl_path = filepath.replace('.obj', '.mtl')
    with open(mtl_path, 'w') as f:
        f.write("newmtl skin\n")
        f.write("Ka 0.2 0.2 0.2\n")
        f.write("Kd 0.8 0.8 0.8\n")
        f.write("Ks 0.1 0.1 0.1\n")
        f.write("map_Kd unum_head_texture.png\n")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=== Unum Avatar Head Builder v2 ===\n")

    # Detect
    print("[1/6] Detecting face landmarks...")
    pts, blendshapes, image = detect_landmarks(INPUT_IMAGE)
    print(f"  {len(pts)} landmarks, {len(blendshapes)} blendshapes")

    # Topology
    print("[2/6] Building mesh topology...")
    faces = get_face_mesh_triangles()
    max_v = 468
    faces = faces[np.all(faces < max_v, axis=1)]
    print(f"  {len(faces)} triangles")

    # 3D vertices
    print("[3/6] Converting to 3D...")
    verts = landmarks_to_3d(pts[:max_v])
    normals = compute_normals(verts, faces)
    print(f"  {len(verts)} vertices")

    # UV mapping — direct projection from landmark positions
    print("[4/6] Computing UV mapping...")
    uvs = np.zeros((max_v, 2), dtype=np.float32)
    uvs[:, 0] = pts[:max_v, 0]         # U = normalized X
    uvs[:, 1] = 1.0 - pts[:max_v, 1]   # V = 1 - normalized Y
    print(f"  UV range: [{uvs.min():.3f}, {uvs.max():.3f}]")

    # Create texture
    print("[5/6] Creating face texture...")
    texture_path = create_face_texture(INPUT_IMAGE, pts, size=1024)
    print(f"  Texture: {texture_path}")

    # Blendshapes
    print("[6/6] Generating blendshapes + exporting...")
    morph_targets = generate_blendshapes(verts)
    morph_names = list(morph_targets.keys())
    print(f"  {len(morph_targets)} morph targets: {', '.join(morph_names)}")

    # Export OBJ
    obj_path = os.path.join(OUTPUT_DIR, "unum_head_v2.obj")
    write_obj(obj_path, verts, faces, uvs, normals)
    print(f"  OBJ: {obj_path}")

    # Export GLB with morph targets
    glb_path = os.path.join(OUTPUT_DIR, "unum_head_v2.glb")
    write_glb(glb_path, verts, faces, uvs, normals, texture_path,
              morph_targets, morph_names)
    print(f"  GLB: {glb_path}")

    # Save blendshapes
    np.savez(os.path.join(OUTPUT_DIR, "blendshape_targets_v2.npz"),
             **{k: v.astype(np.float32) for k, v in morph_targets.items()})

    # Metadata
    with open(os.path.join(OUTPUT_DIR, "face_data_v2.json"), 'w') as f:
        json.dump({
            "version": 2,
            "vertex_count": len(verts),
            "face_count": len(faces),
            "blendshape_names": morph_names,
            "detected_blendshapes": blendshapes,
            "texture_size": 1024,
            "image_size": [image.width, image.height]
        }, f, indent=2)

    print(f"\n=== Done! {len(verts)} verts, {len(faces)} tris, {len(morph_targets)} blendshapes ===")


if __name__ == "__main__":
    main()
