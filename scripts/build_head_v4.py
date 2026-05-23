"""
Unum Avatar Head Builder v4.
Combines MediaPipe (468 dense landmarks + topology) with InsightFace (68 3D landmarks
with real depth). The InsightFace Z-depth is interpolated onto MediaPipe points
for a properly shaped 3D face.
"""
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
import numpy as np
import json
import struct
import os
import cv2
from scipy.interpolate import RBFInterpolator


INPUT_IMAGE = "input/unum-clean-front-neutral.png"
MODEL_PATH = "data/face_landmarker.task"
OUTPUT_DIR = "output"
TEX_SIZE = 2048


# 68-point landmark mapping to MediaPipe indices (approximate correspondence)
# Based on dlib 68-point face landmark ordering
MP_TO_68 = {
    # Jaw contour (0-16)
    0: 10, 1: 338, 2: 297, 3: 332, 4: 284, 5: 251, 6: 389, 7: 356, 8: 454,
    9: 323, 10: 361, 11: 288, 12: 397, 13: 365, 14: 379, 15: 378, 16: 400,
    # Right eyebrow (17-21)
    17: 70, 18: 63, 19: 105, 20: 66, 21: 107,
    # Left eyebrow (22-26)
    22: 336, 23: 296, 24: 334, 25: 293, 26: 300,
    # Nose bridge (27-30)
    27: 168, 28: 6, 29: 197, 30: 195,
    # Nose bottom (31-35)
    31: 5, 32: 4, 33: 1, 34: 274, 35: 275,
    # Right eye (36-41)
    36: 33, 37: 160, 38: 158, 39: 133, 40: 153, 41: 144,
    # Left eye (42-47)
    42: 362, 43: 385, 44: 387, 45: 263, 46: 373, 47: 380,
    # Outer lip (48-59)
    48: 61, 49: 40, 50: 37, 51: 0, 52: 267, 53: 270, 54: 291,
    55: 321, 56: 314, 57: 17, 58: 84, 59: 181,
    # Inner lip (60-67)
    60: 78, 61: 81, 62: 13, 63: 311, 64: 308, 65: 402, 66: 14, 67: 178,
}


def get_face_mesh_triangles():
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
            triangles.add(tuple(sorted([a, b, c])))
    return np.array(list(triangles), dtype=np.int32)


def detect_mediapipe(image_path):
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        output_face_blendshapes=True, num_faces=1
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)
    image = mp.Image.create_from_file(image_path)
    result = landmarker.detect(image)
    landmarker.close()
    if not result.face_landmarks:
        raise RuntimeError("No face detected by MediaPipe")
    lm = result.face_landmarks[0]
    pts = np.array([[l.x, l.y, l.z] for l in lm])
    return pts


def detect_insightface(image_path):
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    img = cv2.imread(image_path)
    faces = app.get(img)
    if not faces:
        raise RuntimeError("No face detected by InsightFace")
    return faces[0].landmark_3d_68, img.shape[:2]


def interpolate_depth(mp_pts, if_pts_3d, img_shape):
    """Use InsightFace 3D landmarks to give real depth to MediaPipe points."""
    h, w = img_shape

    # Get InsightFace 2D positions (pixel coords) and Z values
    if_xy = if_pts_3d[:, :2]
    if_z = if_pts_3d[:, 2]

    # Normalize InsightFace Z to a reasonable range
    z_min, z_max = if_z.min(), if_z.max()
    if_z_norm = (if_z - z_min) / (z_max - z_min + 1e-8)

    # Get MediaPipe 2D positions in pixel coords
    mp_xy_px = mp_pts[:, :2] * np.array([w, h])

    # RBF interpolation: learn Z-depth from InsightFace landmarks, predict for all MediaPipe points
    rbf = RBFInterpolator(if_xy, if_z_norm, kernel='thin_plate_spline', smoothing=1.0)
    mp_z_interp = rbf(mp_xy_px)

    return np.clip(mp_z_interp, 0, 1)


def compute_uvs(pts, max_v):
    face_pts = pts[:max_v, :2]
    x_min, y_min = face_pts.min(axis=0)
    x_max, y_max = face_pts.max(axis=0)
    margin = 0.02
    x_range = x_max - x_min
    y_range = y_max - y_min
    uvs = np.zeros((max_v, 2), dtype=np.float32)
    uvs[:, 0] = margin + (1.0 - 2*margin) * (face_pts[:, 0] - x_min) / x_range
    # glTF UV: V=0 is top of image, V=1 is bottom — same direction as image Y
    uvs[:, 1] = margin + (1.0 - 2*margin) * (face_pts[:, 1] - y_min) / y_range
    return uvs


def bake_texture(image_path, pts, uvs, faces, max_v, tex_size=2048):
    src = cv2.imread(image_path)
    src_h, src_w = src.shape[:2]

    face_region = src[int(src_h*0.3):int(src_h*0.7), int(src_w*0.3):int(src_w*0.7)]
    avg_color = face_region.mean(axis=(0,1)).astype(np.uint8)
    texture = np.full((tex_size, tex_size, 3), avg_color, dtype=np.uint8)

    face_pts_2d = pts[:max_v, :2]
    x_min, y_min = face_pts_2d.min(axis=0)
    x_max, y_max = face_pts_2d.max(axis=0)
    margin = 0.02
    x_range = x_max - x_min
    y_range = y_max - y_min

    # Invert UV back to normalized landmark coords to get source pixel positions
    src_px = (((uvs[:, 0] - margin) / (1.0 - 2*margin)) * x_range + x_min) * src_w
    src_py = (((uvs[:, 1] - margin) / (1.0 - 2*margin)) * y_range + y_min) * src_h

    # UV to texture pixel coords (V=0 is top row, same as image convention)
    uv_px = uvs[:, 0] * tex_size
    uv_py = uvs[:, 1] * tex_size

    for face in faces:
        i0, i1, i2 = face
        uv_tri = np.array([[uv_px[i0], uv_py[i0]], [uv_px[i1], uv_py[i1]],
                           [uv_px[i2], uv_py[i2]]], dtype=np.float32)
        src_tri = np.array([[src_px[i0], src_py[i0]], [src_px[i1], src_py[i1]],
                            [src_px[i2], src_py[i2]]], dtype=np.float32)

        x_min_t = max(0, int(np.floor(uv_tri[:, 0].min())) - 1)
        y_min_t = max(0, int(np.floor(uv_tri[:, 1].min())) - 1)
        x_max_t = min(tex_size, int(np.ceil(uv_tri[:, 0].max())) + 1)
        y_max_t = min(tex_size, int(np.ceil(uv_tri[:, 1].max())) + 1)
        w, h = x_max_t - x_min_t, y_max_t - y_min_t
        if w < 1 or h < 1:
            continue

        uv_local = uv_tri - np.array([x_min_t, y_min_t], dtype=np.float32)
        M = cv2.getAffineTransform(src_tri, uv_local)
        patch = cv2.warpAffine(src, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, uv_local.astype(np.int32), 255, lineType=cv2.LINE_8)
        roi = texture[y_min_t:y_max_t, x_min_t:x_max_t]
        np.copyto(roi, patch, where=(mask[:, :, None] > 0))

    out_path = os.path.join(OUTPUT_DIR, "unum_head_texture_v4.png")
    cv2.imwrite(out_path, texture)
    return out_path


def landmarks_to_3d(pts_2d, z_depth, scale=150.0, depth_scale=1.0):
    """Build 3D mesh with real depth from InsightFace interpolation."""
    n = len(pts_2d)
    verts = np.zeros((n, 3), dtype=np.float32)
    cx = (pts_2d[:, 0].min() + pts_2d[:, 0].max()) / 2
    cy = (pts_2d[:, 1].min() + pts_2d[:, 1].max()) / 2
    verts[:, 0] = (pts_2d[:, 0] - cx) * scale
    verts[:, 1] = -(pts_2d[:, 1] - cy) * scale
    # Use interpolated InsightFace depth instead of MediaPipe's flat Z
    verts[:, 2] = z_depth * scale * depth_scale
    return verts


def subdivide_mesh(verts, faces, uvs, n_iters=2):
    for _ in range(n_iters):
        edge_mid = {}
        new_verts = list(verts)
        new_uvs = list(uvs)
        new_faces = []

        def get_mid(a, b):
            key = (min(a, b), max(a, b))
            if key in edge_mid:
                return edge_mid[key]
            idx = len(new_verts)
            new_verts.append((verts[a] + verts[b]) / 2)
            new_uvs.append((uvs[a] + uvs[b]) / 2)
            edge_mid[key] = idx
            return idx

        for f in faces:
            i0, i1, i2 = f
            m01 = get_mid(i0, i1)
            m12 = get_mid(i1, i2)
            m02 = get_mid(i0, i2)
            new_faces.append([i0, m01, m02])
            new_faces.append([m01, i1, m12])
            new_faces.append([m02, m12, i2])
            new_faces.append([m01, m12, m02])

        verts = np.array(new_verts, dtype=np.float32)
        uvs = np.array(new_uvs, dtype=np.float32)
        faces = np.array(new_faces, dtype=np.int32)

    return verts, faces, uvs


def compute_normals(verts, faces):
    normals = np.zeros_like(verts)
    for face in faces:
        v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
        n = np.cross(v1 - v0, v2 - v0)
        norm = np.linalg.norm(n)
        if norm > 1e-8:
            n /= norm
        for i in face:
            normals[i] += n
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1
    return (normals / norms).astype(np.float32)


def generate_blendshapes(base_verts):
    n = len(base_verts)
    UPPER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
    LOWER_LIP = [146, 91, 181, 84, 17, 314, 405, 321, 375]
    INNER_UP = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
    INNER_LO = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
    MOUTH_L = [61, 78]
    MOUTH_R = [291, 308]
    JAW = [152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
           377, 400, 378, 379, 365, 397, 288, 361, 323]
    L_EYE_T = [159, 158, 157, 173, 133, 246, 161, 160]
    L_EYE_B = [145, 153, 154, 155, 133, 144, 163]
    R_EYE_T = [386, 385, 384, 398, 362, 466, 388, 387]
    R_EYE_B = [374, 380, 381, 382, 362, 373, 390]
    L_BROW = [70, 63, 105, 66, 107, 55, 65]
    R_BROW = [300, 293, 334, 296, 336, 285, 295]

    def d(indices, dx=0, dy=0, dz=0):
        delta = np.zeros((n, 3), dtype=np.float32)
        for i in indices:
            if i < n: delta[i] = [dx, dy, dz]
        return delta

    return {
        "jawOpen": d(LOWER_LIP + INNER_LO, dy=-4) + d(JAW, dy=-2.5),
        "mouthClose": d(LOWER_LIP, dy=1.5) + d(UPPER_LIP, dy=-0.5),
        "mouthFunnel": d(LOWER_LIP + INNER_LO, dy=-2, dz=2) + d(UPPER_LIP + INNER_UP, dy=1, dz=2) + d(MOUTH_L, dx=1) + d(MOUTH_R, dx=-1),
        "mouthPucker": d(UPPER_LIP + INNER_UP + LOWER_LIP + INNER_LO, dz=2.5) + d(MOUTH_L, dx=2) + d(MOUTH_R, dx=-2),
        "mouthSmileLeft": d(MOUTH_L, dx=-2, dy=1.5),
        "mouthSmileRight": d(MOUTH_R, dx=2, dy=1.5),
        "mouthOpen": d(LOWER_LIP + INNER_LO + JAW, dy=-5) + d(MOUTH_L, dx=-1.5) + d(MOUTH_R, dx=1.5),
        "eyeBlinkLeft": d(L_EYE_T, dy=-1.5) + d(L_EYE_B, dy=0.8),
        "eyeBlinkRight": d(R_EYE_T, dy=-1.5) + d(R_EYE_B, dy=0.8),
        "browInnerUp": d(L_BROW + R_BROW, dy=2),
        "browDownLeft": d(L_BROW, dy=-1.5),
        "browDownRight": d(R_BROW, dy=-1.5),
    }


def pad_morph(delta, total_verts, original_verts):
    padded = np.zeros((total_verts, 3), dtype=np.float32)
    padded[:original_verts] = delta[:original_verts]
    return padded


def write_glb(filepath, verts, faces, uvs, normals, texture_path,
              morph_targets=None, morph_names=None):
    verts_f32 = verts.astype(np.float32)
    normals_f32 = normals.astype(np.float32)
    uvs_f32 = uvs.astype(np.float32)
    indices_u16 = faces.flatten().astype(np.uint16)

    buf = bytearray()

    idx_off = len(buf)
    buf.extend(indices_u16.tobytes())
    while len(buf) % 4: buf.extend(b'\x00')

    pos_off = len(buf)
    buf.extend(verts_f32.tobytes())
    norm_off = len(buf)
    buf.extend(normals_f32.tobytes())
    uv_off = len(buf)
    buf.extend(uvs_f32.tobytes())

    morph_info = []
    if morph_targets:
        for name in morph_names:
            delta = morph_targets[name].astype(np.float32)
            off = len(buf)
            buf.extend(delta.tobytes())
            morph_info.append({"off": off, "len": len(delta) * 12,
                               "max": delta.max(axis=0).tolist(),
                               "min": delta.min(axis=0).tolist(),
                               "count": len(delta)})

    img_off = len(buf)
    with open(texture_path, 'rb') as f:
        img_bytes = f.read()
    buf.extend(img_bytes)

    bvs = [
        {"buffer": 0, "byteOffset": idx_off, "byteLength": len(indices_u16)*2, "target": 34963},
        {"buffer": 0, "byteOffset": pos_off, "byteLength": len(verts_f32)*12, "target": 34962, "byteStride": 12},
        {"buffer": 0, "byteOffset": norm_off, "byteLength": len(normals_f32)*12, "target": 34962, "byteStride": 12},
        {"buffer": 0, "byteOffset": uv_off, "byteLength": len(uvs_f32)*8, "target": 34962, "byteStride": 8},
    ]
    accs = [
        {"bufferView": 0, "componentType": 5123, "count": len(indices_u16), "type": "SCALAR",
         "max": [int(indices_u16.max())], "min": [int(indices_u16.min())]},
        {"bufferView": 1, "componentType": 5126, "count": len(verts_f32), "type": "VEC3",
         "max": verts_f32.max(axis=0).tolist(), "min": verts_f32.min(axis=0).tolist()},
        {"bufferView": 2, "componentType": 5126, "count": len(normals_f32), "type": "VEC3"},
        {"bufferView": 3, "componentType": 5126, "count": len(uvs_f32), "type": "VEC2"},
    ]

    morph_tgts = []
    for mi in morph_info:
        bv_i = len(bvs)
        bvs.append({"buffer": 0, "byteOffset": mi["off"], "byteLength": mi["len"], "byteStride": 12})
        ac_i = len(accs)
        accs.append({"bufferView": bv_i, "componentType": 5126, "count": mi["count"],
                      "type": "VEC3", "max": mi["max"], "min": mi["min"]})
        morph_tgts.append({"POSITION": ac_i})

    img_bv = len(bvs)
    bvs.append({"buffer": 0, "byteOffset": img_off, "byteLength": len(img_bytes)})

    prim = {"attributes": {"POSITION": 1, "NORMAL": 2, "TEXCOORD_0": 3}, "indices": 0, "material": 0}
    if morph_tgts:
        prim["targets"] = morph_tgts

    mesh_def = {"primitives": [prim]}
    if morph_names:
        mesh_def["extras"] = {"targetNames": morph_names}
        mesh_def["weights"] = [0.0] * len(morph_names)

    gltf = {
        "asset": {"version": "2.0", "generator": "unum-avatar-v4"},
        "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "UNumHead"}],
        "meshes": [mesh_def], "accessors": accs, "bufferViews": bvs,
        "materials": [{"pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 0.5,
                       "baseColorTexture": {"index": 0}}, "doubleSided": True}],
        "textures": [{"source": 0, "sampler": 0}],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],
        "images": [{"bufferView": img_bv, "mimeType": "image/png"}],
        "buffers": [{"byteLength": len(buf)}]
    }

    jb = json.dumps(gltf, separators=(',', ':')).encode('utf-8')
    while len(jb) % 4: jb += b' '
    while len(buf) % 4: buf.extend(b'\x00')

    with open(filepath, 'wb') as f:
        f.write(struct.pack('<III', 0x46546C67, 2, 12 + 8 + len(jb) + 8 + len(buf)))
        f.write(struct.pack('<II', len(jb), 0x4E4F534A))
        f.write(jb)
        f.write(struct.pack('<II', len(buf), 0x004E4942))
        f.write(buf)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=== Unum Avatar Head v4 (InsightFace depth) ===\n")

    print("[1/8] Detecting MediaPipe landmarks...")
    mp_pts = detect_mediapipe(INPUT_IMAGE)
    max_v = 468
    print(f"  {len(mp_pts)} landmarks")

    print("[2/8] Detecting InsightFace 3D landmarks...")
    if_pts_3d, img_shape = detect_insightface(INPUT_IMAGE)
    print(f"  68 landmarks, Z range: {if_pts_3d[:,2].min():.1f} to {if_pts_3d[:,2].max():.1f}")

    print("[3/8] Interpolating depth...")
    z_depth = interpolate_depth(mp_pts[:max_v], if_pts_3d, img_shape)
    print(f"  Depth range: {z_depth.min():.3f} to {z_depth.max():.3f}")

    print("[4/8] Building topology...")
    faces = get_face_mesh_triangles()
    faces = faces[np.all(faces < max_v, axis=1)]
    print(f"  {len(faces)} triangles")

    print("[5/8] Building 3D mesh with real depth...")
    verts = landmarks_to_3d(mp_pts[:max_v], z_depth, scale=150.0, depth_scale=0.35)

    # Fix winding order
    fixed_faces = []
    for tri in faces:
        i0, i1, i2 = tri
        v0, v1, v2 = verts[i0], verts[i1], verts[i2]
        n = np.cross(v1 - v0, v2 - v0)
        if n[2] < 0:
            fixed_faces.append([i0, i2, i1])
        else:
            fixed_faces.append([i0, i1, i2])
    faces = np.array(fixed_faces, dtype=np.int32)

    print("[6/8] Computing UVs + subdividing...")
    uvs = compute_uvs(mp_pts, max_v)
    verts, faces, uvs = subdivide_mesh(verts, faces, uvs, n_iters=2)
    normals = compute_normals(verts, faces)
    print(f"  {len(verts)} verts, {len(faces)} tris after subdivision")

    print("[7/8] Baking face texture...")
    tex_path = bake_texture(INPUT_IMAGE, mp_pts, uvs, faces, max_v, TEX_SIZE)
    print(f"  Texture: {tex_path}")

    print("[8/8] Generating blendshapes + export...")
    morphs = generate_blendshapes(verts)
    morphs = {name: pad_morph(delta, len(verts), max_v) for name, delta in morphs.items()}
    names = list(morphs.keys())

    glb_path = os.path.join(OUTPUT_DIR, "unum_head_v4.glb")
    write_glb(glb_path, verts, faces, uvs, normals, tex_path, morphs, names)
    print(f"  GLB: {glb_path} ({os.path.getsize(glb_path)/1024/1024:.1f} MB)")
    print(f"  {len(verts)} verts, {len(faces)} tris, {len(morphs)} blendshapes")
    print(f"\n=== Done! Update viewer.html to load unum_head_v4.glb ===")


if __name__ == "__main__":
    main()
