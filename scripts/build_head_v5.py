"""
Unum Avatar Head Builder v5.
Priorities: eye socket fix, mouth cavity, back-of-head, proper multi-mesh GLB.
Uses MediaPipe (468 dense landmarks) + InsightFace (68 3D depth).
Outputs a single GLB with: face mesh (morphed), eyeballs, mouth cavity, cranium.
"""
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
import numpy as np
import json
import struct
import os
import cv2
from scipy.interpolate import RBFInterpolator

# Use best available front photo
import os as _os
_FRONT_OPTIONS = [
    "input/unum-clean-front-v2.png",
    "input/unum-clean-front-v3.png",
    "input/unum-clean-front-neutral.png",
]
INPUT_IMAGE = next((p for p in _FRONT_OPTIONS if _os.path.exists(p)), "input/unum-clean-front-neutral.png")
MODEL_PATH = "data/face_landmarker.task"
OUTPUT_DIR = "output"
TEX_SIZE = 2048


# ── Detection ──────────────────────────────────────────────────────────────

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
    return np.array([[l.x, l.y, l.z] for l in lm])


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
    h, w = img_shape
    if_xy = if_pts_3d[:, :2]
    if_z = if_pts_3d[:, 2]
    z_min, z_max = if_z.min(), if_z.max()
    if_z_norm = (if_z - z_min) / (z_max - z_min + 1e-8)
    mp_xy_px = mp_pts[:, :2] * np.array([w, h])
    rbf = RBFInterpolator(if_xy, if_z_norm, kernel='thin_plate_spline', smoothing=1.0)
    mp_z_interp = rbf(mp_xy_px)
    return np.clip(mp_z_interp, 0, 1)


# ── Mesh building ─────────────────────────────────────────────────────────

def landmarks_to_3d(pts_2d, z_depth, scale=150.0, depth_scale=0.35):
    n = len(pts_2d)
    verts = np.zeros((n, 3), dtype=np.float32)
    cx = (pts_2d[:, 0].min() + pts_2d[:, 0].max()) / 2
    cy = (pts_2d[:, 1].min() + pts_2d[:, 1].max()) / 2
    verts[:, 0] = (pts_2d[:, 0] - cx) * scale
    verts[:, 1] = -(pts_2d[:, 1] - cy) * scale
    verts[:, 2] = z_depth * scale * depth_scale
    return verts


def fix_winding(verts, faces):
    fixed = []
    for tri in faces:
        i0, i1, i2 = tri
        v0, v1, v2 = verts[i0], verts[i1], verts[i2]
        n = np.cross(v1 - v0, v2 - v0)
        if n[2] < 0:
            fixed.append([i0, i2, i1])
        else:
            fixed.append([i0, i1, i2])
    return np.array(fixed, dtype=np.int32)


def subdivide_mesh(verts, faces, uvs, n_iters=2):
    edge_parents = []
    for _ in range(n_iters):
        edge_mid = {}
        new_verts = list(verts)
        new_uvs = list(uvs)
        new_faces = []
        iter_parents = {}

        def get_mid(a, b):
            key = (min(a, b), max(a, b))
            if key in edge_mid:
                return edge_mid[key]
            idx = len(new_verts)
            new_verts.append((verts[a] + verts[b]) / 2)
            new_uvs.append((uvs[a] + uvs[b]) / 2)
            edge_mid[key] = idx
            iter_parents[idx] = (a, b)
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
        edge_parents.append(iter_parents)

    return verts, faces, uvs, edge_parents


def subdivide_deltas(delta, edge_parents):
    """Interpolate morph deltas for subdivided midpoint vertices."""
    for iter_parents in edge_parents:
        new_delta = np.zeros((max(iter_parents.keys()) + 1, 3), dtype=np.float32) if iter_parents else delta
        if len(new_delta) < len(delta):
            new_delta = np.zeros((max(max(iter_parents.keys()) + 1, len(delta)), 3), dtype=np.float32)
        new_delta[:len(delta)] = delta
        for mid_idx, (a, b) in iter_parents.items():
            if a < len(delta) and b < len(delta):
                new_delta[mid_idx] = (delta[a] + delta[b]) / 2
        delta = new_delta
    return delta


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


def compute_uvs(pts, max_v):
    face_pts = pts[:max_v, :2]
    x_min, y_min = face_pts.min(axis=0)
    x_max, y_max = face_pts.max(axis=0)
    margin = 0.02
    x_range = x_max - x_min
    y_range = y_max - y_min
    uvs = np.zeros((max_v, 2), dtype=np.float32)
    uvs[:, 0] = margin + (1.0 - 2*margin) * (face_pts[:, 0] - x_min) / x_range
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

    src_px = (((uvs[:, 0] - margin) / (1.0 - 2*margin)) * x_range + x_min) * src_w
    src_py = (((uvs[:, 1] - margin) / (1.0 - 2*margin)) * y_range + y_min) * src_h

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

    # Inpaint any remaining black gaps
    gray = cv2.cvtColor(texture, cv2.COLOR_BGR2GRAY)
    mask_inpaint = (gray < 5).astype(np.uint8) * 255
    if mask_inpaint.sum() > 0:
        texture = cv2.inpaint(texture, mask_inpaint, 16, cv2.INPAINT_TELEA)

    # Unsharp mask — subtle frequency enhancement for skin pore detail
    blur = cv2.GaussianBlur(texture, (0, 0), 2.0)
    texture = cv2.addWeighted(texture, 1.35, blur, -0.35, 0)
    texture = np.clip(texture, 0, 255).astype(np.uint8)

    out_path = os.path.join(OUTPUT_DIR, "unum_head_texture_v5.png")
    cv2.imwrite(out_path, texture)
    return out_path


# ── Extra geometry: eyeballs, mouth cavity, cranium ───────────────────────

def make_eyeball(center, radius=2.2, segments=24):
    """UV sphere for eyeball. Returns verts, faces, uvs."""
    verts = []
    uvs = []
    for j in range(segments + 1):
        phi = np.pi * j / segments
        for i in range(segments + 1):
            theta = 2 * np.pi * i / segments
            x = radius * np.sin(phi) * np.cos(theta)
            y = radius * np.cos(phi)
            z = radius * np.sin(phi) * np.sin(theta)
            verts.append([center[0] + x, center[1] + y, center[2] + z])
            uvs.append([i / segments, j / segments])

    faces = []
    for j in range(segments):
        for i in range(segments):
            a = j * (segments + 1) + i
            b = a + 1
            c = a + (segments + 1)
            d = c + 1
            faces.append([a, c, b])
            faces.append([b, c, d])

    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32), np.array(uvs, dtype=np.float32)


def make_eye_texture(size=512):
    """Procedural eye texture: sclera + amber iris + pupil + specular."""
    tex = np.full((size, size, 3), [232, 228, 240], dtype=np.uint8)  # warm sclera
    cx, cy = size // 2, size // 2
    iris_r = int(size * 0.22)
    pupil_r = int(size * 0.09)

    # Iris ring with radial detail
    for y in range(size):
        for x in range(size):
            dx, dy = x - cx, y - cy
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < iris_r:
                t = dist / iris_r
                if t < 0.4:
                    # Dark center
                    c = np.array([10, 10, 26])
                elif t < 0.85:
                    # Amber iris with radial variation
                    angle = np.arctan2(dy, dx)
                    variation = 0.15 * np.sin(angle * 12 + dist * 0.3)
                    base = np.array([0, 68 + variation*40, 204 + variation*30])  # BGR amber
                    fade = (t - 0.4) / 0.45
                    c = base * (1 - fade * 0.3)
                else:
                    # Dark rim
                    c = np.array([0, 17, 51])
                tex[y, x] = np.clip(c, 0, 255).astype(np.uint8)
            elif dist < iris_r + 2:
                tex[y, x] = [0, 17, 51]  # limbal ring

    # Pupil
    cv2.circle(tex, (cx, cy), pupil_r, (0, 0, 0), -1, cv2.LINE_AA)

    # Specular highlights
    cv2.circle(tex, (cx - int(size*0.06), cy - int(size*0.07)),
               int(size*0.025), (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(tex, (cx + int(size*0.03), cy - int(size*0.04)),
               int(size*0.012), (230, 230, 230), -1, cv2.LINE_AA)

    path = os.path.join(OUTPUT_DIR, "eye_texture.png")
    cv2.imwrite(path, tex)
    return path


def make_mouth_cavity(face_verts, scale=150.0):
    """Dark oral cavity behind lips. Ordered ring for proper triangulation."""
    # Ordered ring: upper lip L→R then lower lip R→L (clockwise loop)
    mouth_ring = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
                  324, 318, 402, 317, 14, 87, 178, 88, 95]

    mouth_verts = []
    for idx in mouth_ring:
        if idx < len(face_verts):
            mouth_verts.append(face_verts[idx].copy())

    if len(mouth_verts) < 3:
        return None, None, None

    mouth_verts = np.array(mouth_verts, dtype=np.float32)
    center = mouth_verts.mean(axis=0)

    n_ring = len(mouth_verts)
    cavity_verts = []
    for v in mouth_verts:
        cavity_verts.append(v)
    for v in mouth_verts:
        back = v.copy()
        back[2] -= 8.0
        back[:2] = back[:2] * 0.6 + center[:2] * 0.4
        back[1] -= 1.0
        cavity_verts.append(back)
    back_center = center.copy()
    back_center[2] -= 12.0
    back_center[1] -= 1.5
    cavity_verts.append(back_center)

    cavity_verts = np.array(cavity_verts, dtype=np.float32)

    # Triangulate: front-to-back quads + back ring to center
    faces = []
    back_center_idx = 2 * n_ring
    for i in range(n_ring):
        j = (i + 1) % n_ring
        # Quad between front and back rings
        faces.append([i, j, n_ring + j])
        faces.append([i, n_ring + j, n_ring + i])
    # Back cap
    for i in range(n_ring):
        j = (i + 1) % n_ring
        faces.append([n_ring + i, n_ring + j, back_center_idx])

    faces = np.array(faces, dtype=np.int32)
    # Simple UVs (not textured, just dark material)
    uvs = np.zeros((len(cavity_verts), 2), dtype=np.float32)

    return cavity_verts, faces, uvs


def make_eyelid_occluder(face_verts, eye_indices, depth_offset=-2.0):
    """Skin-colored disc behind eye opening to hide eyeball edges during blink."""
    rim_pts = np.array([face_verts[i] for i in eye_indices], dtype=np.float32)
    center = rim_pts.mean(axis=0)
    center[2] += depth_offset

    n_rim = len(rim_pts)
    verts = [center.copy()]
    for pt in rim_pts:
        v = pt.copy()
        v[2] += depth_offset
        verts.append(v)

    faces = []
    for i in range(n_rim):
        a = 1 + i
        b = 1 + (i + 1) % n_rim
        faces.append([0, b, a])

    verts = np.array(verts, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32)
    uvs = np.zeros((len(verts), 2), dtype=np.float32)
    return verts, faces, uvs


def make_cranium(face_verts, face_boundary_indices, scale=150.0):
    """Ellipsoidal back-of-head stitched to face boundary."""
    boundary = []
    for idx in face_boundary_indices:
        if idx < len(face_verts):
            boundary.append(face_verts[idx].copy())
    if len(boundary) < 4:
        return None, None, None

    boundary = np.array(boundary, dtype=np.float32)
    center = boundary.mean(axis=0)

    # Compute head dimensions for proper ellipsoid
    head_width = boundary[:, 0].max() - boundary[:, 0].min()
    head_height = boundary[:, 1].max() - boundary[:, 1].min()
    head_depth = head_width * 0.85  # roughly round skull

    n_boundary = len(boundary)
    n_rings = 8
    all_verts = list(boundary)

    for ring in range(1, n_rings + 1):
        t = ring / n_rings
        # Use cosine interpolation for smoother curve
        t_cos = 0.5 * (1 - np.cos(t * np.pi))
        for i in range(n_boundary):
            v = boundary[i].copy()
            # Shrink toward center as we go back (ellipsoidal)
            shrink = 1.0 - t_cos * 0.7
            v[0] = center[0] + (v[0] - center[0]) * shrink
            v[1] = center[1] + (v[1] - center[1]) * shrink * 0.9
            # Push Z back along ellipsoid curve
            v[2] = center[2] - t * head_depth * 0.7
            all_verts.append(v)

    # Back pole
    back_pole = center.copy()
    back_pole[2] = center[2] - head_depth * 0.75
    all_verts.append(back_pole)

    all_verts = np.array(all_verts, dtype=np.float32)
    back_pole_idx = len(all_verts) - 1

    faces = []
    for ring in range(n_rings):
        for i in range(n_boundary):
            j = (i + 1) % n_boundary
            a = ring * n_boundary + i
            b = ring * n_boundary + j
            c = (ring + 1) * n_boundary + i
            d = (ring + 1) * n_boundary + j
            # Winding: normals face outward (away from center)
            faces.append([a, d, b])
            faces.append([a, c, d])

    last_ring_start = n_rings * n_boundary
    for i in range(n_boundary):
        j = (i + 1) % n_boundary
        faces.append([last_ring_start + i, back_pole_idx, last_ring_start + j])

    faces = np.array(faces, dtype=np.int32)
    uvs = np.zeros((len(all_verts), 2), dtype=np.float32)

    return all_verts, faces, uvs


def get_face_boundary_indices():
    """MediaPipe face silhouette landmark indices (roughly jaw + forehead outline)."""
    return [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]


# ── Teeth geometry ────────────────────────────────────────────────────────

def make_simple_teeth(face_verts):
    """Simple teeth strips - upper and lower."""
    upper_lip_inner = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
    lower_lip_inner = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]

    teeth_verts = []
    teeth_faces = []

    for lip_indices, y_offset, is_upper in [(upper_lip_inner, 0.8, True),
                                             (lower_lip_inner, -0.8, False)]:
        base_idx = len(teeth_verts)
        for idx in lip_indices:
            if idx < len(face_verts):
                v = face_verts[idx].copy()
                v[2] -= 6.0  # well behind lips to avoid clipping through face
                teeth_verts.append(v.copy())
                v2 = v.copy()
                v2[1] += y_offset
                v2[2] -= 2.0
                teeth_verts.append(v2)

        n_pts = len(lip_indices)
        for i in range(n_pts - 1):
            a = base_idx + i * 2
            b = base_idx + i * 2 + 1
            c = base_idx + (i + 1) * 2
            d = base_idx + (i + 1) * 2 + 1
            if is_upper:
                teeth_faces.append([a, c, d])
                teeth_faces.append([a, d, b])
            else:
                teeth_faces.append([a, d, c])
                teeth_faces.append([a, b, d])

    if not teeth_verts:
        return None, None, None

    verts = np.array(teeth_verts, dtype=np.float32)
    faces = np.array(teeth_faces, dtype=np.int32)
    uvs = np.zeros((len(verts), 2), dtype=np.float32)
    return verts, faces, uvs


# ── Blendshapes ───────────────────────────────────────────────────────────

def generate_blendshapes(n):
    UPPER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
    LOWER_LIP = [146, 91, 181, 84, 17, 314, 405, 321, 375]
    INNER_UP = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
    INNER_LO = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
    MOUTH_L = [61, 78]
    MOUTH_R = [291, 308]
    JAW = [152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
           377, 400, 378, 379, 365, 397, 288, 361, 323]

    # Tight eyelid ring
    L_EYE_UPPER = [159, 158, 157, 173, 246, 161, 160]
    L_EYE_LOWER = [145, 153, 154, 155, 144, 163, 7]
    R_EYE_UPPER = [386, 385, 384, 398, 466, 388, 387]
    R_EYE_LOWER = [374, 380, 381, 382, 373, 390, 249]
    # Outer ring (half falloff) for smooth deformation
    L_EYE_OUTER_UP = [56, 28, 27, 29, 30, 247, 33, 130, 25, 110, 24, 23, 22, 26]
    L_EYE_OUTER_LO = [133, 243, 112, 26, 22, 23, 24, 110]
    R_EYE_OUTER_UP = [286, 258, 257, 259, 260, 467, 263, 359, 255, 339, 254, 253, 252, 256]
    R_EYE_OUTER_LO = [362, 463, 341, 256, 252, 253, 254, 339]

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
        "eyeBlinkLeft": d(L_EYE_UPPER, dy=-3.0) + d(L_EYE_LOWER, dy=1.5) + d(L_EYE_OUTER_UP, dy=-1.0) + d(L_EYE_OUTER_LO, dy=0.5),
        "eyeBlinkRight": d(R_EYE_UPPER, dy=-3.0) + d(R_EYE_LOWER, dy=1.5) + d(R_EYE_OUTER_UP, dy=-1.0) + d(R_EYE_OUTER_LO, dy=0.5),
        "browInnerUp": d(L_BROW + R_BROW, dy=2),
        "browDownLeft": d(L_BROW, dy=-1.5),
        "browDownRight": d(R_BROW, dy=-1.5),
    }


def pad_morph(delta, total_verts, original_verts):
    padded = np.zeros((total_verts, 3), dtype=np.float32)
    padded[:original_verts] = delta[:original_verts]
    return padded


# ── Multi-mesh GLB writer ─────────────────────────────────────────────────

class GLBBuilder:
    """Builds a multi-mesh GLB with different materials."""

    def __init__(self):
        self.buf = bytearray()
        self.buffer_views = []
        self.accessors = []
        self.meshes = []
        self.nodes = []
        self.materials = []
        self.textures = []
        self.samplers = []
        self.images = []

    def _pad(self):
        while len(self.buf) % 4:
            self.buf.extend(b'\x00')

    def add_buffer_view(self, data_bytes, target=None, stride=None):
        off = len(self.buf)
        self.buf.extend(data_bytes)
        self._pad()
        bv = {"buffer": 0, "byteOffset": off, "byteLength": len(data_bytes)}
        if target:
            bv["target"] = target
        if stride:
            bv["byteStride"] = stride
        idx = len(self.buffer_views)
        self.buffer_views.append(bv)
        return idx

    def add_accessor(self, bv_idx, comp_type, count, acc_type, min_val=None, max_val=None):
        acc = {"bufferView": bv_idx, "componentType": comp_type,
               "count": count, "type": acc_type}
        if min_val is not None:
            acc["min"] = min_val
        if max_val is not None:
            acc["max"] = max_val
        idx = len(self.accessors)
        self.accessors.append(acc)
        return idx

    def add_image(self, image_path, mime="image/png"):
        with open(image_path, 'rb') as f:
            data = f.read()
        bv = self.add_buffer_view(data)
        idx = len(self.images)
        self.images.append({"bufferView": bv, "mimeType": mime})
        return idx

    def add_texture(self, image_idx):
        if not self.samplers:
            self.samplers.append({"magFilter": 9729, "minFilter": 9987,
                                  "wrapS": 33071, "wrapT": 33071})
        idx = len(self.textures)
        self.textures.append({"source": image_idx, "sampler": 0})
        return idx

    def add_material(self, name, base_color=None, texture_idx=None,
                     metallic=0.0, roughness=0.5, double_sided=True,
                     emissive=None, unlit=False):
        pbr = {"metallicFactor": metallic, "roughnessFactor": roughness}
        if texture_idx is not None:
            pbr["baseColorTexture"] = {"index": texture_idx}
        if base_color is not None:
            pbr["baseColorFactor"] = base_color
        mat = {"name": name, "pbrMetallicRoughness": pbr, "doubleSided": double_sided}
        if emissive:
            mat["emissiveFactor"] = emissive
        if unlit:
            mat.setdefault("extensions", {})["KHR_materials_unlit"] = {}
        idx = len(self.materials)
        self.materials.append(mat)
        return idx

    def add_mesh(self, name, verts, faces, normals, uvs, material_idx,
                 morph_targets=None, morph_names=None):
        verts_f32 = verts.astype(np.float32)
        normals_f32 = normals.astype(np.float32)
        uvs_f32 = uvs.astype(np.float32)
        indices = faces.flatten().astype(np.uint16 if verts.shape[0] < 65536 else np.uint32)
        idx_comp = 5123 if indices.dtype == np.uint16 else 5125

        idx_bv = self.add_buffer_view(indices.tobytes(), target=34963)
        pos_bv = self.add_buffer_view(verts_f32.tobytes(), target=34962, stride=12)
        norm_bv = self.add_buffer_view(normals_f32.tobytes(), target=34962, stride=12)
        uv_bv = self.add_buffer_view(uvs_f32.tobytes(), target=34962, stride=8)

        idx_acc = self.add_accessor(idx_bv, idx_comp, len(indices), "SCALAR",
                                    [int(indices.min())], [int(indices.max())])
        pos_acc = self.add_accessor(pos_bv, 5126, len(verts_f32), "VEC3",
                                    verts_f32.min(axis=0).tolist(), verts_f32.max(axis=0).tolist())
        norm_acc = self.add_accessor(norm_bv, 5126, len(normals_f32), "VEC3")
        uv_acc = self.add_accessor(uv_bv, 5126, len(uvs_f32), "VEC2")

        prim = {"attributes": {"POSITION": pos_acc, "NORMAL": norm_acc, "TEXCOORD_0": uv_acc},
                "indices": idx_acc, "material": material_idx}

        morph_tgts = []
        if morph_targets and morph_names:
            for mname in morph_names:
                delta = morph_targets[mname].astype(np.float32)
                mbv = self.add_buffer_view(delta.tobytes(), stride=12)
                macc = self.add_accessor(mbv, 5126, len(delta), "VEC3",
                                         delta.min(axis=0).tolist(), delta.max(axis=0).tolist())
                morph_tgts.append({"POSITION": macc})
            prim["targets"] = morph_tgts

        mesh_def = {"name": name, "primitives": [prim]}
        if morph_names:
            mesh_def["extras"] = {"targetNames": morph_names}
            mesh_def["weights"] = [0.0] * len(morph_names)

        mesh_idx = len(self.meshes)
        self.meshes.append(mesh_def)

        node = {"mesh": mesh_idx, "name": name}
        node_idx = len(self.nodes)
        self.nodes.append(node)
        return node_idx

    def export(self, filepath):
        node_indices = list(range(len(self.nodes)))
        gltf = {
            "asset": {"version": "2.0", "generator": "unum-avatar-v5"},
            "scene": 0,
            "scenes": [{"nodes": node_indices}],
            "nodes": self.nodes,
            "meshes": self.meshes,
            "accessors": self.accessors,
            "bufferViews": self.buffer_views,
            "materials": self.materials,
            "buffers": [{"byteLength": len(self.buf)}]
        }
        if self.textures:
            gltf["textures"] = self.textures
        if self.samplers:
            gltf["samplers"] = self.samplers
        if self.images:
            gltf["images"] = self.images

        # Check if any material uses KHR_materials_unlit
        uses_unlit = any("extensions" in m and "KHR_materials_unlit" in m.get("extensions", {})
                         for m in self.materials)
        if uses_unlit:
            gltf["extensionsUsed"] = ["KHR_materials_unlit"]

        jb = json.dumps(gltf, separators=(',', ':')).encode('utf-8')
        while len(jb) % 4:
            jb += b' '
        while len(self.buf) % 4:
            self.buf.extend(b'\x00')

        with open(filepath, 'wb') as f:
            f.write(struct.pack('<III', 0x46546C67, 2, 12 + 8 + len(jb) + 8 + len(self.buf)))
            f.write(struct.pack('<II', len(jb), 0x4E4F534A))
            f.write(jb)
            f.write(struct.pack('<II', len(self.buf), 0x004E4942))
            f.write(self.buf)


# ── Lip-sync smoothing ────────────────────────────────────────────────────

def smooth_lipsync(json_path, attack_ms=50, release_ms=120):
    """Apply temporal smoothing to lip-sync animation data."""
    with open(json_path) as f:
        data = json.load(f)

    frames = data["frames"]
    if len(frames) < 2:
        return

    fps = data.get("fps", 30)
    attack_frames = max(1, int(attack_ms / 1000 * fps))
    release_frames = max(1, int(release_ms / 1000 * fps))

    keys = [k for k in frames[0].keys() if k != "time"]
    for key in keys:
        values = [f[key] for f in frames]
        smoothed = [values[0]]
        for i in range(1, len(values)):
            if values[i] > smoothed[-1]:
                alpha = 1.0 / attack_frames
            else:
                alpha = 1.0 / release_frames
            smoothed.append(smoothed[-1] + alpha * (values[i] - smoothed[-1]))
        for i, f in enumerate(frames):
            f[key] = round(smoothed[i], 4)

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=== Unum Avatar Head v5 (eyes + mouth + cranium) ===\n")

    # Detection
    print("[1/9] MediaPipe landmarks...")
    mp_pts = detect_mediapipe(INPUT_IMAGE)
    max_v = 468
    print(f"  {len(mp_pts)} landmarks")

    print("[2/9] InsightFace 3D depth...")
    if_pts_3d, img_shape = detect_insightface(INPUT_IMAGE)
    print(f"  68 landmarks, Z: {if_pts_3d[:,2].min():.0f}–{if_pts_3d[:,2].max():.0f}")

    print("[3/9] Interpolating depth + building mesh...")
    z_depth = interpolate_depth(mp_pts[:max_v], if_pts_3d, img_shape)
    faces = get_face_mesh_triangles()
    faces = faces[np.all(faces < max_v, axis=1)]
    verts = landmarks_to_3d(mp_pts[:max_v], z_depth)
    faces = fix_winding(verts, faces)
    uvs = compute_uvs(mp_pts, max_v)
    print(f"  {len(faces)} base tris")

    print("[4/9] Subdividing...")
    verts, faces, uvs, edge_parents = subdivide_mesh(verts, faces, uvs, n_iters=2)
    normals = compute_normals(verts, faces)
    print(f"  {len(verts)} verts, {len(faces)} tris")

    print("[5/9] Baking texture...")
    tex_path = bake_texture(INPUT_IMAGE, mp_pts, uvs, faces, max_v, TEX_SIZE)

    print("[6/9] Blendshapes...")
    morphs = generate_blendshapes(len(verts))
    morphs = {name: subdivide_deltas(delta[:max_v], edge_parents) for name, delta in morphs.items()}
    for name in morphs:
        d = morphs[name]
        if len(d) < len(verts):
            padded = np.zeros((len(verts), 3), dtype=np.float32)
            padded[:len(d)] = d
            morphs[name] = padded
    morph_names = list(morphs.keys())

    # Build GLB
    print("[7/9] Building multi-mesh GLB...")
    glb = GLBBuilder()

    # Materials
    skin_img = glb.add_image(tex_path)
    skin_tex = glb.add_texture(skin_img)
    skin_mat = glb.add_material("Skin", texture_idx=skin_tex, roughness=0.6)

    eye_tex_path = make_eye_texture()
    eye_img = glb.add_image(eye_tex_path)
    eye_tex = glb.add_texture(eye_img)
    eye_mat = glb.add_material("Eye", texture_idx=eye_tex, roughness=0.05, metallic=0.0)

    mouth_mat = glb.add_material("MouthCavity", base_color=[0.15, 0.05, 0.05, 1.0],
                                  roughness=0.9, metallic=0.0)
    teeth_mat = glb.add_material("Teeth", base_color=[0.9, 0.88, 0.82, 1.0],
                                  roughness=0.4, metallic=0.0)
    # Sample average skin color from texture for cranium
    skin_img = cv2.imread(tex_path)
    avg_skin = skin_img[skin_img.shape[0]//4:skin_img.shape[0]//2,
                         skin_img.shape[1]//3:2*skin_img.shape[1]//3].mean(axis=(0,1)) / 255.0
    cranium_color = [float(avg_skin[2]), float(avg_skin[1]), float(avg_skin[0]), 1.0]  # BGR→RGB
    cranium_mat = glb.add_material("Cranium", base_color=cranium_color,
                                    roughness=0.8, metallic=0.0)

    # Face mesh (with morphs)
    glb.add_mesh("Face", verts, faces, normals, uvs, skin_mat,
                 morph_targets=morphs, morph_names=morph_names)

    # Eyeballs — use original (pre-subdivision) verts for positioning
    base_verts = verts[:max_v]  # original landmark verts (before subdivision added more)
    L_EYE = [159, 145, 33, 133]
    R_EYE = [386, 374, 263, 362]

    for eye_indices in [L_EYE, R_EYE]:
        center = np.mean([base_verts[i] for i in eye_indices], axis=0)
        center[2] -= 8.0  # push very deep to prevent sclera clipping through face
        ev, ef, euv = make_eyeball(center, radius=1.5, segments=20)
        en = compute_normals(ev, ef)
        glb.add_mesh("Eyeball", ev, ef, en, euv, eye_mat)

    # Mouth cavity
    print("[8/9] Adding mouth cavity + teeth + cranium...")
    cv, cf, cuv = make_mouth_cavity(base_verts)
    if cv is not None:
        cn = compute_normals(cv, cf)
        glb.add_mesh("MouthCavity", cv, cf, cn, cuv, mouth_mat)

    # Cranium (back of head)
    boundary_indices = get_face_boundary_indices()
    crv, crf, cruv = make_cranium(base_verts, boundary_indices)
    if crv is not None:
        crn = compute_normals(crv, crf)
        glb.add_mesh("Cranium", crv, crf, crn, cruv, cranium_mat)

    print("[9/9] Exporting...")
    glb_path = os.path.join(OUTPUT_DIR, "unum_head_v5.glb")
    glb.export(glb_path)

    total_verts = len(verts)
    total_tris = len(faces)
    print(f"  GLB: {glb_path} ({os.path.getsize(glb_path)/1024/1024:.1f} MB)")
    print(f"  Face: {len(verts)} verts, {len(faces)} tris, {len(morphs)} morphs")
    print(f"\n=== Done! Update viewer to load unum_head_v5.glb ===")

    # Smooth existing lipsync if present
    ls_path = os.path.join(OUTPUT_DIR, "speech_lipsync.json")
    if os.path.exists(ls_path):
        print("  Smoothing lip-sync data...")
        smooth_lipsync(ls_path)


if __name__ == "__main__":
    main()
