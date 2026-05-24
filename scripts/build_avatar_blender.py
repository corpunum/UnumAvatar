"""
Blender Python script: high-quality Unum avatar GLB builder.
Run with: blender --background --python scripts/build_avatar_blender.py

Produces: output/unum_avatar_v6.glb
  - MediaPipe-density face mesh with proper subdivision
  - Correct eyelid rim occluders (no sclera bleed)
  - Eyeballs with amber iris texture
  - Mouth cavity + teeth strip + tongue stub
  - Cranium/back-of-head shell
  - Neck stub
  - 12 shape keys (all interpolated through subdivision)
  - Multi-view baked skin texture (2048x2048)
  - Clean glTF export
"""

import sys, os, json, math

# Remove any external venv paths that conflict with Blender's Python
sys.path = [p for p in sys.path if "python3.12" not in p and "/venv/" not in p]

import bpy, bmesh, mathutils
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────
BASE = os.path.expanduser("~/3d-avatar")
INPUT  = os.path.join(BASE, "input")
OUTPUT = os.path.join(BASE, "output")
DATA   = os.path.join(BASE, "data")

FRONT_IMG   = os.path.join(INPUT, "unum-clean-front-v2.png")
LEFT_IMG    = os.path.join(INPUT, "unum-clean-left-profile.png")
RIGHT_IMG   = os.path.join(INPUT, "unum-clean-right-profile.png")
MODEL_PATH  = os.path.join(DATA, "face_landmarker.task")
OUT_GLB     = os.path.join(OUTPUT, "unum_avatar_v6.glb")

TEX_SIZE = 2048


# ── Step 1: Get landmarks via MediaPipe ───────────────────────────────────

def get_landmarks():
    import mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions
    from scipy.interpolate import RBFInterpolator

    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        output_face_blendshapes=True, num_faces=1
    )
    lm = vision.FaceLandmarker.create_from_options(options)
    img = mp.Image.create_from_file(FRONT_IMG)
    result = lm.detect(img)
    lm.close()

    pts_2d = np.array([[p.x * img.width, p.y * img.height]
                       for p in result.face_landmarks[0]], dtype=np.float64)

    # InsightFace depth
    import insightface
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1, det_size=(640, 640))
    cv_img = __import__('cv2').imread(FRONT_IMG)
    faces = app.get(cv_img)
    if_3d = faces[0].landmark_3d_68

    img_h, img_w = cv_img.shape[:2]
    if_norm = if_3d.copy()
    if_norm[:, 0] = if_3d[:, 0] / img_w * pts_2d[:, 0].max()
    if_norm[:, 1] = if_3d[:, 1] / img_h * pts_2d[:, 1].max()

    rbf = RBFInterpolator(if_norm[:, :2], if_norm[:, 2],
                          kernel="thin_plate_spline", smoothing=1.0)
    z_raw = rbf(pts_2d[:468])
    z_min, z_max = z_raw.min(), z_raw.max()
    z_norm = (z_raw - z_min) / (z_max - z_min + 1e-8)

    # Normalize XY to scene units (head ~20 units wide)
    max_v = 468
    face_pts = pts_2d[:max_v]
    x_min, y_min = face_pts.min(axis=0)
    x_max, y_max = face_pts.max(axis=0)
    scale = 20.0 / (x_max - x_min)

    # Intuitive space: X=right, Y=up, Z=toward-viewer (glTF-native)
    verts = np.zeros((max_v, 3), dtype=np.float64)
    verts[:, 0] = (face_pts[:, 0] - (x_min + x_max) / 2) * scale
    verts[:, 1] = -((face_pts[:, 1] - (y_min + y_max) / 2) * scale)  # invert image Y
    verts[:, 2] = z_norm * 6.0 - 1.5

    # UV coordinates will be remapped after bake_texture() determines crop bounds
    # Store raw pixel coords for now; they'll be normalized in main()
    return verts, face_pts.copy(), img.width, img.height


def get_triangles():
    import mediapipe as mp
    from mediapipe.tasks.python.vision import FaceLandmarksConnections
    edges = set()
    for conn in FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION:
        edges.add((conn.start, conn.end))
    adj = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    tris = set()
    for a, b in edges:
        for c in adj.get(a, set()) & adj.get(b, set()):
            tris.add(tuple(sorted([a, b, c])))
    return [t for t in tris if all(i < 468 for i in t)]


# ── Step 2: Bake multi-view texture ───────────────────────────────────────

def bake_texture(mp_pts_raw, img_w, img_h):
    """Use photo crop directly as texture — no per-triangle bake, no seam artifacts.
    UV coordinates are mapped to the crop region, so GPU interpolation is seamless."""
    import cv2 as cv

    front = cv.imread(FRONT_IMG)
    src_h, src_w = front.shape[:2]

    max_v = 468
    face_pts = mp_pts_raw[:max_v, :2]
    x_min, y_min = face_pts.min(axis=0)
    x_max, y_max = face_pts.max(axis=0)

    # Add generous padding around face so ears/forehead/chin are included
    pad_x = (x_max - x_min) * 0.15
    pad_y = (y_max - y_min) * 0.15
    cx0 = max(0, int(x_min - pad_x))
    cy0 = max(0, int(y_min - pad_y))
    cx1 = min(src_w, int(x_max + pad_x))
    cy1 = min(src_h, int(y_max + pad_y))

    crop = front[cy0:cy1, cx0:cx1]
    crop_h, crop_w = crop.shape[:2]

    # Sharpen lightly
    blur = cv.GaussianBlur(crop, (0, 0), 1.2)
    crop = cv.addWeighted(crop, 1.4, blur, -0.4, 0)
    crop = np.clip(crop, 0, 255).astype(np.uint8)

    # Resize to TEX_SIZE
    texture = cv.resize(crop, (TEX_SIZE, TEX_SIZE), interpolation=cv.INTER_LANCZOS4)

    # Apply soft elliptical alpha fade so face blends into background at edges
    h, w = texture.shape[:2]
    texture_rgba = cv.cvtColor(texture, cv.COLOR_BGR2BGRA)
    Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
    cx_t, cy_t = w * 0.50, h * 0.54  # slightly below center — nose area is face center
    # Normalized elliptical distance — face slightly taller than wide
    dx = (X - cx_t) / (cx_t * 0.82)
    dy = (Y - cy_t) / (cy_t * 0.90)
    dist = np.sqrt(dx**2 + dy**2)
    # Fade: fully opaque inside radius 0.58, transparent beyond 1.0 — wider fade zone
    alpha_f = np.clip(1.0 - (dist - 0.58) / 0.42, 0.0, 1.0)
    alpha_f = np.power(alpha_f, 0.45)  # smooth gamma — more gradual rolloff

    # Punch a transparent hole for the inner mouth so jaw-open reveals the cavity.
    # Inner mouth region: use MediaPipe inner lip landmark positions in crop-UV space.
    # These are approximate centers derived from typical face layout:
    # inner mouth center is at ~53% across, ~70% down in the cropped texture.
    # Use actual inner-lip landmark positions to place mouth hole accurately.
    # mp_pts_raw is in original image pixels; convert to crop-texture UV space.
    INNER_UP  = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
    INNER_LO  = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324]
    inner_up_pts = np.array([mp_pts_raw[i, :2] for i in INNER_UP
                              if i < mp_pts_raw.shape[0]])
    inner_lo_pts = np.array([mp_pts_raw[i, :2] for i in INNER_LO
                              if i < mp_pts_raw.shape[0]])
    # Convert to texture pixel space
    def to_tex(pts):
        u = (pts[:, 0] - cx0) / (cx1 - cx0) * w
        v = (pts[:, 1] - cy0) / (cy1 - cy0) * h
        return u, v
    up_u, up_v = to_tex(inner_up_pts)
    lo_u, lo_v = to_tex(inner_lo_pts)
    # Mouth hole: anchored at inner upper lip — exposed when lower jaw drops.
    # In source photo the mouth is barely open, so we use fixed pixel height.
    # Very thin horizontal slit at INNER upper lip only — barely visible at rest,
    # exposed when lower jaw drops away.
    mx_t = up_u.mean()
    my_t = up_v.mean()   # anchor at inner upper lip
    mw = (up_u.max() - up_u.min()) * 0.42   # half-width
    mh_t = 14.0  # 28px tall — visible when jaw opens, subtle at rest
    mdx = (X - mx_t) / max(mw, 1)
    mdy = (Y - my_t) / max(mh_t, 1)
    mouth_dist = np.sqrt(mdx**2 + mdy**2)
    mouth_alpha_inv = np.clip((mouth_dist - 0.5) / 0.5, 0.0, 1.0)
    alpha_f = alpha_f * mouth_alpha_inv

    texture_rgba[:, :, 3] = (alpha_f * 255).astype(np.uint8)

    out = os.path.join(OUTPUT, "unum_skin_v6.png")
    cv.imwrite(out, texture_rgba)

    # Return crop bounds so UV generation can match
    return out, cx0, cy0, cx1, cy1


# ── Step 3: Build in Blender ───────────────────────────────────────────────

def clear_scene():
    # Ensure we have a valid context window for headless operation
    if not bpy.context.window:
        bpy.context.window_manager.windows.update()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    # Create a fresh scene and collection
    scene = bpy.context.scene
    if not scene:
        scene = bpy.data.scenes.new("Scene")
    col = bpy.data.collections.new("Collection")
    scene.collection.children.link(col)
    bpy.context.view_layer.active_layer_collection = \
        bpy.context.view_layer.layer_collection.children[col.name]


def make_material(name, img_path=None, base_color=None,
                  roughness=0.5, metallic=0.0, alpha=1.0, unlit=False,
                  use_alpha_channel=False):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    if unlit:
        emit = nodes.new("ShaderNodeEmission")
        if img_path:
            tex = nodes.new("ShaderNodeTexImage")
            img = bpy.data.images.load(img_path)
            tex.image = img
            links.new(tex.outputs["Color"], emit.inputs["Color"])
        elif base_color:
            emit.inputs["Color"].default_value = (*base_color[:3], 1)
        emit.inputs["Strength"].default_value = 1.0
        links.new(emit.outputs["Emission"], out.inputs["Surface"])
    else:
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Roughness"].default_value = roughness
        bsdf.inputs["Metallic"].default_value = metallic
        if img_path:
            tex = nodes.new("ShaderNodeTexImage")
            img = bpy.data.images.load(img_path)
            tex.image = img
            links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
            if use_alpha_channel:
                links.new(tex.outputs["Alpha"], bsdf.inputs["Alpha"])
        elif base_color:
            bsdf.inputs["Base Color"].default_value = (*base_color[:3], alpha)
        links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    mat.blend_method = 'BLEND' if use_alpha_channel else 'OPAQUE'
    mat.show_transparent_back = False
    return mat


def add_face_mesh(verts_3d, tris, uvs_2d, tex_path):
    """Create face mesh object with UV map and skin texture."""
    mesh = bpy.data.meshes.new("FaceMesh")
    verts_list = [tuple(v) for v in verts_3d]
    # Reverse winding: Y-flip in get_landmarks() changes CCW→CW, so restore CCW
    faces_list = [(t[0], t[2], t[1]) for t in tris]
    mesh.from_pydata(verts_list, [], faces_list)
    mesh.validate()

    # UV map
    mesh.uv_layers.new(name="UVMap")
    uv_layer = mesh.uv_layers.active.data
    for poly in mesh.polygons:
        for li in poly.loop_indices:
            vi = mesh.loops[li].vertex_index
            uv_layer[li].uv = (uvs_2d[vi][0], 1.0 - uvs_2d[vi][1])

    obj = bpy.data.objects.new("Face", mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj

    # Simple (Loop) subdivision — preserves original vertex positions for correct shape key mapping
    sub = obj.modifiers.new("Subdivision", "SUBSURF")
    sub.levels = 2
    sub.render_levels = 2
    sub.subdivision_type = 'SIMPLE'

    # Apply subdivision so shape keys work correctly
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier="Subdivision")

    # Recalculate normals to all point outward, then smooth shade
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.editmode_toggle()
    bpy.ops.object.shade_smooth()
    # Also set per-polygon smooth flag
    mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))

    # Material
    mat = make_material("Skin", img_path=tex_path, roughness=0.55, metallic=0.0,
                        use_alpha_channel=True)
    obj.data.materials.append(mat)

    return obj


def add_shape_keys(face_obj, verts_3d):
    """Add all 12 blendshape shape keys with proper interpolation."""
    mesh = face_obj.data
    n = len(mesh.vertices)
    n_orig = len(verts_3d)

    # Blendshape definitions on original landmarks
    UPPER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
    LOWER_LIP = [146, 91, 181, 84, 17, 314, 405, 321, 375]
    INNER_UP  = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
    INNER_LO  = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
    MOUTH_L   = [61, 78]
    MOUTH_R   = [291, 308]
    JAW       = [152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
                 377, 400, 378, 379, 365, 397, 288, 361, 323]

    L_EYE_UPPER    = [159, 158, 157, 173, 246, 161, 160]
    L_EYE_LOWER    = [145, 153, 154, 155, 144, 163, 7]
    L_EYE_OUTER_UP = [56, 247, 33, 130, 25, 110, 24, 23, 22, 26]
    L_EYE_OUTER_LO = [133, 243, 112, 26, 22, 23, 24, 110]
    R_EYE_UPPER    = [386, 385, 384, 398, 466, 388, 387]
    R_EYE_LOWER    = [374, 380, 381, 382, 373, 390, 249]
    R_EYE_OUTER_UP = [286, 467, 263, 359, 255, 339, 254, 253, 252, 256]
    R_EYE_OUTER_LO = [362, 463, 341, 256, 252, 253, 254, 339]
    L_BROW = [70, 63, 105, 66, 107, 55, 65]
    R_BROW = [300, 293, 334, 296, 336, 285, 295]

    def make_delta(indices_dxyz):
        d = np.zeros((n_orig, 3), dtype=np.float64)
        for idxs, dx, dy, dz in indices_dxyz:
            for i in idxs:
                if i < n_orig:
                    d[i] += [dx, dy, dz]
        return d

    sc = 0.6    # eye/brow scale
    sm = 0.65   # mouth/jaw scale — larger for clearly visible jaw drop

    shapes = {
        "Basis": make_delta([]),
        "jawOpen": make_delta([
            (LOWER_LIP + INNER_LO, 0, -2.5*sm, 0),
            (JAW, 0, -1.8*sm, 0),
        ]),
        "mouthClose": make_delta([
            (LOWER_LIP, 0, 1.2*sm, 0),
            (UPPER_LIP, 0, -0.4*sm, 0),
        ]),
        "mouthFunnel": make_delta([
            (LOWER_LIP+INNER_LO, 0, -1.2*sm, 1.5*sm),
            (UPPER_LIP+INNER_UP, 0, 0.7*sm, 1.5*sm),
            (MOUTH_L, 0.8*sm, 0, 0),
            (MOUTH_R, -0.8*sm, 0, 0),
        ]),
        "mouthPucker": make_delta([
            (UPPER_LIP+INNER_UP+LOWER_LIP+INNER_LO, 0, 0, 2.0*sm),
            (MOUTH_L, 1.5*sm, 0, 0),
            (MOUTH_R, -1.5*sm, 0, 0),
        ]),
        "mouthSmileLeft": make_delta([(MOUTH_L, -1.5*sm, 1.2*sm, 0)]),
        "mouthSmileRight": make_delta([(MOUTH_R, 1.5*sm, 1.2*sm, 0)]),
        "mouthOpen": make_delta([
            (LOWER_LIP+INNER_LO+JAW, 0, -3.0*sm, 0),
            (MOUTH_L, -1.0*sm, 0, 0),
            (MOUTH_R, 1.0*sm, 0, 0),
        ]),
        "eyeBlinkLeft": make_delta([
            (L_EYE_UPPER, 0, -1.5*sc, -0.2*sc),
            (L_EYE_LOWER, 0, 0.7*sc, -0.15*sc),
            (L_EYE_OUTER_UP, 0, -0.5*sc, -0.1*sc),
            (L_EYE_OUTER_LO, 0, 0.25*sc, 0),
        ]),
        "eyeBlinkRight": make_delta([
            (R_EYE_UPPER, 0, -1.5*sc, -0.2*sc),
            (R_EYE_LOWER, 0, 0.7*sc, -0.15*sc),
            (R_EYE_OUTER_UP, 0, -0.5*sc, -0.1*sc),
            (R_EYE_OUTER_LO, 0, 0.25*sc, 0),
        ]),
        "browInnerUp": make_delta([(L_BROW+R_BROW, 0, sc, 0)]),
        "browDownLeft": make_delta([(L_BROW, 0, -0.8*sc, 0)]),
        "browDownRight": make_delta([(R_BROW, 0, -0.8*sc, 0)]),
    }

    # Add shape keys — Blender applies them AFTER subdivision,
    # but since we already applied subdivision, we work on the full mesh.
    # We need to map original vertex deltas to subdivided vertex indices.
    # Blender records the nearest original vertex for each subdivided vertex.
    # Strategy: build KD-tree from original landmarks, find nearest for each vert.
    from scipy.spatial import KDTree
    orig_tree = KDTree(verts_3d)
    vert_pos = np.array([v.co for v in mesh.vertices])
    _, nearest_orig = orig_tree.query(vert_pos)

    # Blend weight by distance (closer = more influence)
    _, dists = orig_tree.query(vert_pos, k=4)
    weights_raw = 1.0 / (dists + 1e-6)
    _, k_idxs = orig_tree.query(vert_pos, k=4)

    # Add basis
    if not face_obj.data.shape_keys:
        face_obj.shape_key_add(name="Basis", from_mix=False)

    for sname, delta in shapes.items():
        if sname == "Basis":
            continue
        sk = face_obj.shape_key_add(name=sname, from_mix=False)
        sk.value = 0.0
        for vi, v in enumerate(mesh.vertices):
            # Weighted average of 4 nearest original vertex deltas
            ws = weights_raw[vi]
            ws /= ws.sum()
            ki = k_idxs[vi]
            d = sum(ws[j] * delta[ki[j]] for j in range(4) if ki[j] < n_orig)
            sk.data[vi].co = (
                v.co[0] + d[0],
                v.co[1] + d[1],
                v.co[2] + d[2],
            )

    print(f"  Shape keys: {[sk.name for sk in face_obj.data.shape_keys.key_blocks]}")


def add_eyeball(cx, cy, face_z, iris_r, name="Eyeball", eye_patch_path=None):
    """Photo-eye patch quad: crops from the original photo placed as a flat card
    at the eye socket surface. Looks exactly like the original eye — no procedural iris.

    cx, cy          : XY of eye center in scene units
    face_z          : Z of face mesh at eye position
    iris_r          : radius in scene units (used to size the patch quad)
    eye_patch_path  : path to cropped eye PNG from the original photo
    """
    import cv2 as cv

    if eye_patch_path and os.path.exists(eye_patch_path):
        # Load and resize eye patch, add soft alpha edge so it blends with face
        patch = cv.imread(eye_patch_path, cv.IMREAD_COLOR)
        ph, pw = patch.shape[:2]
        patch_rgba = cv.cvtColor(patch, cv.COLOR_BGR2BGRA)
        # Soft elliptical mask — hides hard rectangular edges
        Y, X = np.mgrid[0:ph, 0:pw].astype(np.float32)
        cx_p, cy_p = pw / 2, ph / 2
        dx = (X - cx_p) / (cx_p * 0.85)
        dy = (Y - cy_p) / (cy_p * 0.80)
        dist_p = np.sqrt(dx**2 + dy**2)
        alpha_p = np.clip(1.0 - (dist_p - 0.70) / 0.30, 0.0, 1.0)
        alpha_p = np.power(alpha_p, 0.5)
        patch_rgba[:, :, 3] = (alpha_p * 255).astype(np.uint8)
        patch_tex_path = os.path.join(OUTPUT, f"{name}_patch.png")
        cv.imwrite(patch_tex_path, patch_rgba)
    else:
        # Fallback: white sclera disc if no patch available
        patch_tex_path = None

    # Quad geometry — sized to match eye socket
    hw = iris_r * 1.8   # half-width of quad
    hh = hw * 0.55      # eye aspect ratio ~2:1
    patch_z = face_z + 0.5
    verts = [
        (cx - hw, cy - hh, patch_z),
        (cx + hw, cy - hh, patch_z),
        (cx + hw, cy + hh, patch_z),
        (cx - hw, cy + hh, patch_z),
    ]
    faces = [[0, 1, 2, 3]]
    uvs = [(0, 1), (1, 1), (1, 0), (0, 0)]  # flip V

    mesh = bpy.data.meshes.new(name + "_PatchMesh")
    mesh.from_pydata(verts, [], faces)
    mesh.validate()
    mesh.uv_layers.new(name="UVMap")
    uv_layer = mesh.uv_layers.active.data
    for poly in mesh.polygons:
        for li, vi in zip(poly.loop_indices, [0, 1, 2, 3]):
            uv_layer[li].uv = uvs[vi]

    obj = bpy.data.objects.new(name, mesh)
    obj.name = name
    bpy.context.collection.objects.link(obj)

    mat = bpy.data.materials.new(name + "_PatchMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    out_n  = nodes.new("ShaderNodeOutputMaterial")
    bsdf   = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 1.0
    if patch_tex_path:
        tex_node = nodes.new("ShaderNodeTexImage")
        img = bpy.data.images.load(patch_tex_path)
        img.colorspace_settings.name = 'sRGB'
        tex_node.image = img
        links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
        links.new(tex_node.outputs["Alpha"], bsdf.inputs["Alpha"])
        mat.blend_method = 'BLEND'
    else:
        bsdf.inputs["Base Color"].default_value = (0.95, 0.93, 0.90, 1.0)
    links.new(bsdf.outputs["BSDF"], out_n.inputs["Surface"])
    obj.data.materials.append(mat)
    return obj

    # (old iris/sclera disc code removed)


def add_eyelid_rim(face_obj, rim_vert_indices, offset_z=-0.12, name="EyelidRim"):
    """Thin skin-toned rim mesh around eye opening to occlude eyeball edges."""
    mesh = face_obj.data
    vert_pos = np.array([v.co for v in mesh.vertices])

    # Map original landmark indices to nearest subdivided verts
    from scipy.spatial import KDTree
    import mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions

    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        output_face_blendshapes=False, num_faces=1
    )
    lm = vision.FaceLandmarker.create_from_options(options)
    img = mp.Image.create_from_file(FRONT_IMG)
    result = lm.detect(img)
    lm.close()

    pts_2d = np.array([[p.x, p.y] for p in result.face_landmarks[0]])

    # We need the 3D positions of rim indices in the face object
    face_pts_raw = np.array([[p.x * img.width, p.y * img.height]
                              for p in result.face_landmarks[0][:468]])
    x_min, y_min = face_pts_raw.min(axis=0)
    x_max, y_max = face_pts_raw.max(axis=0)
    scale = 20.0 / (x_max - x_min)

    rim_3d = np.zeros((len(rim_vert_indices), 3))
    for j, vi in enumerate(rim_vert_indices):
        x = (face_pts_raw[vi, 0] - (x_min + x_max)/2) * scale
        y = -((face_pts_raw[vi, 1] - (y_min + y_max)/2) * scale)
        rim_3d[j] = [x, y, 0]

    # Find z from face mesh at these positions
    tree = KDTree(vert_pos[:, :2])
    _, nn_idx = tree.query(rim_3d[:, :2])
    for j in range(len(rim_vert_indices)):
        rim_3d[j, 2] = vert_pos[nn_idx[j], 2] + offset_z

    center = rim_3d.mean(axis=0)
    verts = [center] + list(rim_3d)
    n = len(rim_3d)
    faces = [[0, i+1, (i+1)%n+1] for i in range(n)]

    rim_mesh = bpy.data.meshes.new(name + "Mesh")
    rim_mesh.from_pydata([tuple(v) for v in verts], [], faces)
    rim_mesh.validate()

    rim_obj = bpy.data.objects.new(name, rim_mesh)
    bpy.context.collection.objects.link(rim_obj)
    return rim_obj


def add_mouth_cavity(verts_3d):
    """Dark mouth interior — no front ring, so nothing bleeds through closed lips.
    Consists of mid-ring → back-ring side walls + back cap.
    The mid-ring sits behind the lip plane; only visible when the jaw is open."""
    mouth_ring_orig = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
                       324, 318, 402, 317, 14, 87, 178, 88, 95]
    mouth_pts = np.array([verts_3d[i] for i in mouth_ring_orig if i < len(verts_3d)])
    center = mouth_pts.mean(axis=0)
    n = len(mouth_pts)

    # Mid ring: inset 30% and pushed 0.9 units behind lip surface
    mid_z = center[2] - 0.9
    mid_ring = []
    for v in mouth_pts:
        mv = np.array(v)
        mv[:2] = mv[:2] * 0.70 + center[:2] * 0.30
        mv[2] = mid_z
        mid_ring.append(tuple(mv))

    # Back ring: inset 50%, pushed 2 units behind
    back_z = center[2] - 2.0
    back_ring = []
    for v in mouth_pts:
        bv = np.array(v)
        bv[:2] = bv[:2] * 0.45 + center[:2] * 0.55
        bv[1] -= 0.10
        bv[2] = back_z
        back_ring.append(tuple(bv))

    # Back center cap
    bc = np.array(center)
    bc[2] = center[2] - 2.4
    bc[1] -= 0.12

    cavity_verts = mid_ring + back_ring + [tuple(bc)]
    back_center_idx = 2 * n

    faces = []
    # Side walls: mid → back
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, n + j])
        faces.append([i, n + j, n + i])
    # Back cap: back ring → center
    for i in range(n):
        j = (i + 1) % n
        faces.append([n + i, n + j, back_center_idx])

    mesh = bpy.data.meshes.new("MouthCavityMesh")
    mesh.from_pydata(cavity_verts, [], faces)
    mesh.validate()
    obj = bpy.data.objects.new("MouthCavity", mesh)
    bpy.context.collection.objects.link(obj)

    mat = make_material("MouthInterior", base_color=(0.12, 0.04, 0.03, 1.0),
                        roughness=0.95, metallic=0.0)
    obj.data.materials.append(mat)
    return obj


def add_teeth(verts_3d):
    """Thin teeth strip behind upper lip."""
    # Upper inner lip positions
    upper_inner = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
    lip_pts = np.array([verts_3d[i] for i in upper_inner if i < len(verts_3d)])
    center = lip_pts.mean(axis=0)

    # Find leftmost and rightmost
    left_pt = lip_pts[lip_pts[:, 0].argmin()]
    right_pt = lip_pts[lip_pts[:, 0].argmax()]

    y_top = lip_pts[:, 1].max()
    y_bot = y_top - 0.35
    z_back = center[2] - 0.35

    # Simple teeth strip: 6 teeth
    n_teeth = 8
    x_coords = np.linspace(left_pt[0] * 0.85, right_pt[0] * 0.85, n_teeth + 1)

    verts = []
    faces = []
    for i in range(n_teeth):
        x0, x1 = x_coords[i], x_coords[i+1]
        # 4 corners per tooth
        base = len(verts)
        verts += [
            (x0, y_top, z_back), (x1, y_top, z_back),
            (x1, y_bot, z_back), (x0, y_bot, z_back),
        ]
        faces.append([base, base+1, base+2, base+3])

    mesh = bpy.data.meshes.new("TeethMesh")
    mesh.from_pydata(verts, [], faces)
    mesh.validate()
    obj = bpy.data.objects.new("Teeth", mesh)
    bpy.context.collection.objects.link(obj)

    mat = make_material("Teeth", base_color=(0.92, 0.90, 0.85, 1.0),
                        roughness=0.25, metallic=0.0)
    obj.data.materials.append(mat)
    return obj


def add_ears(verts_3d, skin_color):
    """Add simple ear geometry at the ear landmarks (234=left, 454=right)."""
    # Ear anchor landmarks
    EAR_L = [234, 127, 93, 132, 58, 172]   # left ear area
    EAR_R = [454, 356, 323, 361, 288, 397]  # right ear area

    face_w = np.linalg.norm(verts_3d[234] - verts_3d[454])
    head_h = np.linalg.norm(verts_3d[10] - verts_3d[152])

    def make_ear(pts, side):
        center = pts.mean(axis=0)
        # Scaled ~2.5x from original: top near brow, bottom near lip
        ear_w = face_w * 0.12
        ear_h = head_h * 0.38
        ear_d = face_w * 0.06
        sign = -1.0 if side == 'L' else 1.0

        # Shift ear center: up toward mid-eye, back behind jaw plane
        ear_cx = center[0] + sign * ear_w * 0.3
        ear_cy = center[1] + head_h * 0.12  # raise toward brow area
        ear_cz = center[2] - head_h * 0.06  # push behind jaw

        n = 16
        verts = []
        faces = []
        # Outer rim ring
        outer = []
        for i in range(n):
            ang = 2 * math.pi * i / n
            x = ear_cx + sign * (ear_w * 0.5 + ear_w * 0.5 * abs(math.cos(ang)))
            y = ear_cy + ear_h * 0.5 * math.sin(ang)
            z = ear_cz + ear_d * 0.15
            outer.append((x, y, z))
        # Inner depression ring (smaller, recessed)
        inner = []
        for i in range(n):
            ang = 2 * math.pi * i / n
            x = ear_cx + sign * (ear_w * 0.25 + ear_w * 0.25 * abs(math.cos(ang)))
            y = ear_cy + ear_h * 0.32 * math.sin(ang)
            z = ear_cz - ear_d * 0.4
            inner.append((x, y, z))
        # Back bowl center
        back_c = (ear_cx + sign * ear_w * 0.2, ear_cy, ear_cz - ear_d * 0.7)

        all_v = outer + inner + [back_c]
        oi, ii, bci = 0, n, 2 * n

        # Outer-to-inner face band
        for i in range(n):
            j = (i + 1) % n
            faces.append([oi + i, oi + j, ii + j, ii + i])
        # Inner ring to back center (bowl)
        for i in range(n):
            j = (i + 1) % n
            faces.append([ii + j, ii + i, bci])
        return all_v, faces

    for side, lms in [('L', EAR_L), ('R', EAR_R)]:
        pts = np.array([verts_3d[i] for i in lms if i < len(verts_3d)])
        if len(pts) == 0:
            continue
        all_v, faces = make_ear(pts, side)
        mesh = bpy.data.meshes.new(f"Ear{side}Mesh")
        mesh.from_pydata(all_v, [], faces)
        mesh.validate()
        obj = bpy.data.objects.new(f"Ear{side}", mesh)
        bpy.context.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.shade_smooth()
        ear_col = tuple(c * 0.85 for c in skin_color)
        mat = make_material(f"Ear{side}", base_color=(*ear_col, 1.0),
                            roughness=0.70, metallic=0.0)
        obj.data.materials.append(mat)


def add_hair_cap(verts_3d, hair_color=(0.026, 0.020, 0.018)):
    """Hair ellipsoid covering top and back of head, positioned outside the cranium."""
    # Derive head center and size from face boundary
    BOUNDARY = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361,
                288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149,
                150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,
                67, 109]
    boundary_pts = np.array([verts_3d[i] for i in BOUNDARY if i < len(verts_3d)])
    center = boundary_pts.mean(axis=0)
    head_w = np.ptp(boundary_pts[:, 0])
    head_h = np.ptp(boundary_pts[:, 1])
    head_d = head_w * 0.82

    # Ellipsoid radii — slightly larger than cranium
    rx = head_w * 0.56   # X (left-right)
    ry = head_h * 0.60   # Y (up-down) — taller to cover forehead and top
    rz = head_d * 0.48   # Z (front-back) — conservative to avoid protruding past face

    # Center is pushed backward so front of ellipsoid aligns with forehead, not nose
    cap_center = (center[0], center[1] + head_h * 0.08, center[2] - head_d * 0.25)

    # Max Z allowed: boundary center Z plus a small margin. Slightly looser than
    # before so the crown keeps a bit of forward volume instead of a flat disc.
    max_z = center[2] + head_d * 0.12

    # Per-longitude hairline: high at front-center (shows forehead), dips low at
    # the temples/sides and back so the hair frames the face instead of reading
    # as a flat swim-cap. theta: 0/pi = sides, pi/2 = front (z+), 3pi/2 = back.
    def local_hairline(theta):
        # Small higher-frequency wobble makes the hem look organic, not a clean curve.
        wobble = 0.012 * math.sin(theta * 7.0) + 0.008 * math.sin(theta * 13.0)
        return center[1] + head_h * (0.10 + 0.14 * math.sin(theta)
                                          - 0.16 * abs(math.cos(theta)) + wobble)

    # Lowest point any hairline reaches — stop building rows below this.
    min_hairline = center[1] - head_h * 0.10

    # Build ellipsoid mesh using UV sphere parametrization
    n_lon = 24   # longitude divisions
    n_lat = 18   # latitude divisions (more rows → smoother silhouette/hem)

    verts_out = []
    faces_out = []

    lat_rows = []
    for lat in range(n_lat + 1):
        phi = math.pi * lat / n_lat   # 0=top, pi=bottom
        y_unit = math.cos(phi)        # 1 at top, -1 at bottom
        y_world = cap_center[1] + ry * y_unit
        if y_world < min_hairline:
            # Below the lowest hairline — stop here, will add rim cap
            break
        row = []
        for lon in range(n_lon):
            theta = 2 * math.pi * lon / n_lon
            # Clamp each vertex up to its local hairline → scalloped hem framing face
            vy = max(y_world, local_hairline(theta))
            x = cap_center[0] + rx * math.sin(phi) * math.cos(theta)
            z_raw = cap_center[2] + rz * math.sin(phi) * math.sin(theta)
            z = min(z_raw, max_z)   # clamp so hair cap doesn't protrude past forehead
            row.append((x, vy, z))
        verts_out.extend(row)
        lat_rows.append(len(lat_rows))   # track which lat rows were added

    n_rows = len(lat_rows)
    if n_rows < 2:
        return None

    # Top cap (pole to first ring)
    # Pole vertex is at the top (lat=0, single point)
    # Actually lat=0 is a full ring at phi=0 (just one point in theory, but we built a ring)
    # Build faces between consecutive rings
    for r in range(n_rows - 1):
        r0_start = r * n_lon
        r1_start = (r + 1) * n_lon
        for i in range(n_lon):
            j = (i + 1) % n_lon
            faces_out.append([r0_start+i, r0_start+j, r1_start+j, r1_start+i])

    # Bottom rim cap — fill the open bottom with a flat disk
    last_row_start = (n_rows - 1) * n_lon
    last_row_pts = verts_out[last_row_start:last_row_start + n_lon]
    rim_y = sum(v[1] for v in last_row_pts) / n_lon
    rim_center_idx = len(verts_out)
    verts_out.append((cap_center[0], rim_y, cap_center[2]))
    for i in range(n_lon):
        j = (i + 1) % n_lon
        # Wind inward (toward camera) so face is visible from front
        faces_out.append([last_row_start+j, last_row_start+i, rim_center_idx])

    mesh = bpy.data.meshes.new("HairCapMesh")
    mesh.from_pydata(verts_out, [], faces_out)
    mesh.validate()
    obj = bpy.data.objects.new("HairCap", mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.shade_smooth()
    mat = make_material("HairCap", base_color=(*hair_color, 1.0),
                        roughness=0.80, metallic=0.0)
    obj.data.materials.append(mat)
    return obj


def add_cranium(verts_3d, skin_color):
    """High-quality ellipsoidal cranium stitched to face boundary."""
    BOUNDARY = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361,
                288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149,
                150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,
                67, 109]

    boundary_pts = np.array([verts_3d[i] for i in BOUNDARY if i < len(verts_3d)])
    center = boundary_pts.mean(axis=0)
    head_w = np.ptp(boundary_pts[:, 0])
    head_h = np.ptp(boundary_pts[:, 1])
    head_d = head_w * 0.82

    n_b = len(boundary_pts)
    n_rings = 10
    all_verts = [tuple(v) for v in boundary_pts]

    for ring in range(1, n_rings + 1):
        t = ring / n_rings
        # Cosine blend for smooth shape
        t_cos = (1 - math.cos(t * math.pi)) / 2
        ring_pts = []
        for i, bp in enumerate(boundary_pts):
            # Interpolate toward center and then push back
            x = bp[0] * (1 - t_cos) + center[0] * t_cos
            y = bp[1] * (1 - t_cos * 0.6) + center[1] * t_cos * 0.6
            z = center[2] - head_d * math.sin(t * math.pi * 0.5)
            ring_pts.append((x, y, z))
        all_verts.extend(ring_pts)

    # Back cap center
    back_c = (center[0], center[1], center[2] - head_d * 0.95)
    all_verts.append(back_c)
    back_idx = len(all_verts) - 1

    faces = []
    # Ring stitching
    for ring in range(n_rings):
        r0 = ring * n_b
        r1 = (ring + 1) * n_b
        for i in range(n_b):
            j = (i + 1) % n_b
            faces.append([r0+i, r0+j, r1+j, r1+i])

    # Last ring to center
    last_ring_start = n_rings * n_b
    for i in range(n_b):
        j = (i + 1) % n_b
        faces.append([last_ring_start+i, last_ring_start+j, back_idx])

    mesh = bpy.data.meshes.new("CraniumMesh")
    mesh.from_pydata(all_verts, [], faces)
    mesh.validate()
    bpy.ops.object.select_all(action='DESELECT')
    obj = bpy.data.objects.new("Cranium", mesh)
    bpy.context.collection.objects.link(obj)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.shade_smooth()

    mat = make_material("Cranium", base_color=(*skin_color, 1.0),
                        roughness=0.65, metallic=0.0)
    obj.data.materials.append(mat)
    return obj


def add_neck(verts_3d, skin_color):
    """Landmark-relative neck: proportioned to face_w and head_h."""
    face_w = np.linalg.norm(verts_3d[234] - verts_3d[454])
    head_h = np.linalg.norm(verts_3d[10] - verts_3d[152])
    chin = verts_3d[152]
    face_z = chin[2]

    r_top = face_w * 0.14
    r_mid = face_w * 0.17
    r_bot = face_w * 0.21
    neck_h = head_h * 0.26
    top_y = chin[1] - head_h * 0.02
    cz = face_z - head_h * 0.08
    cx = (verts_3d[234][0] + verts_3d[454][0]) / 2.0

    n_seg = 24
    n_rings = 8
    verts = []
    ring_idx = []
    for ri in range(n_rings + 1):
        t = ri / n_rings
        y = top_y - t * neck_h
        if t < 0.5:
            r = r_top + (r_mid - r_top) * (t / 0.5)
        else:
            r = r_mid + (r_bot - r_mid) * ((t - 0.5) / 0.5)
        start = len(verts)
        for s in range(n_seg):
            a = 2 * math.pi * s / n_seg
            verts.append((cx + r * math.cos(a), y, cz + r * math.sin(a)))
        ring_idx.append((start, r, y))

    # Rounded bottom dome
    bot_r = ring_idx[-1][1]
    bot_y = ring_idx[-1][2]
    dome_d = bot_r * 0.6
    n_cap = 3
    for ci in range(1, n_cap + 1):
        u = ci / n_cap
        r = bot_r * math.cos(u * math.pi / 2)
        y = bot_y - dome_d * math.sin(u * math.pi / 2)
        start = len(verts)
        for s in range(n_seg):
            a = 2 * math.pi * s / n_seg
            verts.append((cx + r * math.cos(a), y, cz + r * math.sin(a)))
        ring_idx.append((start, r, y))
    pole_idx = len(verts)
    verts.append((cx, bot_y - dome_d, cz))

    faces = []
    for ri in range(len(ring_idx) - 1):
        a0 = ring_idx[ri][0]
        a1 = ring_idx[ri + 1][0]
        for s in range(n_seg):
            sn = (s + 1) % n_seg
            faces.append([a0 + s, a0 + sn, a1 + sn, a1 + s])
    last_start = ring_idx[-1][0]
    for s in range(n_seg):
        sn = (s + 1) % n_seg
        faces.append([last_start + s, last_start + sn, pole_idx])

    # Desaturate skin color slightly for neck
    nc = tuple(c * 0.88 for c in skin_color)
    mesh = bpy.data.meshes.new("NeckMesh")
    mesh.from_pydata(verts, [], faces)
    mesh.validate()
    neck = bpy.data.objects.new("Neck", mesh)
    bpy.context.collection.objects.link(neck)
    bpy.context.view_layer.objects.active = neck
    neck.select_set(True)
    bpy.ops.object.shade_smooth()
    mat = make_material("Neck", base_color=(*nc, 1.0), roughness=0.70, metallic=0.0)
    neck.data.materials.append(mat)
    return neck


def add_shoulders(verts_3d, skin_color):
    """Dark-clothing shoulder/bust silhouette, wider than head, grounding the neck."""
    face_w = np.linalg.norm(verts_3d[234] - verts_3d[454])
    head_h = np.linalg.norm(verts_3d[10] - verts_3d[152])
    chin = verts_3d[152]
    cx = (verts_3d[234][0] + verts_3d[454][0]) / 2.0
    face_z = chin[2]

    neck_h = head_h * 0.26
    neck_bot_y = chin[1] - head_h * 0.02 - neck_h

    sw = face_w * 1.1       # shoulder half-width (wider than head)
    sh = head_h * 0.42      # shoulder height
    sd = face_w * 0.45      # shoulder depth
    top_y = neck_bot_y - head_h * 0.04
    cz = face_z - head_h * 0.04

    # Rounded trapezoid: top ring narrower, bottom ring wider, with rounded caps
    n_seg = 20
    n_rings = 6
    verts = []
    ring_idx = []
    for ri in range(n_rings + 1):
        t = ri / n_rings
        y = top_y - t * sh
        # Top narrower (0.7x), bottom full width — trapezoid silhouette
        w = sw * (0.7 + 0.3 * t)
        d = sd * (0.8 + 0.2 * t)
        start = len(verts)
        for s in range(n_seg):
            a = 2 * math.pi * s / n_seg
            verts.append((cx + w * math.cos(a), y, cz + d * math.sin(a)))
        ring_idx.append(start)

    faces = []
    for ri in range(n_rings):
        a0 = ring_idx[ri]
        a1 = ring_idx[ri + 1]
        for s in range(n_seg):
            sn = (s + 1) % n_seg
            faces.append([a0 + s, a0 + sn, a1 + sn, a1 + s])
    # Top cap
    tc = len(verts)
    verts.append((cx, top_y, cz))
    for s in range(n_seg):
        sn = (s + 1) % n_seg
        faces.append([ring_idx[0] + s, ring_idx[0] + sn, tc])
    # Bottom cap
    bc = len(verts)
    verts.append((cx, top_y - sh, cz))
    last = ring_idx[-1]
    for s in range(n_seg):
        sn = (s + 1) % n_seg
        faces.append([last + sn, last + s, bc])

    mesh = bpy.data.meshes.new("ShouldersMesh")
    mesh.from_pydata(verts, [], faces)
    mesh.validate()
    obj = bpy.data.objects.new("Shoulders", mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.shade_smooth()
    # Dark desaturated clothing, not pure black
    clothing = (0.12, 0.11, 0.13)
    mat = make_material("Shoulders", base_color=(*clothing, 1.0),
                        roughness=0.85, metallic=0.0)
    obj.data.materials.append(mat)
    return obj


def place_eyeballs(verts_3d, mp_pts_raw=None):
    """Position eyeballs using MediaPipe iris center landmarks (468=left, 473=right)."""
    # Eye corner landmarks for width measurement
    L_CORNERS = (33, 133)   # left corner, right corner of left eye
    R_CORNERS = (263, 362)  # left corner, right corner of right eye

    # Iris center landmarks from MediaPipe (beyond the 468-face-landmark set)
    # 468 = left iris center, 473 = right iris center
    # These are in the same pixel space as mp_pts_raw; we need to convert to verts_3d space
    if mp_pts_raw is not None and mp_pts_raw.shape[0] > 473:
        # Reconstruct scale/offset from verts_3d vs face_pts_raw
        face_pts = mp_pts_raw[:468, :2]
        x_min, y_min = face_pts.min(axis=0)
        x_max, y_max = face_pts.max(axis=0)
        scale = 20.0 / (x_max - x_min)
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2

        def px_to_3d(pt2d, z):
            x = (pt2d[0] - cx) * scale
            y = -((pt2d[1] - cy) * scale)
            return np.array([x, y, z])

        patches = {
            "EyeballL": os.path.join(OUTPUT, "eye_patch_L.png"),
            "EyeballR": os.path.join(OUTPUT, "eye_patch_R.png"),
        }
        iris_data = [
            ("EyeballL", 468, L_CORNERS),
            ("EyeballR", 473, R_CORNERS),
        ]
        objs = []
        for name, iris_lm, corners in iris_data:
            lc = verts_3d[corners[0]]
            rc = verts_3d[corners[1]]
            face_z = float((lc[2] + rc[2]) / 2)
            center = px_to_3d(mp_pts_raw[iris_lm, :2], face_z)
            eye_width = np.linalg.norm(rc[:2] - lc[:2])
            iris_r = eye_width * 0.16
            obj = add_eyeball(float(center[0]), float(center[1]),
                              face_z, iris_r, name=name,
                              eye_patch_path=patches.get(name))
            objs.append(obj)
        return objs

    # Fallback: average eye corner landmarks
    L_EYE = [159, 145, 33, 133]
    R_EYE = [386, 374, 263, 362]
    objs = []
    for name, indices, corners in [("EyeballL", L_EYE, L_CORNERS),
                                    ("EyeballR", R_EYE, R_CORNERS)]:
        center = np.mean([verts_3d[i] for i in indices], axis=0)
        face_z = float(center[2])
        eye_width = np.linalg.norm(verts_3d[corners[1]][:2] - verts_3d[corners[0]][:2])
        iris_r = eye_width * 0.16
        obj = add_eyeball(float(center[0]), float(center[1] - 0.05),
                          face_z, iris_r, name=name)
        objs.append(obj)
    return objs


def get_avg_skin_color(tex_path):
    import cv2 as cv
    img = cv.imread(tex_path)
    h, w = img.shape[:2]
    roi = img[h//4:h//2, w//3:2*w//3]
    avg = roi.mean(axis=(0, 1)) / 255.0
    return (float(avg[2]), float(avg[1]), float(avg[0]))  # BGR→RGB


def orient_scene():
    """Rotate all objects 180° around Y so face points toward +Z (camera)."""
    import mathutils
    rot = mathutils.Euler((0, math.radians(180), 0))
    for obj in bpy.data.objects:
        obj.rotation_euler.rotate(rot)
    # Also flip normals by negating X to correct mirroring
    # Actually just rotate 180Y — face was built with Z+ as depth away from camera


def export_glb(out_path):
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.export_scene.gltf(
        filepath=out_path,
        export_format='GLB',
        export_apply=True,
        export_texcoords=True,
        export_normals=True,
        export_materials='EXPORT',
        export_skins=False,
        export_morph=True,
        export_morph_normal=False,
        export_morph_tangent=False,
        export_cameras=False,
        export_lights=False,
        export_image_format='AUTO',
        # Keep our XYZ = glTF XYZ (no Blender->glTF axis swap)
        export_yup=False,
    )
    print(f"\n✓ Exported: {out_path}")
    size = os.path.getsize(out_path) / 1024 / 1024
    print(f"  Size: {size:.1f} MB")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT, exist_ok=True)
    print("=== Unum Avatar Builder v6 (Blender) ===\n")

    print("[1/8] Detecting landmarks...")
    verts_3d, uvs_2d, img_w, img_h = get_landmarks()
    print(f"  {len(verts_3d)} landmarks")

    # Raw 2D landmark pixels (needed for texture baking)
    import mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        output_face_blendshapes=False, num_faces=1
    )
    lm = vision.FaceLandmarker.create_from_options(options)
    raw_img = mp.Image.create_from_file(FRONT_IMG)
    result = lm.detect(raw_img)
    lm.close()
    mp_pts_raw = np.array([[p.x * raw_img.width, p.y * raw_img.height, 0.0]
                            for p in result.face_landmarks[0]])

    print("[2/8] Baking multi-view texture...")
    tex_path, cx0, cy0, cx1, cy1 = bake_texture(mp_pts_raw, img_w, img_h)
    skin_color = get_avg_skin_color(tex_path)
    print(f"  Skin RGB: {tuple(round(c,2) for c in skin_color)}")

    # Map raw pixel landmark coords to UV space of the crop
    crop_w = cx1 - cx0
    crop_h = cy1 - cy0
    face_pts_raw = mp_pts_raw[:468, :2]
    uvs_u = np.clip((face_pts_raw[:, 0] - cx0) / crop_w, 0.0, 1.0)
    uvs_v = np.clip((face_pts_raw[:, 1] - cy0) / crop_h, 0.0, 1.0)
    uvs_2d = np.stack([uvs_u, uvs_v], axis=1)

    print("[3/8] Getting triangles...")
    tris = get_triangles()
    print(f"  {len(tris)} base triangles")

    print("[4/8] Building Blender scene...")
    clear_scene()

    print("  → Face mesh + subdivision + UV...")
    face_obj = add_face_mesh(verts_3d, tris, uvs_2d, tex_path)
    print(f"    {len(face_obj.data.vertices)} verts, {len(face_obj.data.polygons)} polys")

    print("  → Shape keys...")
    add_shape_keys(face_obj, verts_3d)

    print("[5/8] Adding eyeballs...")
    place_eyeballs(verts_3d, mp_pts_raw)

    print("[6/8] Adding mouth cavity + teeth...")
    add_mouth_cavity(verts_3d)
    add_teeth(verts_3d)

    print("[7/8] Adding cranium + hair + ears + neck...")
    add_cranium(verts_3d, skin_color)
    add_hair_cap(verts_3d)
    add_ears(verts_3d, skin_color)
    add_neck(verts_3d, skin_color)
    add_shoulders(verts_3d, skin_color)

    print("[8/8] Exporting GLB...")
    export_glb(OUT_GLB)

    print("\n=== Done! ===")
    print(f"  → {OUT_GLB}")


if __name__ == "__main__":
    main()
