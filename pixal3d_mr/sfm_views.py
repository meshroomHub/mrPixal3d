"""Pixal3D multi-view input (transforms.json views folder) from the posed views of an AliceVision SfMData.

Pixal3D expects the object centered at the origin of a Z-up frame, fitting [-0.5, 0.5]^3, square views with a
centered principal point, and the main view (frame 0) at the canonical front pose (camera at (0, -d, 0) looking
at the origin). For each posed view, a virtual camera with the same center is rotated to look at the object center,
with a FOV framing the object like the upstream example rig; its image is the undistorted photo warped by the exact
rotation homography. The object frame is defined by the main virtual camera, and maps back to the SfM frame.

Cameras: AliceVision camToWorld (x right, y down, z forward); Blender camToWorld (x right, y up, z backward).
"""
import json
import math
import os

import numpy as np

# Upstream example rig (assets/mv_images/example): 20 deg FOV at 3.1192 units, the object (unit cube) filling ~90%
# of the frame: the half width of the view at the object center is 0.55 units.
FRAME_HALF_WIDTH = 3.1192049980163574 * math.tan(math.radians(10.0))
RESOLUTION = 1024  # virtual views, Pixal3D resizes them to 512 and 1024
CV_TO_BLENDER = np.diag([1.0, -1.0, -1.0])
FRONT_ROTATION = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])  # canonical front, Blender c2w
# Pixal3D mesh / voxel grid coordinates to its Z-up object frame (ProjGrid: grid_points @ rotation_matrix.T)
GRID_ROTATION = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
MIN_OBJECT_LANDMARKS = 20


def posedViews(data):
    """viewInfo of every view with a pose and a pinhole intrinsic, sorted by view id."""
    from . import sfm_camera
    views = []
    for viewId in sorted(data.getViews().keys()):
        view = data.getViews()[viewId]
        if data.isPoseAndIntrinsicDefined(view):
            views.append(sfm_camera.viewInfo(data, viewId))
    return views


def landmarks(data):
    return np.array([np.asarray(lm.X, dtype=float).ravel() for lm in data.getLandmarks().values()]).reshape(-1, 3)


def project(view, points):
    """Pixel coordinates (undistorted) and depth of world points in a view."""
    c2w = view["camToWorld"]
    cam = (points - c2w[:3, 3]) @ c2w[:3, :3]  # world to camera: R^T (X - C)
    z = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.stack([view["fx"] * cam[:, 0] / z + view["cx"], view["fy"] * cam[:, 1] / z + view["cy"]], axis=1)
    return uv, z


def objectLandmarks(points, views, masks, minRatio=0.9):
    """Visual hull test: landmarks inside the object mask of (almost) every view that sees them in frame.

    masks: boolean foreground arrays per view, at any resolution. Returns the kept points.
    """
    seen = np.zeros(len(points), dtype=int)
    inside = np.zeros(len(points), dtype=int)
    for view, mask in zip(views, masks):
        uv, z = project(view, points)
        inFrame = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < view["width"]) & (uv[:, 1] >= 0) & (uv[:, 1] < view["height"])
        mh, mw = mask.shape
        u = np.clip((uv[inFrame, 0] * mw / view["width"]).astype(int), 0, mw - 1)
        v = np.clip((uv[inFrame, 1] * mh / view["height"]).astype(int), 0, mh - 1)
        seen += inFrame
        inside[inFrame] += mask[v, u]
    keep = (seen >= 2) & (inside >= np.ceil(minRatio * seen))
    return points[keep]


def lookAt(center, target, upHint):
    """AliceVision camToWorld rotation (x right, y down, z forward) looking at target, keeping upHint up."""
    z = target - center
    z = z / np.linalg.norm(z)
    y = -(upHint - np.dot(upHint, z) * z)
    if np.linalg.norm(y) < 1e-6:
        raise RuntimeError("A camera looks along its up axis toward the object center.")
    y = y / np.linalg.norm(y)
    return np.stack([np.cross(y, z), y, z], axis=1)


def virtualRotation(view, center):
    """The view's camera rotated to look at the object center, keeping its roll."""
    c2w = view["camToWorld"]
    return lookAt(c2w[:3, 3], center, -c2w[:3, 1])


def objectRotation(mainView, center):
    """Q (world to object rotation) such that the main virtual camera is the canonical front view."""
    return FRONT_ROTATION @ (virtualRotation(mainView, center) @ CV_TO_BLENDER).T


def bounds(points, rotation):
    """Center (world) and size (largest extent) of points, as a box aligned with the object frame (1-99%)."""
    local = points @ rotation.T
    low, high = np.percentile(local, 1, axis=0), np.percentile(local, 99, axis=0)
    return rotation.T @ ((low + high) / 2), float(np.max(high - low))


def mainView(views, center):
    """The view whose optical axis passes the closest (in angle) to the object center."""
    def angle(view):
        c2w = view["camToWorld"]
        direction = center - c2w[:3, 3]
        return math.acos(np.clip(np.dot(direction / np.linalg.norm(direction), c2w[:3, 2]), -1.0, 1.0))
    return min(views, key=angle)


def objectToWorld(center, size, rotation):
    """4x4 from the Pixal3D object frame (unit cube) to the SfM frame."""
    m = np.eye(4)
    m[:3, :3] = size * rotation.T
    m[:3, 3] = center
    return m


def virtualCamera(view, center, size, rotation):
    """(Blender camToWorld in the object frame, horizontal FOV, homography virtual pixel -> source pixel)."""
    c2w = view["camToWorld"]
    distance = np.linalg.norm(c2w[:3, 3] - center) / size
    if distance <= 1.05 * FRAME_HALF_WIDTH:
        raise RuntimeError(f"View {view['viewId']} is too close to the object (distance {distance:.2f} object sizes).")
    fov = 2.0 * math.atan(FRAME_HALF_WIDTH / distance)
    virtual = virtualRotation(view, center)

    focal = (RESOLUTION / 2.0) / math.tan(fov / 2.0)
    kVirtual = np.array([[focal, 0, RESOLUTION / 2.0], [0, focal, RESOLUTION / 2.0], [0, 0, 1]])
    kSource = np.array([[view["fx"], 0, view["cx"]], [0, view["fy"], view["cy"]], [0, 0, 1]])
    virtualToSource = kSource @ c2w[:3, :3].T @ virtual @ np.linalg.inv(kVirtual)

    blender = np.eye(4)
    blender[:3, :3] = rotation @ virtual @ CV_TO_BLENDER
    blender[:3, 3] = rotation @ (c2w[:3, 3] - center) / size
    return blender, fov, virtualToSource


def virtualView(view, center, size, rotation, image):
    """(warped RGB image, Blender camToWorld in the object frame, horizontal FOV) of the view looking at the center.

    image: the undistorted view image (PIL).
    """
    import cv2
    from PIL import Image

    blender, fov, virtualToSource = virtualCamera(view, center, size, rotation)
    warped = cv2.warpPerspective(np.asarray(image), virtualToSource, (RESOLUTION, RESOLUTION),
                                 flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return Image.fromarray(warped), blender, fov


def writeViewsFolder(folder, views, images, center, size, rotation, mainViewId):
    """transforms.json + view images (main view first) in the upstream format. Returns the transforms path."""
    from . import image_io
    os.makedirs(folder, exist_ok=True)
    ordered = sorted(zip(views, images), key=lambda vi: vi[0]["viewId"] != mainViewId)
    frames = []
    for view, image in ordered:
        warped, blender, fov = virtualView(view, center, size, rotation, image)
        name = f"{view['viewId']}.png"
        image_io.write(os.path.join(folder, name), warped)
        frames.append({"file_path": name, "name": str(view["viewId"]), "camera_angle_x": fov,
                       "transform_matrix": blender.tolist()})
    path = os.path.join(folder, "transforms.json")
    with open(path, "w") as f:
        json.dump({
            "mesh_scale": 1.0,
            "frames": frames,
            # not read by Pixal3D: the object frame in the SfM
            "sfm": {"mainViewId": int(mainViewId), "center": list(map(float, center)), "size": size,
                    "objectToWorld": objectToWorld(center, size, rotation).tolist()},
        }, f, indent=4)
    return path
