"""Pixal3D camera from an AliceVision SfMData view (instead of the MoGe-2 estimate).

Pixal3D sees a square crop around the object (pipeline.preprocess_image) and needs the horizontal FOV of that
crop. With a calibrated view: FOV = 2 atan(cropWidth / (2 fx)), crop width and fx in original image pixels.
Lens distortion is removed first (pinhole intrinsics only), keeping fx and the principal point.

The crop is generally not centered on the principal point: Pixal3D assumes it is (as with MoGe). The offset is
logged and saved, with the view pose, for aligning the mesh with the SfM later.
"""
import math

import numpy as np

UNDISTORTION_GRID_STRIDE = 16  # pixels between exact undistortion samples, bilinear in between


def loadSfm(sfmPath):
    from pyalicevision import sfmData, sfmDataIO

    data = sfmData.SfMData()
    if not sfmDataIO.load(data, sfmPath, sfmDataIO.ALL):
        raise RuntimeError(f"Cannot load the SfMData '{sfmPath}'")
    return data


def loadView(sfmPath, viewId):
    """Image, intrinsic and pose of one SfMData view."""
    return viewInfo(loadSfm(sfmPath), viewId, sfmPath)


def viewInfo(data, viewId, sfmPath=""):
    """Image, intrinsic and pose (camToWorld, None if not posed) of a view of a loaded SfMData."""
    from pyalicevision import camera as cam

    views = data.getViews()
    if viewId not in views:
        ids = sorted(views.keys())
        raise RuntimeError(f"View {viewId} is not in '{sfmPath}' ({len(ids)} views, e.g. {ids[:10]})")
    view = views[viewId]
    intrinsics = data.getIntrinsics()
    if view.getIntrinsicId() not in intrinsics:
        raise RuntimeError(f"View {viewId} has no intrinsic in '{sfmPath}'")
    intrinsic = intrinsics[view.getIntrinsicId()]
    if not cam.isPinhole(intrinsic.getType()):
        raise RuntimeError(f"View {viewId}: only pinhole intrinsics are supported (with or without distortion).")
    intrinsic = cam.IntrinsicScaleOffsetDisto.cast(intrinsic)

    camToWorld = None
    if data.isPoseAndIntrinsicDefined(view):
        pose = data.getPose(view)  # keep it alive: rotation() returns a view into its memory
        transform = pose.getTransform()
        camToWorld = np.eye(4)
        camToWorld[:3, :3] = np.array(transform.rotation(), dtype=float).T
        camToWorld[:3, 3] = np.array(transform.center(), dtype=float).ravel()

    fx, fy = np.asarray(intrinsic.getScale(), dtype=float).ravel()
    cx, cy = np.asarray(intrinsic.getPrincipalPoint(), dtype=float).ravel()
    return {
        "viewId": int(viewId),
        "imagePath": view.getImage().getImagePath(),
        "width": int(intrinsic.w()),
        "height": int(intrinsic.h()),
        "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
        "hasDistortion": bool(intrinsic.hasDistortion()),
        "camToWorld": camToWorld,
        "intrinsic": intrinsic,
        "sfmData": data,  # owns the intrinsic
    }


def readImage(view):
    """The view image as an 8 bit sRGB PIL image (RGBA if masked, see image_io.read), checked against its intrinsic."""
    from . import image_io

    image = image_io.read(view["imagePath"])
    if image.size != (view["width"], view["height"]):
        raise RuntimeError(f"View {view['viewId']}: image size {image.size} does not match its intrinsic "
                           f"({view['width']}x{view['height']}).")
    return image


def undistortionMap(view, stride=UNDISTORTION_GRID_STRIDE):
    """For each pixel of the undistorted image (same fx, fy, principal point), its position in the distorted one.

    Exact on a grid every `stride` pixels, bilinear in between. Returns float32 (mapX, mapY) for cv2.remap.
    """
    import torch
    import torch.nn.functional as F

    width, height, intrinsic = view["width"], view["height"], view["intrinsic"]
    xs = np.linspace(0.0, width - 1.0, math.ceil((width - 1) / stride) + 1)
    ys = np.linspace(0.0, height - 1.0, math.ceil((height - 1) / stride) + 1)
    grid = np.empty((2, len(ys), len(xs)))
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            grid[:, j, i] = np.asarray(intrinsic.getDistortedPixel(np.array([x, y])), dtype=float).ravel()
    full = F.interpolate(torch.from_numpy(grid)[None], size=(height, width), mode="bilinear", align_corners=True)[0]
    return full[0].numpy().astype(np.float32), full[1].numpy().astype(np.float32)


def undistort(view, image):
    import cv2
    from PIL import Image
    mapX, mapY = undistortionMap(view)
    return Image.fromarray(cv2.remap(np.asarray(image), mapX, mapY, cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0))


def cropCamera(view, crop):
    """Horizontal FOV of a crop (box in original pixels) and the angle between its center and the optical axis."""
    x0, y0, x1, y1 = crop
    fov = 2.0 * math.atan((x1 - x0) / (2.0 * view["fx"]))
    dx = ((x0 + x1) / 2.0 - view["cx"]) / view["fx"]
    dy = ((y0 + y1) / 2.0 - view["cy"]) / view["fy"]
    offAxis = math.degrees(math.atan(math.hypot(dx, dy)))
    return fov, offAxis
