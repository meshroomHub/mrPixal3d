"""Object masks (background removal) of SfMData views.

Pixal3dImageMasking writes them, the Pixal3D nodes read them (masksFolder) instead of matting the views themselves.
Mask files follow the AliceVision convention (FeatureExtraction, PrepareDenseScene): <masksFolder>/<viewId>.<ext>,
else <masksFolder>/<image file stem>.<ext>; grey (color masks are converted to grey), white = object, the size of
the view image (distorted, as the photo). Soft masks are kept: they become the alpha of the view image, as the
matting model's output does.
"""
import logging
import os
import time

import numpy as np

from . import common

logger = logging.getLogger(__name__)

MASK_EXTENSIONS = ("png", "exr", "jpg")
MASK_SIZE = 1024  # max size of the images matted for the landmark visual hull


def findMask(masksFolder, extension, view):
    """Mask file of a view (viewInfo) in masksFolder: <viewId>.<ext>, else <image stem>.<ext>; None if none."""
    if not masksFolder:
        return None
    stem = os.path.splitext(os.path.basename(view["imagePath"]))[0]
    for name in (str(view["viewId"]), stem):
        path = os.path.join(masksFolder, f"{name}.{extension}")
        if os.path.isfile(path):
            return path
    return None


def readMask(path, view):
    """Mask file as uint8 HxW (0: background, 255: object), checked against the view size."""
    from pyalicevision import image as avimg

    mask = avimg.Image_float()
    avimg.readImage(path, mask, avimg.ImageReadOptions(avimg.EImageColorSpace_NO_CONVERSION))
    array = np.asarray(mask.getNumpyArray(), dtype=np.float32)
    if array.ndim == 3:
        array = array[..., 0]
    if array.shape != (view["height"], view["width"]):
        raise RuntimeError(f"View {view['viewId']}: mask {path} is {array.shape[1]}x{array.shape[0]}, the image "
                           f"{view['width']}x{view['height']}.")
    return np.round(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)


def applyMask(image, mask):
    """RGBA PIL image: the RGB of image, mask (uint8 HxW) as alpha."""
    from PIL import Image
    rgba = image.convert("RGB")
    rgba.putalpha(Image.fromarray(mask, "L"))
    return rgba


def loadView(view, masksFolder=None, extension="png"):
    """The view image (sfm_camera.readImage), its mask file as alpha if masksFolder has one, undistorted.

    A mask file replaces the alpha the image may already have. A view without mask file in masksFolder is logged.
    Returns (PIL image, mask path or None).
    """
    from . import sfm_camera

    image = sfm_camera.readImage(view)
    maskPath = findMask(masksFolder, extension, view)
    if maskPath:
        image = applyMask(image, readMask(maskPath, view))
    elif masksFolder:
        logger.warning(f"View {view['viewId']}: no {view['viewId']}.{extension} mask (nor image stem) in "
                       f"{masksFolder}: background removed by the node itself")
    if view["hasDistortion"]:
        image = sfm_camera.undistort(view, image)  # also the alpha, 0 outside of the photo
    return image, maskPath


def objectMask(rembg, image):
    """Foreground mask (bool) of an image, matted at most MASK_SIZE pixels wide."""
    small = image.copy()
    small.thumbnail((MASK_SIZE, MASK_SIZE))
    return np.asarray(rembg(small.convert("RGB")).getchannel("A")) > 127


def alphaMask(image):
    """Foreground mask (bool) of an RGBA image."""
    return np.asarray(image.getchannel("A")) > 127


def run(outputDir, sfmData, rembg="birefnet", viewIds=None, log=logger.info):
    """Write the background removal mask of each view of sfmData (or only viewIds) as <outputDir>/<viewId>.png.

    The mask of an image with an alpha channel (not fully opaque) is its alpha, as Pixal3D uses it as the mask;
    other images are matted with the background removal model of the Pixal3D pipelines (rembg), at full size.
    Returns the written mask paths.
    """
    from . import image_io, sfm_camera

    if not sfmData or not os.path.isfile(sfmData):
        raise RuntimeError(f"SfMData not found: '{sfmData}'")
    data = sfm_camera.loadSfm(sfmData)
    views = data.getViews()
    if viewIds is None:
        viewIds = sorted(views.keys())
    missing = [v for v in viewIds if v not in views]
    if missing:
        raise RuntimeError(f"Views {missing} are not in '{sfmData}'")
    if not viewIds:
        raise RuntimeError(f"No view in '{sfmData}'")
    os.makedirs(outputDir, exist_ok=True)

    start = time.time()
    log(f"Loading the background removal model ({rembg})")
    model = common.loadRembgModel(rembg)
    model.cuda()
    paths = []
    for index, viewId in enumerate(viewIds):
        imagePath = views[viewId].getImage().getImagePath()
        image = image_io.read(imagePath)
        if image.mode == "RGBA":
            log(f"[{index + 1}/{len(viewIds)}] View {viewId}: {imagePath}, mask from its alpha")
            mask = np.asarray(image.getchannel("A"))
        else:
            log(f"[{index + 1}/{len(viewIds)}] View {viewId}: {imagePath}, matting")
            mask = np.asarray(model(image).getchannel("A"))
        if not (mask > 0.8 * 255).any():
            logger.warning(f"View {viewId}: empty mask (no object found)")
        path = os.path.join(outputDir, f"{viewId}.png")
        image_io.writeMask(path, mask)
        paths.append(path)
    log(f"Done in {time.time() - start:.1f}s: {len(paths)} masks in {outputDir}")
    return paths
