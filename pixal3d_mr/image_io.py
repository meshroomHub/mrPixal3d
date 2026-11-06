"""Image I/O through pyalicevision (OpenImageIO: EXR, RAW, ... with AliceVision's color management).

Images are handed to Pixal3D as 8 bit sRGB PIL images, as its scripts expect.
"""
import numpy as np


def read(path):
    """8 bit sRGB PIL image: RGBA with straight (not premultiplied) colors if it has a mask, else RGB.

    A fully opaque alpha counts as no mask, as in Pixal3D (preprocess_image, load_rgba): RGB then, so that later
    steps (e.g. undistortion borders) do not create a mask.
    AliceVision returns RGBA colors premultiplied by alpha (OpenImageIO associated alpha): they are divided back,
    since Pixal3D composites with alpha itself.
    """
    from PIL import Image
    from pyalicevision import image as avimg

    image = avimg.Image_RGBAColor()
    avimg.readImage(path, image, avimg.ImageReadOptions(avimg.EImageColorSpace_SRGB))
    rgba = np.asarray(image.getNumpyArray(), dtype=np.uint8).copy()
    if np.all(rgba[..., 3] == 255):
        return Image.fromarray(np.ascontiguousarray(rgba[..., :3]), "RGB")
    alpha = rgba[..., 3:4].astype(np.float32)
    partial = (alpha > 0) & (alpha < 255)
    if partial.any():
        straight = np.clip(np.round(rgba[..., :3] * 255.0 / np.maximum(alpha, 1.0)), 0, 255).astype(np.uint8)
        rgba[..., :3] = np.where(partial, straight, rgba[..., :3])
    return Image.fromarray(rgba, "RGBA")


def write(path, image):
    """Write a PIL image as 8 bit sRGB, no color conversion: RGBA (straight alpha, PNG) if it has an alpha, else RGB.

    RGBA goes through PIL: OpenImageIO takes the colors as premultiplied by alpha and divides them on writing.
    """
    if image.mode == "RGBA":
        image.save(path)
        return
    from pyalicevision import image as avimg

    out = avimg.Image_RGBColor()
    out.fromNumpyArray(np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8)))
    options = avimg.ImageWriteOptions()
    options.fromColorSpace(avimg.EImageColorSpace_SRGB)
    options.toColorSpace(avimg.EImageColorSpace_SRGB)
    avimg.writeImage(path, out, options)


def writeMask(path, mask):
    """Write a uint8 HxW mask as an 8 bit grey image, no color conversion."""
    from pyalicevision import image as avimg

    out = avimg.Image_uchar()
    out.fromNumpyArray(np.ascontiguousarray(np.asarray(mask, dtype=np.uint8)[..., None]))
    options = avimg.ImageWriteOptions()
    options.fromColorSpace(avimg.EImageColorSpace_NO_CONVERSION)
    options.toColorSpace(avimg.EImageColorSpace_NO_CONVERSION)
    avimg.writeImage(path, out, options)
