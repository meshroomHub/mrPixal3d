"""Pixal3D single image to GLB, offline, on the rez packages.

Mirrors run_inference() of the official scripts/inference.py (same calls, same order, same sampler
settings) so results match it for a given seed, with:
 - every Hugging Face / torch.hub id redirected to pixal3d_nnModels and naf,
 - the decimation target and texture size exposed,
 - the preprocessed image and the camera kept as outputs,
 - NAF's neighborhood attention computed without upsampling keys/values (naf_lowmem, same result).
"""
import json
import logging
import math
import os
import time

from . import common

logger = logging.getLogger(__name__)

# run_inference() defaults
MESH_SCALE = 1.0
EXTEND_PIXEL = 0
IMAGE_RESOLUTION = 512


def loadInference():
    return common.loadScript("inference")


def fovCamera(inference, fovRadians):
    """Camera parameters from a known horizontal FOV, as inference.py does for --fov."""
    import torch
    distance = inference.distance_from_fov(
        fovRadians, torch.tensor([-1.0, 0.0, 0.0]),
        torch.tensor([0 - EXTEND_PIXEL, IMAGE_RESOLUTION - 1 + EXTEND_PIXEL]),
        MESH_SCALE, IMAGE_RESOLUTION,
    )["distance_from_x"]
    return {"camera_angle_x": float(fovRadians), "distance": float(distance), "mesh_scale": MESH_SCALE}


def preprocess(pipeline, image, bgColor=(0, 0, 0)):
    """Pixal3DImageTo3DPipeline.preprocess_image, line for line, also returning where the crop is.

    Returns (preprocessed image, crop box (x0, y0, x1, y1) in pixels of the input image).
    """
    import numpy as np
    from PIL import Image

    width, height = image.size
    has_alpha = False
    if image.mode == 'RGBA':
        alpha = np.array(image)[:, :, 3]
        if not np.all(alpha == 255):
            has_alpha = True
    max_size = max(image.size)
    scale = min(1, 1024 / max_size)
    if scale < 1:
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
    scaleX, scaleY = image.width / width, image.height / height
    if has_alpha:
        output = image
    else:
        image = image.convert('RGB')
        if pipeline.low_vram:
            pipeline.rembg_model.to(pipeline.device)
        output = pipeline.rembg_model(image)
        if pipeline.low_vram:
            pipeline.rembg_model.cpu()
    output_np = np.array(output)
    alpha = output_np[:, :, 3]
    bbox = np.argwhere(alpha > 0.8 * 255)
    if len(bbox) == 0:
        raise RuntimeError("No object found in the image (empty background removal mask).")
    bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(size * 1.1)
    bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
    output = output.crop(bbox)  # type: ignore
    output = np.array(output).astype(np.float32) / 255
    rgb = output[:, :, :3]
    a = output[:, :, 3:4]
    bg = np.array(bgColor, dtype=np.float32) / 255.0
    output = rgb * a + bg * (1.0 - a)
    output = Image.fromarray((np.clip(output, 0, 1) * 255).astype(np.uint8))

    # PIL rounds the crop box to integer pixels
    x0, y0, x1, y1 = (int(round(v)) for v in bbox)
    return output, (x0 / scaleX, y0 / scaleY, x1 / scaleX, y1 / scaleY)


CAMERA_SOURCES = ("sfm", "moge", "manual")


def run(imagePath=None, outputDir=".", seed=42, resolution=1536, lowVram=False, rembg="birefnet",
        fovDegrees=None, sfmData=None, viewIds=None, cameraSource=None, decimationTarget=1000000, textureSize=4096,
        skipFailedViews=True, masksFolder=None, maskExtension="png", log=logger.info):
    """Generate a textured GLB from each view of sfmData (undistorted), or from imagePath.

    sfmData: every view with an intrinsic is processed in turn, by view id (or only viewIds), the pipeline loaded once.
    Its outputs are named <viewId>_mesh.glb, <viewId>_preprocessed.png, <viewId>_camera.json; with imagePath:
    mesh.glb, preprocessed.png, camera.json.
    cameraSource: "sfm" (calibrated focal of the view), "moge" (MoGe-2 estimate) or "manual" (fovDegrees, horizontal
    FOV of the preprocessed image). Default: sfm with an SfMData, else manual if fovDegrees is given, else moge.
    skipFailedViews: log a view that fails (e.g. no object found) and go on with the next one, instead of stopping.
    Out of GPU memory errors always stop (the next views would fail the same way), as does every view failing.
    masksFolder: object masks of the SfMData views (see masking), used instead of matting them; views without a
    mask file there are matted (or use the alpha their image may have).
    Returns the written files of each processed image: a list of {mesh, preprocessed, camera} paths.
    """
    if sfmData:
        if not os.path.isfile(sfmData):
            raise RuntimeError(f"SfMData not found: '{sfmData}'")
    elif not imagePath or not os.path.isfile(imagePath):
        raise RuntimeError(f"Input image not found: '{imagePath}'")
    cameraSource = cameraSource or ("sfm" if sfmData else "manual" if fovDegrees else "moge")
    if cameraSource not in CAMERA_SOURCES:
        raise ValueError(f"Unknown camera source '{cameraSource}', expected one of {CAMERA_SOURCES}.")
    if cameraSource == "sfm" and not sfmData:
        raise ValueError("The sfm camera source needs an SfMData.")
    if cameraSource == "manual" and not fovDegrees:
        raise ValueError("The manual camera source needs a FOV.")
    common.checkResolution(resolution)
    configFile = common.pipelineConfig(rembg)
    common.setSingleViewConfig(configFile)

    if sfmData:
        from . import sfm_camera
        data = sfm_camera.loadSfm(sfmData)
        views = data.getViews()
        intrinsics = data.getIntrinsics()
        if viewIds is None:
            viewIds = sorted(views.keys())
        for viewId in [v for v in viewIds if v in views and views[v].getIntrinsicId() not in intrinsics]:
            logger.warning(f"View {viewId} has no intrinsic: skipped")
        # viewInfo checks every view first (exists, pinhole), before the long generation
        inputs = [sfm_camera.viewInfo(data, v, sfmData) for v in viewIds
                  if v not in views or views[v].getIntrinsicId() in intrinsics]
        if not inputs:
            raise RuntimeError(f"No view with an intrinsic in '{sfmData}'")
        log(f"{len(inputs)} views to process in '{sfmData}'")
    else:
        inputs = [None]
    inference = loadInference()

    os.makedirs(outputDir, exist_ok=True)
    start = time.time()
    results = []
    failed = []
    import torch
    with common.gpuErrors(lowVram, resolution):
        log(f"Loading Pixal3D ({configFile}, low VRAM: {lowVram})")
        pipeline = inference.init_pipeline(common.pixal3dPath(), low_vram=lowVram)
        for index, view in enumerate(inputs):
            if view is not None:
                log(f"[{index + 1}/{len(inputs)}] View {view['viewId']}")
            try:
                results.append(processImage(
                    inference, pipeline, outputDir, view, imagePath, sfmData, cameraSource, fovDegrees, seed,
                    resolution, decimationTarget, textureSize, masksFolder, maskExtension, log))
            except torch.OutOfMemoryError:
                raise
            except Exception:
                if view is None or not skipFailedViews:
                    raise
                logger.exception(f"View {view['viewId']} failed: skipped")
                failed.append(view["viewId"])
                torch.cuda.empty_cache()

    if failed:
        logger.warning(f"{len(failed)} of {len(inputs)} views failed: {failed}")
        if not results:
            raise RuntimeError(f"Every view failed: {failed}")
    log(f"Done in {time.time() - start:.1f}s, {common.gpuSummary()}: {len(results)} meshes in {outputDir}")
    return results


def processImage(inference, pipeline, outputDir, view, imagePath, sfmData, cameraSource, fovDegrees, seed,
                 resolution, decimationTarget, textureSize, masksFolder, maskExtension, log):
    """One image (an SfM view, or imagePath if view is None) to its mesh, preprocessed image and camera."""
    import torch

    from . import image_io

    prefix = "" if view is None else f"{view['viewId']}_"
    paths = {
        "mesh": os.path.join(outputDir, f"{prefix}mesh.glb"),
        "preprocessed": os.path.join(outputDir, f"{prefix}preprocessed.png"),
        "camera": os.path.join(outputDir, f"{prefix}camera.json"),
    }
    start = time.time()
    if view is None:
        log(f"Preprocessing {imagePath}")
        inputImage = image_io.read(imagePath)
    else:
        from . import masking, sfm_camera
        log(f"SfM view {view['viewId']}: {view['imagePath']} ({view['width']}x{view['height']}, fx {view['fx']:.1f} "
            f"px, distortion: {view['hasDistortion']}, posed: {view['camToWorld'] is not None})")
        inputImage, maskPath = masking.loadView(view, masksFolder, maskExtension)
        if maskPath:
            log(f"Mask: {maskPath}")
    image, crop = preprocess(pipeline, inputImage)
    image_io.write(paths["preprocessed"], image)

    cameraInfo = {"source": cameraSource, "cropBox": list(crop)}
    if view is not None:
        sfmFov, offAxis = sfm_camera.cropCamera(view, crop)
        cameraInfo.update({
            "sfmData": sfmData, "viewId": view["viewId"], "imagePath": view["imagePath"],
            "fx": view["fx"], "fy": view["fy"], "principalPoint": [view["cx"], view["cy"]],
            "undistorted": view["hasDistortion"], "offAxisDegrees": offAxis,
            "camToWorld": None if view["camToWorld"] is None else view["camToWorld"].tolist(),
        })
    if cameraSource == "sfm":
        camera = fovCamera(inference, sfmFov)
        log(f"Camera from the SfM focal: crop {crop[2] - crop[0]:.0f} px wide, its center "
            f"{offAxis:.1f} deg off the optical axis")
        if offAxis > 10:
            log(f"Warning: the object is {offAxis:.1f} deg off the optical axis; Pixal3D assumes it is centered "
                f"(the mesh may be skewed). Prefer a view where the object is near the image center.")
    elif cameraSource == "manual":
        camera = fovCamera(inference, math.radians(fovDegrees))
        log(f"Camera from the given FOV: {fovDegrees:.2f} deg")
    else:
        log("Estimating the camera with MoGe-2")
        moge = inference.load_moge_model(
            device="cuda", model_name=os.path.join(common.modelsPath(), "moge-2-vitl", "model.pt"))
        camera = inference.get_camera_params_wild_moge(
            paths["preprocessed"], moge, device="cuda",
            mesh_scale=MESH_SCALE, extend_pixel=EXTEND_PIXEL, image_resolution=IMAGE_RESOLUTION)
        moge.cpu()
        del moge
        torch.cuda.empty_cache()
    camera = {k: float(v) for k, v in camera.items()}
    log(f"Camera: FOV {math.degrees(camera['camera_angle_x']):.2f} deg, distance {camera['distance']:.4f}")
    with open(paths["camera"], "w") as f:
        json.dump({**camera, **cameraInfo}, f, indent=4)

    log(f"Generating ({resolution}_cascade, seed {seed})")
    torch.manual_seed(seed)
    meshes, (_, _, gridSize) = pipeline.run(
        image,
        camera_params=camera,
        seed=seed,
        preprocess_image=False,
        return_latent=True,
        pipeline_type=f"{resolution}_cascade",
        max_num_tokens=common.MAX_NUM_TOKENS,
        **common.samplerParams(),
    )

    log(f"Extracting the GLB ({decimationTarget} faces, {textureSize} px textures)")
    common.exportGlb(pipeline, meshes[0], gridSize, paths["mesh"], decimationTarget, textureSize)
    del meshes
    torch.cuda.empty_cache()
    log(f"{paths['mesh']} in {time.time() - start:.1f}s")
    return paths
