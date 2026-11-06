"""Pixal3D posed views of an SfMData to a GLB in the SfM frame, offline, on the rez packages.

The posed views are turned into an upstream views folder (sfm_views: virtual cameras looking at the object center,
object frame from the main view), then processed as run_inference() of the official scripts/inference_mv.py does
(same calls, order and sampler settings). The mesh is written back in the SfM frame.
"""
import logging
import os
import time

import numpy as np

from . import common

logger = logging.getLogger(__name__)

BOUNDS_SOURCES = ("landmarks", "manual")


def loadInferenceMv():
    inference = common.loadScript("inference_mv")
    inference.load_rgba = loadRgba
    return inference


def loadRgba(path, rembg=None):
    """inference_mv.load_rgba, reading through pyalicevision: (RGBA view, was matted)."""
    from . import image_io
    image = image_io.read(path)
    if image.mode == "RGBA":  # has a mask
        return image, False
    if rembg is None:
        raise ValueError(f"{path} has no alpha channel and no matting model was given")
    return rembg(image).convert("RGBA"), True


def run(outputDir, sfmData, mainViewId=-1, boundsSource="landmarks", boundsCenter=None, boundsSize=None,
        seed=42, resolution=1536, lowVram=False, rembg="birefnet", decimationTarget=1000000, textureSize=4096,
        masksFolder=None, maskExtension="png", log=logger.info):
    """Generate a textured GLB, in the SfM frame, from all the posed views of an SfMData.

    mainViewId: view defining Pixal3D's front, -1: the view looking the most directly at the object.
    boundsSource: "landmarks" (SfM landmarks inside the background removal mask of the views that see them) or
    "manual" (boundsCenter, 3 values, and boundsSize, the largest extent of the object, in SfM units).
    masksFolder: object masks of the views (see masking), used instead of matting them; views without a mask file
    there are matted. Without masksFolder, the alpha the images may have is ignored and every view is matted.
    Returns the paths of the written files: mesh (.glb), transforms (.json, the Pixal3D views folder cameras).
    """
    from . import masking, sfm_camera, sfm_views

    if not sfmData or not os.path.isfile(sfmData):
        raise RuntimeError(f"SfMData not found: '{sfmData}'")
    if boundsSource not in BOUNDS_SOURCES:
        raise ValueError(f"Unknown bounds source '{boundsSource}', expected one of {BOUNDS_SOURCES}.")
    if boundsSource == "manual" and (boundsCenter is None or len(boundsCenter) != 3 or not boundsSize or boundsSize <= 0):
        raise ValueError("Manual bounds need a center (3 values) and a size > 0.")
    common.checkResolution(resolution)
    configFile = common.pipelineConfig(rembg, multiview=True)

    data = sfm_camera.loadSfm(sfmData)
    views = sfm_views.posedViews(data)
    if len(views) < 2:
        raise RuntimeError(f"Pixal3D multi-view needs at least 2 posed views, '{sfmData}' has {len(views)}.")
    viewIds = [view["viewId"] for view in views]
    if mainViewId >= 0 and mainViewId not in viewIds:
        raise RuntimeError(f"Main view {mainViewId} is not a posed view of '{sfmData}'.")
    log(f"{len(views)} posed views: {viewIds}")

    paths = {
        "mesh": os.path.join(outputDir, "mesh.glb"),
        "transforms": os.path.join(outputDir, "views", "transforms.json"),
    }
    os.makedirs(outputDir, exist_ok=True)

    log("Reading the views")
    images = []
    for view in views:
        image, maskPath = masking.loadView(view, masksFolder, maskExtension)
        images.append(image if maskPath else image.convert("RGB"))  # RGBA: masked, RGB: to be matted
    if masksFolder:
        log(f"{sum(image.mode == 'RGBA' for image in images)} of {len(views)} views masked from {masksFolder}")

    inference = loadInferenceMv()
    import torch

    start = time.time()
    with common.gpuErrors(lowVram, resolution):
        log(f"Loading Pixal3D multi-view ({configFile}, low VRAM: {lowVram})")
        pipeline = inference.init_pipeline(common.pixal3dPath(), configFile, low_vram=lowVram)
        matting = inference.make_rembg(pipeline)

        def pickMain(center):
            return next(v for v in views if v["viewId"] == mainViewId) if mainViewId >= 0 \
                else sfm_views.mainView(views, center)

        if boundsSource == "landmarks":
            points = sfm_views.landmarks(data)
            log(f"Object bounds from the landmarks ({len(points)}) inside the object masks")
            masks = [masking.alphaMask(image) if image.mode == "RGBA" else masking.objectMask(matting, image)
                     for image in images]
            points = sfm_views.objectLandmarks(points, views, masks)
            if len(points) < sfm_views.MIN_OBJECT_LANDMARKS:
                raise RuntimeError(f"Only {len(points)} landmarks on the object: use manual bounds.")
            center = np.median(points, axis=0)
            for _ in range(2):  # the object frame depends on the center
                main = pickMain(center)
                rotation = sfm_views.objectRotation(main, center)
                center, size = sfm_views.bounds(points, rotation)
            log(f"{len(points)} object landmarks")
        else:
            center, size = np.asarray(boundsCenter, dtype=float), float(boundsSize)
        main = pickMain(center)
        rotation = sfm_views.objectRotation(main, center)
        log(f"Object center {np.round(center, 4).tolist()}, size {size:.4g}, main view {main['viewId']}")

        sfm_views.writeViewsFolder(os.path.dirname(paths["transforms"]), views, images, center, size, rotation,
                                   main["viewId"])
        viewsDir = os.path.dirname(paths["transforms"])
        log(f"Loading the views from {viewsDir}")
        pixal3dViews = inference.load_views(viewsDir, None, rembg=matting)
        inference.check_main_view(pixal3dViews)

        log(f"Generating ({resolution}_cascade, seed {seed})")
        torch.manual_seed(seed)
        meshes, (_, _, gridSize) = pipeline.run_mv(
            pixal3dViews,
            seed=seed,
            return_latent=True,
            pipeline_type=f"{resolution}_cascade",
            max_num_tokens=common.MAX_NUM_TOKENS,
            **common.samplerParams(),
        )

        log(f"Extracting the GLB in the SfM frame ({decimationTarget} faces, {textureSize} px textures)")
        toSfm = sfm_views.objectToWorld(center, size, rotation) @ sfm_views.GRID_ROTATION
        common.exportGlb(pipeline, meshes[0], gridSize, paths["mesh"], decimationTarget, textureSize, transform=toSfm)

    log(f"Done in {time.time() - start:.1f}s, {common.gpuSummary()}: {paths['mesh']}")
    return paths
