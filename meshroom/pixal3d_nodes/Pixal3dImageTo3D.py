__version__ = "1.1"

import logging
import os

import meshroom
from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL


class Pixal3dImageTo3D(desc.Node):
    category = "pixal3d"

    documentation = """
Generates a textured PBR mesh (GLB) from each view of an SfMData with [Pixal3D](https://github.com/TencentARC/Pixal3D),
the views being processed one after the other (the models are loaded once).

Each view image is read with its intrinsic (lens distortion removed, pinhole intrinsics only), and its mask from
Masks Folder if given (e.g. from Pixal3dImageMasking).
Same processing as Pixal3D's official inference.py (same seed, same result):
 - background removal and centering of the subject,
 - camera FOV from the calibrated focal of the view, or estimated with MoGe-2,
 - sparse structure, shape and texture generation, pixel-aligned with the input image,
 - remeshing, decimation and texture baking.

GPU memory: about 9 GB with Low VRAM and resolution 1024 (works on 16 GB GPUs, ~5 min per view on an RTX A4000).
The default settings (all models on the GPU, resolution 1536) need a 24 GB+ GPU.
"""

    gpu = desc.Level.INTENSIVE

    inputs = [
        desc.File(
            name="sfmData",
            label="SfMData",
            description="SfMData with the views (e.g. the StructureFromMotion or CameraInit output): one mesh per "
                        "view with an intrinsic. Pinhole intrinsics only, with or without distortion. The background "
                        "is removed automatically. Pixal3D assumes the object is near the image center.",
            value="",
        ),
        desc.File(
            name="masksFolder",
            label="Masks Folder",
            description="Object masks of the views (e.g. Pixal3dImageMasking or ImageMasking output), used instead "
                        "of removing the background: <viewId>.<ext> or <image name>.<ext>, grey, white = object, the "
                        "size of the image. Views without a mask there get their background removed by this node.",
            value="",
        ),
        desc.ChoiceParam(
            name="maskExtension",
            label="Mask Extension",
            description="File extension of the masks.",
            values=["png", "exr", "jpg"],
            value="png",
            enabled=lambda node: bool(node.masksFolder.value),
        ),
        desc.BoolParam(
            name="skipFailedViews",
            label="Skip Failed Views",
            description="Log a view that fails (e.g. no object found) and go on with the next views, instead of "
                        "stopping the node. Out of GPU memory errors always stop, as does every view failing.",
            value=True,
        ),
        desc.IntParam(
            name="seed",
            label="Seed",
            description="Random seed.",
            value=42,
            range=(0, 2**31 - 1, 1),
        ),
        desc.ChoiceParam(
            name="resolution",
            label="Resolution",
            description="Resolution of the shape generation cascade.",
            values=[1024, 1536],
            value=1024,
        ),
        desc.BoolParam(
            name="lowVram",
            label="Low VRAM",
            description="Keep the models on the CPU and move each one to the GPU only when used (slower).",
            value=True,
        ),
        desc.ChoiceParam(
            name="cameraSource",
            label="Camera Source",
            description="sfm: the calibrated focal of the view.\n"
                        "moge: estimate the FOV with MoGe-2 (as upstream, e.g. for a view without reliable focal).",
            values=["sfm", "moge"],
            value="sfm",
        ),
        desc.ChoiceParam(
            name="backgroundRemoval",
            label="Background Removal",
            description="birefnet: ZhengPeng7/BiRefNet (MIT).\n"
                        "rmbg: briaai/RMBG-2.0, the upstream default (non-commercial licence, "
                        "must be downloaded with a Hugging Face token).",
            values=["birefnet", "rmbg"],
            value="birefnet",
        ),
        desc.IntParam(
            name="decimationTarget",
            label="Decimation Target",
            description="Target number of faces of the output mesh.",
            value=1000000,
            range=(10000, 5000000, 10000),
        ),
        desc.ChoiceParam(
            name="textureSize",
            label="Texture Size",
            description="Size of the baked PBR textures.",
            values=[1024, 2048, 4096],
            value=4096,
        ),
        desc.ChoiceParam(
            name="verboseLevel",
            label="Verbose Level",
            description="Verbosity level (fatal, error, warning, info, debug, trace).",
            values=VERBOSE_LEVEL,
            value="info",
        ),
    ]

    outputs = [
        desc.File(
            name="output",
            label="Mesh",
            description="Textured mesh of each view (GLB, PBR materials).",
            semantic="3d",
            value="{nodeCacheFolder}/<VIEW_ID>_mesh.glb",
        ),
        desc.File(
            name="preprocessed",
            label="Preprocessed Image",
            description="Each view image after background removal (or masking) and centering, as seen by "
                        "Pixal3D.",
            semantic="image",
            value="{nodeCacheFolder}/<VIEW_ID>_preprocessed.png",
        ),
        desc.File(
            name="camera",
            label="Camera",
            description="Camera of each view used for the pixel alignment: camera_angle_x (radians), distance, "
                        "mesh_scale.",
            semantic="textFile",
            value="{nodeCacheFolder}/<VIEW_ID>_camera.json",
        ),
    ]

    def processChunk(self, chunk):
        # logging reaches the chunk log: meshroom_compute routes the root logger there (node.prepareLogger)
        node = chunk.node
        logging.getLogger().setLevel(meshroom.logStringToPython[node.verboseLevel.value])
        from pixal3d_mr import image_to_3d

        image_to_3d.run(
            outputDir=os.path.dirname(node.output.value),
            seed=node.seed.value,
            resolution=int(node.resolution.value),
            lowVram=node.lowVram.value,
            rembg=node.backgroundRemoval.value,
            sfmData=node.sfmData.value,
            cameraSource=node.cameraSource.value,
            decimationTarget=node.decimationTarget.value,
            textureSize=int(node.textureSize.value),
            skipFailedViews=node.skipFailedViews.value,
            masksFolder=node.masksFolder.value or None,
            maskExtension=node.maskExtension.value,
        )
