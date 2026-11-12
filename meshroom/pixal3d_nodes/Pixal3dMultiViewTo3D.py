__version__ = "2.1"

import logging

import meshroom
from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL


def _coordinate(axis):
    return desc.FloatParam(name=axis, label=axis.upper(), description=f"{axis.upper()} coordinate.", value=0.0,
                           range=None)


class Pixal3dMultiViewTo3D(desc.Node):
    category = "pixal3d"

    documentation = """
Generates a textured PBR mesh (GLB), in the SfM frame, from all the posed views of an SfMData with
[Pixal3D](https://github.com/TencentARC/Pixal3D)'s multi-view model.

The object is located in the SfM (center and size: from the landmarks inside the object masks, or given), then each
view is turned into a Pixal3D view: a virtual camera at the same position looking at the object center, framing it
like Pixal3D's training views, its image being the undistorted photo warped by the exact rotation homography.
The main view defines Pixal3D's front. Then same processing as Pixal3D's official inference_mv.py.

The object masks come from Masks Folder if given (e.g. from Pixal3dImageMasking): they are warped with the views,
and used for the landmarks bounds. Views without a mask get their background removed by this node.

Pinhole intrinsics only (with or without distortion). GPU memory and time grow with the number of views: about
11.5 GB with 4 views, Low VRAM and resolution 1024 (~4 min on an RTX A4000).
"""

    gpu = desc.Level.INTENSIVE

    inputs = [
        desc.File(
            name="sfmData",
            label="SfMData",
            description="SfMData with the posed views of the object (e.g. the StructureFromMotion output). "
                        "All the posed views are used.",
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
        desc.IntParam(
            name="mainViewId",
            label="Main View Id",
            description="View defining Pixal3D's front. -1: the view looking the most directly at the object.",
            value=-1,
            range=None,
        ),
        desc.ChoiceParam(
            name="boundsSource",
            label="Bounds Source",
            description="landmarks: the SfM landmarks inside the background removal mask of every view that sees "
                        "them (needs landmarks on the object).\n"
                        "manual: the given center and size.",
            values=["landmarks", "manual"],
            value="landmarks",
        ),
        desc.GroupAttribute(
            name="boundsCenter",
            label="Bounds Center",
            description="Center of the object, in SfM coordinates.",
            items=[_coordinate("x"), _coordinate("y"), _coordinate("z")],
            joinChar=",",
            enabled=lambda node: node.boundsSource.value == "manual",
        ),
        desc.FloatParam(
            name="boundsSize",
            label="Bounds Size",
            description="Largest extent of the object, in SfM units.",
            value=1.0,
            range=None,
            enabled=lambda node: node.boundsSource.value == "manual",
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
            name="outputFolder",
            label="Output Folder",
            description="Folder of the outputs: mesh.glb and views/ (the Pixal3D views: transforms.json + images).",
            value="{nodeCacheFolder}",
        ),
        desc.File(
            name="output",
            label="Mesh",
            description="Textured mesh (GLB, PBR materials), in the SfM frame.",
            semantic="3d",
            value="{nodeCacheFolder}/mesh.glb",
        ),
        desc.File(
            name="transforms",
            label="Pixal3D Views",
            description="transforms.json of the Pixal3D views (virtual cameras, object frame), with the object "
                        "frame in the SfM ('sfm': center, size, objectToWorld).",
            semantic="textFile",
            value="{nodeCacheFolder}/views/transforms.json",
        ),
    ]

    def processChunk(self, chunk):
        # logging reaches the chunk log: meshroom_compute routes the root logger there (node.prepareLogger)
        node = chunk.node
        logging.getLogger().setLevel(meshroom.logStringToPython[node.verboseLevel.value])
        from pixal3d_mr import multiview_to_3d

        manual = node.boundsSource.value == "manual"
        multiview_to_3d.run(
            outputDir=node.outputFolder.value,
            sfmData=node.sfmData.value,
            mainViewId=node.mainViewId.value,
            boundsSource=node.boundsSource.value,
            boundsCenter=[node.boundsCenter.x.value, node.boundsCenter.y.value, node.boundsCenter.z.value]
            if manual else None,
            boundsSize=node.boundsSize.value if manual else None,
            seed=node.seed.value,
            resolution=int(node.resolution.value),
            lowVram=node.lowVram.value,
            rembg=node.backgroundRemoval.value,
            decimationTarget=node.decimationTarget.value,
            textureSize=int(node.textureSize.value),
            masksFolder=node.masksFolder.value or None,
            maskExtension=node.maskExtension.value,
        )
