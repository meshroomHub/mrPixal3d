__version__ = "1.1"

import logging

import meshroom
from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL


class Pixal3dImageMasking(desc.Node):
    category = "pixal3d"

    documentation = """
Object mask of each view of an SfMData, with the background removal model of the Pixal3D pipelines (BiRefNet or
RMBG-2.0): the masks Pixal3dImageTo3D and Pixal3dMultiViewTo3D would compute themselves, computed once, to check or
edit before the generation (their Masks Folder input).

An image that already has an alpha channel gets its alpha as mask.
Output: <image>.png (the view image file name without extension), 8 bit grey, white = object, the size of the original image (lens distortion not removed).
"""

    gpu = desc.Level.NORMAL

    inputs = [
        desc.File(
            name="sfmData",
            label="SfMData",
            description="SfMData with the views to mask (e.g. the CameraInit or StructureFromMotion output).",
            value="",
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
            label="Masks Folder",
            description="Folder of the masks, <image>.png (the view image file name without extension).",
            value="{nodeCacheFolder}",
        ),
        desc.File(
            name="masks",
            label="Masks",
            description="Mask of each view: 8 bit grey, white = object.",
            semantic="image",
            value="{nodeCacheFolder}/<FILESTEM>.png",
        ),
    ]

    def processChunk(self, chunk):
        # logging reaches the chunk log: meshroom_compute routes the root logger there (node.prepareLogger)
        node = chunk.node
        logging.getLogger().setLevel(meshroom.logStringToPython[node.verboseLevel.value])
        from pixal3d_mr import masking

        masking.run(
            outputDir=node.output.value,
            sfmData=node.sfmData.value,
            rembg=node.backgroundRemoval.value,
        )
