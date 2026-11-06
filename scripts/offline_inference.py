"""Pixal3D single image to GLB, offline on the rez packages: same code as the Pixal3dImageTo3D node.

    rez env pixal3d-1.0.0 pixal3d_nnModels-1.0.0 naf-1.0.0 aliceVision-develop -- \
        python offline_inference.py --sfm_data sfm.abc --output_dir out [--low_vram --resolution 1024]
    rez env ... -- python offline_inference.py --image input.png --output_dir out   (MoGe-2 camera, as upstream)

Writes out/<viewId>_mesh.glb, _preprocessed.png and _camera.json for each view (--view_id: only these views),
or out/mesh.glb, out/preprocessed.png and out/camera.json with --image.
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pixal3d_mr import common, image_to_3d  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
source = parser.add_mutually_exclusive_group(required=True)
source.add_argument("--sfm_data", help="SfMData with the view (the node's input)")
source.add_argument("--image", help="image file instead of an SfMData view (no sfm camera)")
parser.add_argument("--view_id", type=int, nargs="+", default=None, help="views to process (default: all)")
parser.add_argument("--camera_source", choices=image_to_3d.CAMERA_SOURCES, default=None,
                    help="default: sfm with --sfm_data, else manual with --fov, else moge")
parser.add_argument("--output_dir", default="./output")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--resolution", type=int, choices=[1024, 1536], default=1536)
parser.add_argument("--low_vram", action="store_true")
parser.add_argument("--rembg", choices=common.REMBG_MODELS, default="birefnet")
parser.add_argument("--fov", type=float, default=None, help="horizontal FOV in degrees (manual camera)")
parser.add_argument("--decimation_target", type=int, default=1000000)
parser.add_argument("--texture_size", type=int, default=4096)
parser.add_argument("--stop_on_failure", action="store_true", help="stop at the first failed view instead of skipping it")
args = parser.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

image_to_3d.run(
    imagePath=args.image,
    sfmData=args.sfm_data,
    viewIds=args.view_id,
    cameraSource=args.camera_source,
    outputDir=args.output_dir,
    seed=args.seed,
    resolution=args.resolution,
    lowVram=args.low_vram,
    rembg=args.rembg,
    fovDegrees=args.fov,
    decimationTarget=args.decimation_target,
    textureSize=args.texture_size,
    skipFailedViews=not args.stop_on_failure,
)
