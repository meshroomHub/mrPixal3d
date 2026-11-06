"""Pixal3D posed views of an SfMData to a GLB in the SfM frame: same code as the Pixal3dMultiViewTo3D node.

    rez env pixal3d-1.0.0 pixal3d_nnModels-1.0.0 naf-1.0.0 aliceVision-develop -- \
        python offline_inference_mv.py --sfm_data sfm.abc --output_dir out [--bounds_center x y z --bounds_size s]

Writes out/mesh.glb and out/views/ (the Pixal3D views: transforms.json + images).
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pixal3d_mr import common, multiview_to_3d  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--sfm_data", required=True, help="SfMData with the posed views (all are used)")
parser.add_argument("--main_view_id", type=int, default=-1, help="view defining Pixal3D's front, -1: automatic")
parser.add_argument("--bounds_center", type=float, nargs=3, default=None,
                    help="object center in SfM coordinates (with --bounds_size), default: from the landmarks")
parser.add_argument("--bounds_size", type=float, default=None, help="largest object extent, in SfM units")
parser.add_argument("--output_dir", default="./output_mv")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--resolution", type=int, choices=common.RESOLUTIONS, default=1536)
parser.add_argument("--low_vram", action="store_true")
parser.add_argument("--rembg", choices=common.REMBG_MODELS, default="birefnet")
parser.add_argument("--masks_folder", default=None, help="object masks of the views (<viewId>.<ext>), e.g. from "
                    "offline_masking.py, instead of removing the background")
parser.add_argument("--mask_extension", choices=["png", "exr", "jpg"], default="png")
parser.add_argument("--decimation_target", type=int, default=1000000)
parser.add_argument("--texture_size", type=int, default=4096)
args = parser.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

multiview_to_3d.run(
    outputDir=args.output_dir,
    sfmData=args.sfm_data,
    mainViewId=args.main_view_id,
    boundsSource="manual" if args.bounds_center is not None else "landmarks",
    boundsCenter=args.bounds_center,
    boundsSize=args.bounds_size,
    seed=args.seed,
    resolution=args.resolution,
    lowVram=args.low_vram,
    rembg=args.rembg,
    decimationTarget=args.decimation_target,
    textureSize=args.texture_size,
    masksFolder=args.masks_folder,
    maskExtension=args.mask_extension,
)
