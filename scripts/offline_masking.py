"""Object masks of the views of an SfMData: same code as the Pixal3dImageMasking node.

    rez env pixal3d-1.0.0 pixal3d_nnModels-1.0.0 naf-1.0.0 aliceVision-develop -- \
        python offline_masking.py --sfm_data sfm.abc --output_dir masks

Writes masks/<image stem>.png (8 bit grey, white = object), for --masks_folder of offline_inference(_mv).py.
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pixal3d_mr import common, masking  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--sfm_data", required=True, help="SfMData with the views to mask")
parser.add_argument("--view_id", type=int, nargs="+", default=None, help="views to mask (default: all)")
parser.add_argument("--output_dir", default="./masks")
parser.add_argument("--rembg", choices=common.REMBG_MODELS, default="birefnet")
args = parser.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

masking.run(outputDir=args.output_dir, sfmData=args.sfm_data, rembg=args.rembg, viewIds=args.view_id)
