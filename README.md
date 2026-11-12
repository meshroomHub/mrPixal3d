# mrPixal3d

Meshroom plugin (rez package) running [Pixal3D](https://github.com/TencentARC/Pixal3D) image-to-3D inference,
fully offline on rez packages.

Node environment (`MRPIXAL3D_SUBREQUIRES`):

| Package | Content |
|---|---|
| `pixal3d-1.0.0` | Pixal3D code and its CUDA stack (TRELLIS.2 ops, o_voxel, cumesh, flex_gemm, nvdiffrast, MoGe, natten): pytorch-2.14, cuda-12.8, built for sm_75 (Turing) and sm_86 (Ampere, also runs on Ada), attention through torch sdpa |
| `pixal3d_nnModels-1.0.0` | Weights: Pixal3D (single and multi-view), DINOv3 ViT-L/16, MoGe-2 ViT-L, BiRefNet (+ RMBG-2.0 on demand) |
| `naf-1.0.0` | NAF feature upsampler as an offline torch.hub cache (needs `natten`) |
| `aliceVision-develop` | `pyalicevision`, to read SfMData cameras |

## Object masks

`Masks Folder` (Pixal3D nodes): object masks used instead of removing the background of the views. It follows the
AliceVision convention (FeatureExtraction, PrepareDenseScene), so the masks can come from `Pixal3dImageMasking`,
`ImageMasking`, `MaskProcessing` or elsewhere: `<viewId>.<ext>`, else `<image name>.<ext>` (`Mask Extension`: png, exr, jpg), grey,
white = object (0 = background), the size of the original (distorted) image. Soft masks are kept: the mask becomes
the alpha of the view image, then is undistorted (and warped) with it. A view without a mask file in the folder gets
its background removed by the node, as without `Masks Folder`.

## Node: Pixal3dImageMasking

The object mask of each view of an SfMData (`<image>.png`, the view image file name without extension, 8 bit grey,
white = object, original image size), with the background removal model of the Pixal3D pipelines (`Background
Removal`, as below). An image that already has an alpha channel gets its alpha as mask. These are the masks the
Pixal3D nodes would compute: computed once, they can be checked or edited before the generation, and fed to their
`Masks Folder`.

## Node: Pixal3dImageTo3D

Each view of an SfMData (no direct image input), processed one after the other with the models loaded once, to a
textured PBR mesh (`<image>_mesh.glb`, `<image>` being the view image file name without extension), plus
`<image>_preprocessed.png` (background removed, centered: what the model sees) and `<image>_camera.json` (FOV in
radians, distance, scale used for the pixel alignment, the view, its intrinsics, pose and the crop box in original
pixels, for aligning the mesh with the SfM later). The view image is read with pyalicevision and its lens distortion
removed (pinhole intrinsics only). `Skip Failed Views` (default on): a view that fails is logged and skipped; out of
GPU memory, or every view failing, still stops the node.

Same calls, order and sampler settings as Pixal3D's `scripts/inference.py`, so a given seed gives the same mesh.
Differences: hub ids are redirected to the local packages, the decimation target and texture size are exposed,
the camera can come from the SfM calibration.

`Camera Source`:
- `sfm` (default): the calibrated focal of the view. Pixal3D sees a square crop around the object, so the FOV is
  the crop's: `2 atan(cropWidth / (2 fx))`, the crop being computed exactly as upstream's `preprocess_image` does.
  Pixal3D assumes the object is on the optical axis: the off-axis angle of the crop is logged (warning above 10 deg),
  prefer a view with the object near the image center.
- `moge` (as upstream): MoGe-2 estimates the FOV of the preprocessed image.

`scripts/offline_inference.py` also takes an image file (`--image`) instead of an SfMData view, and a manual FOV
(`--fov`, degrees).

With `Masks Folder`, the view's mask replaces the alpha the image may have; the background removal model is not
run on it.

- `Background Removal`: `birefnet` (ZhengPeng7/BiRefNet, MIT) by default. `rmbg` is the upstream default
  briaai/RMBG-2.0 (CC BY-NC 4.0, gated): download it with `pixal3d_nnModels/1.0.0/download.py --rmbg`.
- GPU: ~9 GB peak with `Low VRAM` and resolution 1024 (~5 min on an RTX A4000). The default settings (all models
  on the GPU, resolution 1536) need 24 GB+.
- NAF's neighborhood attention is replaced by `pixal3d_mr/naf_lowmem.py`: same result as natten (tested to 1e-4,
  borders included) without upsampling keys/values first. Upstream needs 4 GiB more for the 1024 px texture
  conditioning alone, which does not fit on 16 GB GPUs. ComfyUI uses the same idea (with zero padding at borders).

## Node: Pixal3dMultiViewTo3D

All the posed views of an SfMData (the only input) to a textured PBR mesh **in the SfM frame**, with Pixal3D's
multi-view weights (`ckpts/*_mv`). Pinhole intrinsics, with or without distortion.

Pixal3D expects the object centered in a unit cube, square views with a centered principal point and the main view
at the canonical front pose. `pixal3d_mr/sfm_views.py` builds that from the SfM:
- Object bounds (`Bounds Source`): `landmarks`, the SfM landmarks inside the background removal mask of (90% of)
  the views that see them in frame (a visual hull test), their 1-99% box in the object frame; or `manual`
  (`Bounds Center`, `Bounds Size` = largest extent, SfM units).
- Each view becomes a virtual camera at the same position, rotated to look at the object center, with the FOV
  framing it like upstream's example rig (half width 0.55 object sizes at the center); its 1024 px image is the
  undistorted photo warped by the exact rotation homography.
- `Main View Id` (default: the view looking the most directly at the object) defines the object frame: its virtual
  camera is the canonical front view.
- The views are written as an upstream views folder (`views/transforms.json`, with the object frame in the SfM under
  `sfm`), processed as `scripts/inference_mv.py` does, and the mesh is transformed back to the SfM frame.

`Masks Folder`: the masks are warped with the views (the views folder images are then RGBA, and not matted again by
`inference_mv.py`'s `load_views`), and used for the `landmarks` bounds. Without it, every view is matted twice as
upstream (the alpha the images may have is ignored): once for the landmark bounds, once in `load_views`.

GPU memory and time grow with the number of views: ~11.5 GB peak with 4 views, `Low VRAM` and resolution 1024
(~4 min on an RTX A4000).

Stage code lives in `pixal3d_mr/` and is only imported from `processChunk`, in the node environment.

## Scripts

Same code as the nodes, from the command line:

```
rez env pixal3d-1.0.0 pixal3d_nnModels-1.0.0 naf-1.0.0 aliceVision-develop -- \
    python scripts/offline_masking.py --sfm_data sfm.abc --output_dir masks
rez env pixal3d-1.0.0 pixal3d_nnModels-1.0.0 naf-1.0.0 aliceVision-develop -- \
    python scripts/offline_inference.py --sfm_data sfm.abc --view_id 123 --output_dir out
rez env pixal3d-1.0.0 pixal3d_nnModels-1.0.0 naf-1.0.0 aliceVision-develop -- \
    python scripts/offline_inference_mv.py --sfm_data sfm.abc --output_dir out_mv [--masks_folder masks]
```

## Tests

```
rez env pixal3d-1.0.0 pixal3d_nnModels-1.0.0 naf-1.0.0 aliceVision-develop pytest psutil tomli \
    -c 'PYTHONPATH=$PYTHONPATH:<meshroom repo> python -m pytest tests -v'
```

The GPU tests (NAF parity with natten, single-view, SfM and multi-view end-to-end) need a 16 GB+ GPU and take
~15 min.
