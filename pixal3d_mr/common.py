"""Shared setup of the Pixal3D runs: offline redirects, pipeline configs, GLB export, errors.

The official scripts (inference.py, inference_mv.py) are imported as modules and their Hugging Face /
torch.hub ids redirected to the pixal3d_nnModels and naf rez packages.
"""
import contextlib
import importlib
import os
import shutil
import sys

MAX_NUM_TOKENS = 49152
# run_inference() defaults of both scripts
SS_SAMPLER = {"steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.7, "rescale_t": 5.0}
SHAPE_SAMPLER = {"steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.5, "rescale_t": 3.0}
TEX_SAMPLER = {"steps": 12, "guidance_strength": 1.0, "guidance_rescale": 0.0, "rescale_t": 3.0}
# Pixal3D/TRELLIS.2 frame to glTF Y-up
GLB_ROTATION = [
    [-1, 0, 0, 0],
    [0, 0, -1, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1],
]
REMBG_MODELS = ("birefnet", "rmbg")
RESOLUTIONS = (1024, 1536)

_singleViewConfig = {"name": "pipeline_birefnet.json"}  # read by the patched single-view from_pretrained
_scripts = {}
_setup = []


def env(name, package):
    value = os.environ.get(name)
    if not value or not os.path.exists(value):
        raise RuntimeError(f"{name} is not set or does not exist: the {package} rez package is missing "
                           f"from the node environment.")
    return value


def modelsPath():
    return env("PIXAL3D_MODELS_PATH", "pixal3d_nnModels")


def pixal3dPath():
    return os.path.join(modelsPath(), "Pixal3D")


def pipelineConfig(rembg, multiview=False):
    """Local pipeline json for a background removal model (see pixal3d_nnModels download.py)."""
    if rembg not in REMBG_MODELS:
        raise ValueError(f"Unknown background removal model '{rembg}', expected one of {REMBG_MODELS}.")
    name = f"pipeline{'_mv' if multiview else ''}_{rembg}.json"
    if not os.path.isfile(os.path.join(pixal3dPath(), name)):
        hint = " Run pixal3d_nnModels download.py --rmbg with a Hugging Face token." if rembg == "rmbg" else ""
        raise RuntimeError(f"Missing {name} in {pixal3dPath()}.{hint}")
    return name


def checkResolution(resolution):
    if resolution not in RESOLUTIONS:
        raise ValueError(f"Resolution must be one of {RESOLUTIONS}, got {resolution}.")


def setupProcess():
    """Offline redirects and patches, once per process, before importing pixal3d."""
    if _setup:
        return
    _setup.append(True)
    # Before huggingface_hub / transformers are imported: they read these at import time.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # the scripts default it into their own (read-only) package folder
    os.environ.setdefault("FLEX_GEMM_AUTOTUNE_CACHE_PATH", os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "pixal3d", "flex_gemm_autotune.json"))
    os.makedirs(os.path.dirname(os.environ["FLEX_GEMM_AUTOTUNE_CACHE_PATH"]), exist_ok=True)
    # Triton (flex_gemm) compiles its launcher with $CC/$CXX: one naming a compiler missing from the environment
    # (e.g. CC=clang from the user's shell) breaks it, without them it uses the gcc of the rez environment.
    for var in ("CC", "CXX"):
        if os.environ.get(var) and shutil.which(os.environ[var]) is None:
            del os.environ[var]

    import torch
    # NAF: torch.hub.load("valeoai/NAF", ...) resolves to the naf package's hub cache
    torch.hub.set_dir(env("NAF_TORCH_HUB", "naf"))

    scripts = env("PIXAL3D_SCRIPTS", "pixal3d")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)

    # NAF attention without the 4 GiB upsampled values (same result as natten, see naf_lowmem).
    # DinoV3ProjMultiViewFeatureExtractor inherits _load_naf.
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import DinoV3ProjFeatureExtractor

    from . import naf_lowmem
    loadNaf = DinoV3ProjFeatureExtractor._load_naf

    def _loadNaf(self):
        loadNaf(self)
        naf_lowmem.patch(type(self.naf_model.upsampler))
    DinoV3ProjFeatureExtractor._load_naf = _loadNaf


def loadScript(name):
    """Import an official script ("inference" or "inference_mv") with its hub ids redirected to local files."""
    if name in _scripts:
        return _scripts[name]
    setupProcess()
    module = importlib.import_module(name)

    dino = os.path.join(modelsPath(), "dinov3-vitl16-pretrain-lvd1689m")
    for cfg in module.IMAGE_COND_CONFIGS.values():
        cfg["model_name"] = dino

    if name == "inference":
        # init_pipeline() loads "pipeline.json", whose rembg model is the gated briaai/RMBG-2.0 hub id
        # (inference_mv.init_pipeline takes the config file as an argument)
        fromPretrained = module.Pixal3DImageTo3DPipeline.from_pretrained.__func__
        module.Pixal3DImageTo3DPipeline.from_pretrained = classmethod(
            lambda cls, path, config_file=None: fromPretrained(cls, path, config_file or _singleViewConfig["name"]))

    _scripts[name] = module
    return module


def loadRembgModel(rembg):
    """The background removal model alone, built from the pipeline json as from_pretrained does (same masks)."""
    import json
    setupProcess()
    from pixal3d.pipelines import rembg as rembgModels
    with open(os.path.join(pixal3dPath(), pipelineConfig(rembg))) as f:
        config = json.load(f)["args"]["rembg_model"]
    return getattr(rembgModels, config["name"])(**config["args"])


def setSingleViewConfig(name):
    _singleViewConfig["name"] = name


def samplerParams():
    return {
        "sparse_structure_sampler_params": dict(SS_SAMPLER),
        "shape_slat_sampler_params": dict(SHAPE_SAMPLER),
        "tex_slat_sampler_params": dict(TEX_SAMPLER),
    }


def exportGlb(pipeline, mesh, gridSize, path, decimationTarget, textureSize, transform=None):
    """Remesh, decimate, bake and export, as the official scripts do.

    transform: 4x4 applied to the mesh (unit cube coordinates), default GLB_ROTATION (glTF Y-up, as upstream).
    """
    import gc

    import numpy as np
    import o_voxel
    import torch
    # CuMesh allocates with cudaMalloc, outside of torch's caching allocator: give it back what the
    # generation stages left cached, or it runs out of memory on 16 GB GPUs.
    gc.collect()
    torch.cuda.empty_cache()
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
        coords=mesh.coords, attr_layout=pipeline.pbr_attr_layout,
        grid_size=gridSize, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimationTarget, texture_size=textureSize,
        remesh=True, remesh_band=1, remesh_project=0, use_tqdm=True,
    )
    glb.apply_transform(np.array(GLB_ROTATION if transform is None else transform, dtype=np.float64))
    glb.export(path, extension_webp=True)


@contextlib.contextmanager
def gpuErrors(lowVram, resolution):
    """Turn CUDA out of memory errors into an actionable message."""
    import torch
    try:
        yield
    except torch.OutOfMemoryError as e:
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        hint = "Use a GPU with more free memory (~9 GiB needed)" if lowVram and resolution == 1024 else \
            "Enable Low VRAM and use resolution 1024 (~9 GiB), or use a 24 GB+ GPU"
        raise RuntimeError(f"Out of GPU memory on {torch.cuda.get_device_name(0)} ({total:.0f} GiB). {hint}.") from e


def gpuSummary():
    import torch
    return f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB on {torch.cuda.get_device_name(0)}"
