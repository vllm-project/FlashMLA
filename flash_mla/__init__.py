__version__ = "1.0.0"

import os
import torch
from pathlib import Path

# The Ascend extension is a pybind module (flash_mla.npu) imported by name, so only the CUDA
# build has a library to load here. The NPU driver exposes /dev/davinci_manager, so its
# absence means this is a CUDA machine (the same probe setup.py uses to pick a build target).
IS_ASCEND = os.path.exists("/dev/davinci_manager")

if not IS_ASCEND:
    # Load the compiled extension so its STABLE_TORCH_LIBRARY registrations run.
    # Operators are then available as torch.ops._flashmla_C.<op>. The glob matches
    # the abi3 name (_flashmla_C.abi3.so) and the cpython-*.so name.
    _so_files = list(Path(__file__).parent.glob("_flashmla_C*.so"))
    assert len(_so_files) == 1, f"Expected one _flashmla_C*.so file, found {_so_files}"
    torch.ops.load_library(_so_files[0])

from flash_mla.flash_mla_interface import (
    get_mla_metadata,
    flash_mla_with_kvcache,
    flash_attn_varlen_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_varlen_kvpacked_func,
    flash_mla_sparse_fwd
)

from . import fused_norm_rope_attn_rope_cast

__all__ = [
    "get_mla_metadata",
    "flash_mla_with_kvcache",
    "flash_attn_varlen_func",
    "flash_attn_varlen_qkvpacked_func",
    "flash_attn_varlen_kvpacked_func",
    "flash_mla_sparse_fwd",
    "fused_norm_rope_attn_rope_cast"
]
