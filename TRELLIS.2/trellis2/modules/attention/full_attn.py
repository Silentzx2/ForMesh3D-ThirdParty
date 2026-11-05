from typing import *
import torch
import math
from . import config


__all__ = [
    'scaled_dot_product_attention',
]


def _naive_sdpa(q, k, v):
    """
    Naive implementation of scaled dot product attention.
    """
    q = q.permute(0, 2, 1, 3)   # [N, H, L, C]
    k = k.permute(0, 2, 1, 3)   # [N, H, L, C]
    v = v.permute(0, 2, 1, 3)   # [N, H, L, C]
    scale_factor = 1 / math.sqrt(q.size(-1))
    attn_weight = q @ k.transpose(-2, -1) * scale_factor
    attn_weight = torch.softmax(attn_weight, dim=-1)
    out = attn_weight @ v
    out = out.permute(0, 2, 1, 3)   # [N, L, H, C]
    return out


def _is_pre_ampere(device=None) -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        if device is None:
            dev_idx = torch.cuda.current_device()
        elif isinstance(device, int):
            dev_idx = device
        elif isinstance(device, torch.device):
            dev_idx = device.index if device.index is not None else torch.cuda.current_device()
        else:
            dev_str = str(device)
            dev_idx = int(dev_str.split(":")[-1]) if ":" in dev_str else torch.cuda.current_device()
        return torch.cuda.get_device_capability(dev_idx)[0] < 8
    except Exception:
        try:
            return torch.cuda.get_device_capability()[0] < 8
        except Exception:
            return False


@overload
def scaled_dot_product_attention(qkv: torch.Tensor) -> torch.Tensor:
    """
    Apply scaled dot product attention.

    Args:
        qkv (torch.Tensor): A [N, L, 3, H, C] tensor containing Qs, Ks, and Vs.
    """
    ...

@overload
def scaled_dot_product_attention(q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
    """
    Apply scaled dot product attention.

    Args:
        q (torch.Tensor): A [N, L, H, C] tensor containing Qs.
        kv (torch.Tensor): A [N, L, 2, H, C] tensor containing Ks and Vs.
    """
    ...

@overload
def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Apply scaled dot product attention.

    Args:
        q (torch.Tensor): A [N, L, H, Ci] tensor containing Qs.
        k (torch.Tensor): A [N, L, H, Ci] tensor containing Ks.
        v (torch.Tensor): A [N, L, H, Co] tensor containing Vs.

    Note:
        k and v are assumed to have the same coordinate map.
    """
    ...

def scaled_dot_product_attention(*args, **kwargs):
    arg_names_dict = {
        1: ['qkv'],
        2: ['q', 'kv'],
        3: ['q', 'k', 'v']
    }
    num_all_args = len(args) + len(kwargs)
    assert num_all_args in arg_names_dict, f"Invalid number of arguments, got {num_all_args}, expected 1, 2, or 3"
    for key in arg_names_dict[num_all_args][len(args):]:
        assert key in kwargs, f"Missing argument {key}"

    if num_all_args == 1:
        qkv = args[0] if len(args) > 0 else kwargs['qkv']
        assert len(qkv.shape) == 5 and qkv.shape[2] == 3, f"Invalid shape for qkv, got {qkv.shape}, expected [N, L, 3, H, C]"
        device = qkv.device

    elif num_all_args == 2:
        q = args[0] if len(args) > 0 else kwargs['q']
        kv = args[1] if len(args) > 1 else kwargs['kv']
        assert q.shape[0] == kv.shape[0], f"Batch size mismatch, got {q.shape[0]} and {kv.shape[0]}"
        assert len(q.shape) == 4, f"Invalid shape for q, got {q.shape}, expected [N, L, H, C]"
        assert len(kv.shape) == 5, f"Invalid shape for kv, got {kv.shape}, expected [N, L, 2, H, C]"
        device = q.device

    elif num_all_args == 3:
        q = args[0] if len(args) > 0 else kwargs['q']
        k = args[1] if len(args) > 1 else kwargs['k']
        v = args[2] if len(args) > 2 else kwargs['v']
        assert q.shape[0] == k.shape[0] == v.shape[0], f"Batch size mismatch, got {q.shape[0]}, {k.shape[0]}, and {v.shape[0]}"
        assert len(q.shape) == 4, f"Invalid shape for q, got {q.shape}, expected [N, L, H, Ci]"
        assert len(k.shape) == 4, f"Invalid shape for k, got {k.shape}, expected [N, L, H, Ci]"
        assert len(v.shape) == 4, f"Invalid shape for v, got {v.shape}, expected [N, L, H, Co]"
        device = q.device    

    if config.BACKEND == 'xformers':
        if 'xops' not in globals():
            import xformers.ops as xops
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=2)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=2)
        _orig_dtype = q.dtype
        if _orig_dtype == torch.bfloat16 and _is_pre_ampere(q.device):
            q, k, v = q.half(), k.half(), v.half()
        out = xops.memory_efficient_attention(q, k, v)
        if out.dtype != _orig_dtype:
            out = out.to(_orig_dtype)
    elif config.BACKEND == 'flash_attn':
        if 'flash_attn' not in globals():
            import flash_attn
        if num_all_args == 1:
            out = flash_attn.flash_attn_qkvpacked_func(qkv)
        elif num_all_args == 2:
            out = flash_attn.flash_attn_kvpacked_func(q, kv)
        elif num_all_args == 3:
            out = flash_attn.flash_attn_func(q, k, v)
    elif config.BACKEND == 'flash_attn_3':
        if 'flash_attn_3' not in globals():
            import flash_attn_interface as flash_attn_3
            if num_all_args == 1:
                out = flash_attn_3.flash_attn_qkvpacked_func(qkv)
            elif num_all_args == 2:
                k, v = kv.unbind(dim=2)
                out = flash_attn_3.flash_attn_func(q, k, v)
            elif num_all_args == 3:
                out = flash_attn_3.flash_attn_func(q, k, v)
    elif config.BACKEND == 'sdpa':
        if 'sdpa' not in globals():
            from torch.nn.functional import scaled_dot_product_attention as sdpa
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=2)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=2)
        _orig_dtype = q.dtype
        if _orig_dtype == torch.bfloat16 and _is_pre_ampere(q.device):
            q, k, v = q.half(), k.half(), v.half()
        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()
        out = sdpa(q, k, v)         # [N, H, L, C]
        out = out.permute(0, 2, 1, 3).contiguous()
        if out.dtype != _orig_dtype:
            out = out.to(_orig_dtype)
    elif config.BACKEND == 'naive':
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=2)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=2)
        out = _naive_sdpa(q, k, v)
    else:
        raise ValueError(f"Unknown attention module: {config.BACKEND}")
    
    return out
