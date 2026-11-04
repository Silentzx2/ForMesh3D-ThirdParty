"""Unified FA3, FA2, and PyTorch SDPA attention backend."""

from __future__ import annotations

import torch
import torch.nn.functional as F

_FA3_AVAILABLE = False
_FA2_AVAILABLE = False
_fa3_func = None
_fa3_varlen_func = None
_fa2_func = None
_fa2_varlen_func = None

try:
    from flash_attn_interface import (
        flash_attn_func as _fa3_func,
        flash_attn_varlen_func as _fa3_varlen_func,
    )

    _FA3_AVAILABLE = True
except Exception:
    pass

try:
    from flash_attn import (
        flash_attn_func as _fa2_func,
        flash_attn_varlen_func as _fa2_varlen_func,
    )

    _FA2_AVAILABLE = True
except Exception:
    pass

def _can_use_flash_attn() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability()
        return major >= 8
    except Exception:
        return False


def _get_default_compute_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        try:
            if torch.cuda.get_device_capability()[0] < 8:
                # Pre-Ampere GPUs (Turing sm_75, Volta sm_70) do not support
                # bfloat16 in Cutlass/SDPA memory-efficient attention. Using
                # bfloat16 forces PyTorch to fall back to SDPBackend.MATH, which
                # materializes the full N x N attention matrix (allocating 10+ GiB)
                # and OOMs. Float16 uses hardware Tensor Cores and Cutlass tiled attention.
                return torch.float16
        except Exception:
            pass
    return torch.bfloat16


if _FA3_AVAILABLE and _can_use_flash_attn():
    ATTENTION_BACKEND = "fa3"
elif _FA2_AVAILABLE and _can_use_flash_attn():
    ATTENTION_BACKEND = "fa2"
else:
    ATTENTION_BACKEND = "sdpa"

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    _HAS_SDPA_KERNEL = True
except Exception:
    _HAS_SDPA_KERNEL = False

_COMPUTE_DTYPE = _get_default_compute_dtype()


def _unwrap(output):
    return output[0] if isinstance(output, (tuple, list)) else output


def _sdpa_dense(q, k, v):
    if q.is_cuda and q.dtype == torch.bfloat16:
        try:
            if torch.cuda.get_device_capability(q.device)[0] < 8:
                return _sdpa_dense(q.half(), k.half(), v.half()).to(q.dtype)
        except Exception:
            pass

    if _HAS_SDPA_KERNEL:
        if q.is_cuda:
            try:
                with sdpa_kernel(
                    [
                        SDPBackend.FLASH_ATTENTION,
                        SDPBackend.EFFICIENT_ATTENTION,
                    ]
                ):
                    return F.scaled_dot_product_attention(q, k, v)
            except RuntimeError:
                pass
        with sdpa_kernel(
            [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ]
        ):
            return F.scaled_dot_product_attention(q, k, v)
    with torch.backends.cuda.sdp_kernel(
        enable_flash=True,
        enable_math=True,
        enable_mem_efficient=True,
    ):
        return F.scaled_dot_product_attention(q, k, v)


def _sdpa_masked(q, k, v, mask):
    if q.is_cuda and q.dtype == torch.bfloat16:
        try:
            if torch.cuda.get_device_capability(q.device)[0] < 8:
                return _sdpa_masked(q.half(), k.half(), v.half(), mask).to(q.dtype)
        except Exception:
            pass

    if _HAS_SDPA_KERNEL:
        if q.is_cuda:
            try:
                with sdpa_kernel(
                    [
                        SDPBackend.EFFICIENT_ATTENTION,
                        SDPBackend.FLASH_ATTENTION,
                    ]
                ):
                    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            except RuntimeError:
                pass
        with sdpa_kernel(
            [
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.MATH,
            ]
        ):
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


def flash_attn_qkv(q, k, v):
    """Dense attention with ``(batch, heads, tokens, channels)`` tensors."""
    original_dtype = q.dtype
    if ATTENTION_BACKEND in {"fa3", "fa2"}:
        qf = q.transpose(1, 2).to(_COMPUTE_DTYPE).contiguous()
        kf = k.transpose(1, 2).to(_COMPUTE_DTYPE).contiguous()
        vf = v.transpose(1, 2).to(_COMPUTE_DTYPE).contiguous()
        function = _fa3_func if ATTENTION_BACKEND == "fa3" else _fa2_func
        output = _unwrap(function(qf, kf, vf, causal=False))
        return output.transpose(1, 2).to(original_dtype)
    output = _sdpa_dense(
        q.to(_COMPUTE_DTYPE),
        k.to(_COMPUTE_DTYPE),
        v.to(_COMPUTE_DTYPE),
    )
    return output.to(original_dtype)


def flash_attn_varlen_qkv(q, k, v, valid_mask):
    """Variable-length self-attention; ``True`` mask entries are valid."""
    if valid_mask is None:
        return flash_attn_qkv(q, k, v)

    batch, heads, tokens, channels = q.shape
    original_dtype = q.dtype
    device = q.device
    valid_mask = valid_mask.to(device=device, dtype=torch.bool)
    if valid_mask.shape != (batch, tokens):
        raise ValueError(
            f"valid_mask has shape {tuple(valid_mask.shape)}, expected "
            f"{(batch, tokens)}"
        )
    if not bool(valid_mask.any(dim=1).all()):
        raise ValueError("Each sample must contain at least one valid token")

    if ATTENTION_BACKEND in {"fa3", "fa2"}:
        qf = q.transpose(1, 2).contiguous().to(_COMPUTE_DTYPE)
        kf = k.transpose(1, 2).contiguous().to(_COMPUTE_DTYPE)
        vf = v.transpose(1, 2).contiguous().to(_COMPUTE_DTYPE)
        lengths = valid_mask.sum(dim=1).to(torch.int32)
        cumulative = torch.zeros(batch + 1, dtype=torch.int32, device=device)
        cumulative[1:] = torch.cumsum(lengths, dim=0)
        maximum = int(lengths.max().item())
        valid_indices = valid_mask.reshape(-1).nonzero(as_tuple=True)[0]
        q_packed = qf.reshape(batch * tokens, heads, channels)[valid_indices]
        k_packed = kf.reshape(batch * tokens, heads, channels)[valid_indices]
        v_packed = vf.reshape(batch * tokens, heads, channels)[valid_indices]
        function = (
            _fa3_varlen_func if ATTENTION_BACKEND == "fa3" else _fa2_varlen_func
        )
        packed = _unwrap(
            function(
                q_packed,
                k_packed,
                v_packed,
                cumulative,
                cumulative,
                maximum,
                maximum,
                causal=False,
            )
        )
        full = torch.zeros(
            batch * tokens,
            heads,
            channels,
            device=device,
            dtype=packed.dtype,
        )
        full[valid_indices] = packed
        return full.reshape(batch, tokens, heads, channels).transpose(1, 2).to(
            original_dtype
        )

    mask = valid_mask[:, None, None, :]
    return _sdpa_masked(
        q.to(_COMPUTE_DTYPE),
        k.to(_COMPUTE_DTYPE),
        v.to(_COMPUTE_DTYPE),
        mask,
    ).to(original_dtype)


def flash_attn_varlen_cross_qkv(q, k, v, valid_key_mask):
    """Cross-attention with a padded key/value sequence."""
    if valid_key_mask is None:
        return flash_attn_qkv(q, k, v)

    batch, heads, query_tokens, channels = q.shape
    key_tokens = k.shape[2]
    original_dtype = q.dtype
    device = q.device
    valid_key_mask = valid_key_mask.to(device=device, dtype=torch.bool)
    if valid_key_mask.shape != (batch, key_tokens):
        raise ValueError(
            f"valid_key_mask has shape {tuple(valid_key_mask.shape)}, expected "
            f"{(batch, key_tokens)}"
        )
    if not bool(valid_key_mask.any(dim=1).all()):
        raise ValueError("Each sample must contain at least one valid context token")

    if ATTENTION_BACKEND in {"fa3", "fa2"}:
        qf = q.transpose(1, 2).contiguous().to(_COMPUTE_DTYPE)
        kf = k.transpose(1, 2).contiguous().to(_COMPUTE_DTYPE)
        vf = v.transpose(1, 2).contiguous().to(_COMPUTE_DTYPE)
        query_cumulative = torch.arange(
            0,
            (batch + 1) * query_tokens,
            query_tokens,
            dtype=torch.int32,
            device=device,
        )
        key_lengths = valid_key_mask.sum(dim=1).to(torch.int32)
        key_cumulative = torch.zeros(
            batch + 1, dtype=torch.int32, device=device
        )
        key_cumulative[1:] = torch.cumsum(key_lengths, dim=0)
        valid_indices = valid_key_mask.reshape(-1).nonzero(as_tuple=True)[0]
        q_packed = qf.reshape(batch * query_tokens, heads, channels)
        k_packed = kf.reshape(batch * key_tokens, heads, channels)[valid_indices]
        v_packed = vf.reshape(batch * key_tokens, heads, channels)[valid_indices]
        function = (
            _fa3_varlen_func if ATTENTION_BACKEND == "fa3" else _fa2_varlen_func
        )
        packed = _unwrap(
            function(
                q_packed,
                k_packed,
                v_packed,
                query_cumulative,
                key_cumulative,
                query_tokens,
                int(key_lengths.max().item()),
                causal=False,
            )
        )
        return packed.reshape(
            batch, query_tokens, heads, channels
        ).transpose(1, 2).to(original_dtype)

    mask = valid_key_mask[:, None, None, :]
    return _sdpa_masked(
        q.to(_COMPUTE_DTYPE),
        k.to(_COMPUTE_DTYPE),
        v.to(_COMPUTE_DTYPE),
        mask,
    ).to(original_dtype)
