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


def get_compute_dtype(device=None) -> torch.dtype:
    """Return bfloat16 on modern CUDA devices, and float32 on CPU or pre-Ampere GPUs.
    Preserves 8-bit DiT exponent range to prevent FP16 NaN overflow during Euler flow matching.
    """
    if device is not None and getattr(device, "type", str(device)) == "cpu":
        return torch.float32
    if _is_pre_ampere(device):
        return torch.float32
    return torch.bfloat16


_get_default_compute_dtype = get_compute_dtype


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

_COMPUTE_DTYPE = get_compute_dtype()


def _unwrap(output):
    return output[0] if isinstance(output, (tuple, list)) else output


def _sdpa_dense(q, k, v):
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    if q.is_cuda and _is_pre_ampere(q.device) and (q.dtype == torch.bfloat16 or k.dtype == torch.bfloat16 or v.dtype == torch.bfloat16):
        return _sdpa_dense(q.half(), k.half(), v.half()).to(q.dtype).contiguous()
    if k.dtype != q.dtype:
        k = k.to(q.dtype)
    if v.dtype != q.dtype:
        v = v.to(q.dtype)

    if q.is_cuda and _is_pre_ampere(q.device):
        try:
            import xformers.ops as xops
            q_xf = q.transpose(1, 2).contiguous()
            k_xf = k.transpose(1, 2).contiguous()
            v_xf = v.transpose(1, 2).contiguous()
            out = xops.memory_efficient_attention(q_xf, k_xf, v_xf)
            return out.transpose(1, 2).to(q.dtype).contiguous()
        except Exception:
            pass

        # Try hardware Cutlass memory-efficient attention first for float16 (sm < 80 Tensor Cores)
        if _HAS_SDPA_KERNEL and q.dtype == torch.float16:
            try:
                with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
                    return F.scaled_dot_product_attention(q, k, v).contiguous()
            except RuntimeError:
                pass

        # Query chunking fallback for pre-Ampere when sequence length is large (prevents 10GB MATH OOM)
        if q.shape[2] > 2048:
            chunk_size = 1024
            chunks = []
            for i in range(0, q.shape[2], chunk_size):
                qc = q[:, :, i:i + chunk_size, :]
                chunks.append(F.scaled_dot_product_attention(qc, k, v))
            return torch.cat(chunks, dim=2).contiguous()

    if _HAS_SDPA_KERNEL:
        if q.is_cuda:
            try:
                with sdpa_kernel(
                    [
                        SDPBackend.FLASH_ATTENTION,
                        SDPBackend.EFFICIENT_ATTENTION,
                    ]
                ):
                    return F.scaled_dot_product_attention(q, k, v).contiguous()
            except RuntimeError:
                pass
        with sdpa_kernel(
            [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ]
        ):
            return F.scaled_dot_product_attention(q, k, v).contiguous()
    with torch.backends.cuda.sdp_kernel(
        enable_flash=True,
        enable_math=True,
        enable_mem_efficient=True,
    ):
        return F.scaled_dot_product_attention(q, k, v).contiguous()


def _sdpa_masked(q, k, v, mask):
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    if q.is_cuda and _is_pre_ampere(q.device) and (q.dtype == torch.bfloat16 or k.dtype == torch.bfloat16 or v.dtype == torch.bfloat16):
        return _sdpa_masked(q.half(), k.half(), v.half(), mask).to(q.dtype).contiguous()
    if k.dtype != q.dtype:
        k = k.to(q.dtype)
    if v.dtype != q.dtype:
        v = v.to(q.dtype)

    if q.is_cuda and _is_pre_ampere(q.device):
        try:
            import xformers.ops as xops
            q_xf = q.transpose(1, 2).contiguous()
            k_xf = k.transpose(1, 2).contiguous()
            v_xf = v.transpose(1, 2).contiguous()
            out = xops.memory_efficient_attention(q_xf, k_xf, v_xf, attn_bias=mask)
            return out.transpose(1, 2).to(q.dtype).contiguous()
        except Exception:
            pass

        # Try hardware Cutlass memory-efficient attention first for float16 (sm < 80 Tensor Cores)
        if _HAS_SDPA_KERNEL and q.dtype == torch.float16:
            try:
                with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
                    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask).contiguous()
            except RuntimeError:
                pass

        # Query chunking fallback for pre-Ampere when sequence length is large (prevents 10GB MATH OOM)
        if q.shape[2] > 2048:
            chunk_size = 1024
            chunks = []
            for i in range(0, q.shape[2], chunk_size):
                qc = q[:, :, i:i + chunk_size, :]
                chunks.append(F.scaled_dot_product_attention(qc, k, v, attn_mask=mask))
            return torch.cat(chunks, dim=2).contiguous()

    if _HAS_SDPA_KERNEL:
        if q.is_cuda:
            try:
                with sdpa_kernel(
                    [
                        SDPBackend.EFFICIENT_ATTENTION,
                        SDPBackend.FLASH_ATTENTION,
                    ]
                ):
                    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask).contiguous()
            except RuntimeError:
                pass
        with sdpa_kernel(
            [
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.MATH,
            ]
        ):
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask).contiguous()
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask).contiguous()


def flash_attn_qkv(q, k, v):
    """Dense attention with ``(batch, heads, tokens, channels)`` tensors."""
    original_dtype = q.dtype
    is_pre_amp = _is_pre_ampere(q.device)
    compute_dtype = torch.float16 if (is_pre_amp and original_dtype == torch.float16) else get_compute_dtype(q.device)
    if ATTENTION_BACKEND in {"fa3", "fa2"}:
        qf = q.transpose(1, 2).to(compute_dtype).contiguous()
        kf = k.transpose(1, 2).to(compute_dtype).contiguous()
        vf = v.transpose(1, 2).to(compute_dtype).contiguous()
        function = _fa3_func if ATTENTION_BACKEND == "fa3" else _fa2_func
        output = _unwrap(function(qf, kf, vf, causal=False))
        return output.transpose(1, 2).to(original_dtype)
    output = _sdpa_dense(
        q.to(compute_dtype).contiguous(),
        k.to(compute_dtype).contiguous(),
        v.to(compute_dtype).contiguous(),
    )
    return output.to(original_dtype).contiguous()


def flash_attn_varlen_qkv(q, k, v, valid_mask):
    """Variable-length self-attention; ``True`` mask entries are valid."""
    if valid_mask is None:
        return flash_attn_qkv(q, k, v)

    batch, heads, tokens, channels = q.shape
    original_dtype = q.dtype
    device = q.device
    is_pre_amp = _is_pre_ampere(device)
    compute_dtype = torch.float16 if (is_pre_amp and original_dtype == torch.float16) else get_compute_dtype(device)
    valid_mask = valid_mask.to(device=device, dtype=torch.bool)
    if valid_mask.shape != (batch, tokens):
        raise ValueError(
            f"valid_mask has shape {tuple(valid_mask.shape)}, expected "
            f"{(batch, tokens)}"
        )
    if not bool(valid_mask.any(dim=1).all()):
        raise ValueError("Each sample must contain at least one valid token")

    if ATTENTION_BACKEND in {"fa3", "fa2"}:
        qf = q.transpose(1, 2).contiguous().to(compute_dtype)
        kf = k.transpose(1, 2).contiguous().to(compute_dtype)
        vf = v.transpose(1, 2).contiguous().to(compute_dtype)
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
        q.to(compute_dtype).contiguous(),
        k.to(compute_dtype).contiguous(),
        v.to(compute_dtype).contiguous(),
        mask,
    ).to(original_dtype).contiguous()


def flash_attn_varlen_cross_qkv(q, k, v, valid_key_mask):
    """Cross-attention with a padded key/value sequence."""
    if valid_key_mask is None:
        return flash_attn_qkv(q, k, v)

    batch, heads, query_tokens, channels = q.shape
    key_tokens = k.shape[2]
    original_dtype = q.dtype
    device = q.device
    is_pre_amp = _is_pre_ampere(device)
    compute_dtype = torch.float16 if (is_pre_amp and original_dtype == torch.float16) else get_compute_dtype(device)
    valid_key_mask = valid_key_mask.to(device=device, dtype=torch.bool)
    if valid_key_mask.shape != (batch, key_tokens):
        raise ValueError(
            f"valid_key_mask has shape {tuple(valid_key_mask.shape)}, expected "
            f"{(batch, key_tokens)}"
        )
    if not bool(valid_key_mask.any(dim=1).all()):
        raise ValueError("Each sample must contain at least one valid context token")

    if ATTENTION_BACKEND in {"fa3", "fa2"}:
        qf = q.transpose(1, 2).contiguous().to(compute_dtype)
        kf = k.transpose(1, 2).contiguous().to(compute_dtype)
        vf = v.transpose(1, 2).contiguous().to(compute_dtype)
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
        q.to(compute_dtype).contiguous(),
        k.to(compute_dtype).contiguous(),
        v.to(compute_dtype).contiguous(),
        mask,
    ).to(original_dtype).contiguous()
