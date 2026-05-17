# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""Encode/decode a single kernel argument for the wire.

The client side does not import torch_npu (it has no NPU). It only sees CPU
tensors. The server has torch + torch_npu and rehydrates onto the device.

Tensor args are always treated as bidirectional in v1: every input is also
copied back D2H after launch and written into the caller's tensor. Triton
kernel signatures don't carry in/out direction, so this is the only safe
choice without a per-tensor annotation system.
"""
from __future__ import annotations

from typing import Any

from .protocol import ArgPayload, TensorPayload


def _torch():
    import torch
    return torch


def _dtype_str(t) -> str:
    # 'torch.float32' -> 'float32'
    s = str(t.dtype)
    return s.split('.', 1)[1] if s.startswith('torch.') else s


def _dtype_from_str(s: str):
    torch = _torch()
    name = s.split('.', 1)[1] if s.startswith('torch.') else s
    return getattr(torch, name)


def encode_arg(arg: Any) -> ArgPayload:
    """Convert one kernel arg from the caller into an ArgPayload."""
    torch = _torch()
    if isinstance(arg, torch.Tensor):
        cpu = arg.detach().contiguous().cpu()
        payload = TensorPayload(
            shape=list(cpu.shape),
            dtype=_dtype_str(cpu),
            data=bytes(cpu.untyped_storage()),
        )
        return ArgPayload(kind='tensor', value=payload)
    # Scalars: int / float / bool / None. Pickle handles primitives natively.
    return ArgPayload(kind='scalar', value=arg)


def encode_args(args) -> list:
    return [encode_arg(a) for a in args]


def apply_response_args(local_args, response_args) -> None:
    """Write D2H tensor bytes from the response back into the caller's tensors.

    Scalar args are not touched. Tensor args are copied in place via
    ``.copy_()`` so that the caller's reference observes the update.
    """
    torch = _torch()
    if len(local_args) != len(response_args):
        raise RuntimeError(
            f"remote launch returned {len(response_args)} args; expected {len(local_args)}"
        )
    for i, (local, remote) in enumerate(zip(local_args, response_args)):
        if remote.kind != 'tensor':
            continue
        if not isinstance(local, torch.Tensor):
            raise RuntimeError(
                f"remote arg {i} is a tensor but local arg is {type(local).__name__}"
            )
        payload: TensorPayload = remote.value
        dtype = _dtype_from_str(payload.dtype)
        updated = torch.frombuffer(
            bytearray(payload.data), dtype=dtype
        ).reshape(payload.shape)
        local.copy_(updated)


# ----- server-side helpers (only used in server.py; safe to import on client
# because torch_npu is only imported inside the functions below) -----

def to_npu_tensor(payload: TensorPayload):
    """Server-side: turn a TensorPayload into an on-device torch tensor."""
    torch = _torch()
    import torch_npu  # noqa: F401 — registers 'npu' device
    dtype = _dtype_from_str(payload.dtype)
    host = torch.frombuffer(bytearray(payload.data), dtype=dtype).reshape(payload.shape)
    return host.to('npu')


def from_npu_tensor(t) -> TensorPayload:
    """Server-side: D2H a tensor into a fresh TensorPayload."""
    cpu = t.detach().contiguous().cpu()
    return TensorPayload(
        shape=list(cpu.shape),
        dtype=_dtype_str(cpu),
        data=bytes(cpu.untyped_storage()),
    )
