# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""Wire-protocol dataclasses for remote kernel execution.

Serialized via pickle for v1; both ends are trusted Python processes (run
on localhost or behind an SSH tunnel). Wire format will become msgpack +
explicit codecs in v1.1 so the protocol can survive Python/torch version
skew between client and server.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union


PROTOCOL_VERSION = 1


@dataclass
class TensorPayload:
    """A torch tensor flattened for the wire.

    ``data`` is the raw contiguous bytes of the tensor's storage. The server
    rehydrates by ``torch.frombuffer(data, dtype=...).reshape(shape)`` and
    pushes onto NPU memory.
    """
    shape: List[int]
    dtype: str          # torch dtype string, e.g. 'torch.float32'
    data: bytes


@dataclass
class ArgPayload:
    """One positional kernel argument.

    For ``kind == 'tensor'``, ``value`` is a TensorPayload.
    For ``kind == 'scalar'``, ``value`` is the raw Python scalar (int/float/bool).
    """
    kind: str           # 'tensor' | 'scalar'
    value: Any


@dataclass
class LaunchRequest:
    protocol_version: int = PROTOCOL_VERSION

    # Kernel binary + load_binary args
    binary: bytes = b""
    kernel_name: str = ""
    shared: int = 0
    mix_mode: str = ""

    # Launch dispatch
    grid: Tuple[int, int, int] = (1, 1, 1)
    packed_metadata: Dict[str, Any] = field(default_factory=dict)

    # Server-side launcher reconstruction
    signature: Dict[int, str] = field(default_factory=dict)       # normalized int keys
    constants: Dict[int, Any] = field(default_factory=dict)       # normalized int keys
    fn_arg_names: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)        # full namedtuple as dict

    # Positional kernel args (in signature order, post grid/stream/fn/metadata/hooks)
    args: List[ArgPayload] = field(default_factory=list)


@dataclass
class LaunchResponse:
    protocol_version: int = PROTOCOL_VERSION

    # Tensor args mirrored back after D2H; scalars are echoed unchanged.
    # Same length and ordering as request.args.
    args: List[ArgPayload] = field(default_factory=list)

    # Set when the remote launch failed. Client raises on receipt.
    error: Optional[str] = None
