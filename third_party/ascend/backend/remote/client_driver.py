# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""Client-side driver and launcher for remote NPU execution.

The local machine never instantiates real CANN. ``RemoteNPUDriver`` is a
duck-typed sibling of ``NPUDriver`` that satisfies the bits of the driver
contract Triton's compile/launch path actually exercises:

  - get_current_device / get_current_stream  (faked)
  - get_current_target                       (from TRITON_ASCEND_ARCH)
  - launcher_cls                             (RemoteNPULauncher)
  - utils.load_binary                        (no-op — binary travels in the request)
  - utils.get_device_properties              (stub with generous limits)

The launcher itself extracts everything it needs from its (src, metadata,
binary, packed_metadata) inputs at construction time, then on each call
serializes the args and round-trips through a Transport.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from triton.backends.driver import DriverBase
from triton.backends.compiler import GPUTarget

from .protocol import LaunchRequest
from .tensor_codec import apply_response_args, encode_args
from .transport import Transport


def _resolve_arch() -> str:
    # Reuse the existing convention — same env var the real NPUDriver reads.
    from triton.backends.ascend.utils import get_ascend_arch_from_env
    arch = get_ascend_arch_from_env()
    if not arch:
        raise RuntimeError(
            "TRITON_REMOTE_NPU is set but no Ascend arch was provided. "
            "Set TRITON_ASCEND_ARCH (e.g. 'Ascend910B1') so the local "
            "compiler knows what to target."
        )
    return arch


class RemoteNPUUtils:
    """Stub for the local side. Never touches CANN."""

    def load_binary(self, name, kernel, shared, device, mix_mode):
        # The binary travels in the LaunchRequest. Return shapes that match
        # what CompiledKernel._init_handles expects to unpack.
        del name, kernel, shared, device, mix_mode
        return (None, None, 0, 0, 1 << 30)  # module, function, n_regs, n_spills, n_max_threads

    def get_device_properties(self, device):
        del device
        # Generous limits so CompiledKernel's shared-memory check never fires
        # on the client (the real check, if any, will happen on the server).
        return {"max_shared_mem": 1 << 30, "num_aicore": 24, "num_vectorcore": 48}

    def get_arch(self):
        return _resolve_arch()

    def get_aicore_num(self):
        return self.get_device_properties(0)["num_aicore"]

    def get_aivector_core_num(self):
        return self.get_device_properties(0)["num_vectorcore"]


class RemoteNPULauncher:
    """Replaces NPULauncher on the client side. Sends launches over a Transport.

    Construction is non-standard: in addition to the (src, metadata) signature
    the real launcher uses, we accept the compiled binary and the packed
    metadata. The patched ``CompiledKernel._init_handles`` (see activation.py)
    is what constructs us; nothing else does.
    """

    def __init__(self, src, metadata, binary: bytes, packed_metadata: Dict[str, Any],
                 transport: Transport):
        self._transport = transport
        self._binary = binary
        self._packed_metadata = packed_metadata
        self._kernel_name = metadata.kernel_name
        self._shared = metadata.shared
        self._mix_mode = metadata.mix_mode

        # Normalize signature/constants to int-keyed dicts (same logic
        # NPULauncher uses in driver.py — see NPULauncher.__init__).
        constants = src.constants if hasattr(src, "constants") else {}
        signature = src.signature
        arg_names = list(src.fn.arg_names)

        def _key(i):
            return arg_names.index(i) if isinstance(i, str) else i

        self._signature = {_key(k): v for k, v in signature.items()}
        self._constants = {_key(k): v for k, v in constants.items()}
        self._fn_arg_names = arg_names

        # Cache a frozen metadata dict for shipping.
        self._metadata_dict = _metadata_to_dict(metadata)

    def __call__(self, *args, **kwargs):
        # The C++ launch wrapper takes:
        #   gridX, gridY, gridZ, stream, function,
        #   packedMetadata, launch_metadata,
        #   launch_enter_hook, launch_exit_hook,
        #   *kernel_args
        # See make_launcher in driver.py.
        if len(args) < 9:
            raise RuntimeError(f"RemoteNPULauncher: unexpected args layout ({len(args)} args)")
        grid_x, grid_y, grid_z = args[0], args[1], args[2]
        # args[3] stream and args[4] function are local handles we ignore.
        # args[5] packed_metadata is also already in self._packed_metadata
        # but we trust the caller's copy (in case it was mutated).
        packed_metadata = args[5]
        # args[6..8] are launch_metadata + profiling hooks. v1 drops profiling.
        kernel_args = list(args[9:])

        encoded_args = encode_args(kernel_args)
        request = LaunchRequest(
            binary=self._binary,
            kernel_name=self._kernel_name,
            shared=self._shared,
            mix_mode=self._mix_mode,
            grid=(int(grid_x), int(grid_y), int(grid_z)),
            packed_metadata=dict(packed_metadata),
            signature=self._signature,
            constants=self._constants,
            fn_arg_names=self._fn_arg_names,
            metadata=self._metadata_dict,
            args=encoded_args,
        )
        response = self._transport.launch(request)
        apply_response_args(kernel_args, response.args)


def _metadata_to_dict(metadata) -> Dict[str, Any]:
    """Flatten a KernelMetadata namedtuple into a plain dict.

    The server rebuilds a matching namedtuple from this so that
    ``make_launcher`` (which reads attributes like ``mix_mode``,
    ``workspace_size``, etc.) keeps working unchanged.
    """
    if hasattr(metadata, "_asdict"):
        d = dict(metadata._asdict())
    else:
        d = dict(metadata)
    # GPUTarget is a namedtuple too; flatten so pickle on the wire stays
    # decoupled from triton's class layout.
    target = d.get("target")
    if target is not None and hasattr(target, "_asdict"):
        d["target"] = dict(target._asdict())
    return d


class RemoteNPUDriver(DriverBase):
    """Active driver when TRITON_REMOTE_NPU is set."""

    def __init__(self, transport: Transport):
        self._transport = transport
        self.utils = RemoteNPUUtils()
        # launcher_cls is read by CompiledKernel; the patched _init_handles
        # constructs the launcher manually, so this attribute is largely
        # vestigial. Set it for any third-party code that introspects it.
        self.launcher_cls = self._build_launcher
        self.binary_ext = "npubin"
        super().__init__()

    def _build_launcher(self, src, metadata):
        # This path should not be hit because activation.py patches
        # CompiledKernel._init_handles to construct RemoteNPULauncher with
        # the binary attached. Surface a loud error if it ever is.
        raise RuntimeError(
            "RemoteNPUDriver.launcher_cls was called directly. "
            "activation.maybe_enable_remote_mode() must run before "
            "CompiledKernel._init_handles."
        )

    @classmethod
    def is_active(cls):
        return bool(os.environ.get("TRITON_REMOTE_NPU"))

    def get_current_target(self):
        return GPUTarget("npu", _resolve_arch(), 0)

    def get_current_device(self):
        # Opaque on the client — server picks its own current device.
        return 0

    def set_current_device(self, device):
        # No-op on the client; server is single-device in v1.
        return None

    def get_current_stream(self, device: Optional[int] = None):
        # Opaque stream handle. The server creates its own.
        del device
        return 0

    def get_active_torch_device(self):
        import torch
        return torch.device("cpu")

    def get_benchmarker(self):
        from triton.testing import do_bench
        return do_bench

    def get_device_interface(self):
        # Triton uses this for context managers (torch.cuda-like API).
        # On the client we don't have torch.npu; return a minimal stand-in.
        raise NotImplementedError(
            "device_interface is not available in remote mode (v1)."
        )

    def get_empty_cache_for_benchmark(self):
        raise NotImplementedError(
            "benchmark cache is not available in remote mode (v1)."
        )

    def clear_cache(self, cache):
        del cache

    def map_python_to_cpp_type(self, ty: str) -> str:
        # Reuse the real ty_to_cpp — it's pure-Python, no CANN dep.
        from triton.backends.ascend.driver import ty_to_cpp
        return ty_to_cpp(ty)
