# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""Remote NPU execution daemon.

Run on a CANN-equipped host:

    python -m triton.backends.ascend.remote.server --port 8001

On the client (the box doing the compiling), set
``TRITON_REMOTE_NPU=<host>:8001`` and run your kernel script as usual.

v1 simplifications:
  - single-process, threaded HTTP server (BaseHTTPRequestHandler)
  - one device (whichever ``torch.npu.current_device()`` reports)
  - per-request torch tensor allocation on NPU; freed at end of request
  - all tensor args treated as both input and output
  - the kernel binary is shipped on every request (binary cache is a
    one-line addition later — see _KernelCache)
"""
from __future__ import annotations

import argparse
import logging
import socketserver
import sys
import threading
from collections import namedtuple
from http.server import HTTPServer
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

from .protocol import ArgPayload, LaunchRequest, LaunchResponse, TensorPayload
from .tensor_codec import from_npu_tensor, to_npu_tensor
from .transport import make_request_handler

_log = logging.getLogger("triton.ascend.remote.server")


# --------- metadata / launcher reconstruction ---------

def _rebuild_metadata(d: Dict[str, Any]):
    """Inverse of client_driver._metadata_to_dict."""
    from triton.backends.compiler import GPUTarget

    md = dict(d)
    target = md.get("target")
    if isinstance(target, dict):
        md["target"] = GPUTarget(target["backend"], target["arch"], target["warp_size"])
    if "cluster_dims" in md and not isinstance(md["cluster_dims"], tuple):
        md["cluster_dims"] = tuple(md["cluster_dims"])
    KernelMetadata = namedtuple("KernelMetadata", sorted(md.keys()))
    return KernelMetadata(**md)


def _fake_src(signature: Dict[int, str], constants: Dict[int, Any],
              fn_arg_names: list) -> SimpleNamespace:
    """A stand-in for ASTSource carrying only what NPULauncher reads."""
    return SimpleNamespace(
        signature=signature,
        constants=constants,
        fn=SimpleNamespace(arg_names=list(fn_arg_names)),
    )


# --------- kernel cache ---------

_KernelCacheEntry = namedtuple(
    "_KernelCacheEntry", ["function", "launcher", "n_max_threads"]
)


class _KernelCache:
    """Hash-keyed cache of (CANN function handle, NPULauncher instance).

    A miss costs one load_binary + one make_npu_launcher_stub C++ compile.
    A hit is free.
    """

    def __init__(self):
        self._entries: Dict[str, _KernelCacheEntry] = {}
        self._lock = threading.Lock()

    def get_or_load(self, request: LaunchRequest, device: int) -> _KernelCacheEntry:
        key = request.packed_metadata.get("hash") or request.metadata.get("hash")
        if not key:
            # Fall back to per-binary identity. Slower (no reuse across launches)
            # but correct.
            key = f"binary:{id(request.binary)}"
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                return entry

        # Build outside the lock — these are slow.
        from triton.backends.ascend.driver import NPUDriver, NPULauncher
        driver = NPUDriver()
        module, function, n_regs, n_spills, n_max_threads = driver.utils.load_binary(
            request.kernel_name, request.binary, request.shared, device, request.mix_mode,
        )
        metadata = _rebuild_metadata(request.metadata)
        src = _fake_src(request.signature, request.constants, request.fn_arg_names)
        launcher = NPULauncher(src, metadata)
        del module, n_regs, n_spills  # we only need the function handle
        entry = _KernelCacheEntry(
            function=function, launcher=launcher, n_max_threads=n_max_threads
        )

        with self._lock:
            # Last writer wins; both entries are functionally equivalent.
            self._entries[key] = entry
        return entry


# --------- per-request execution ---------

def _materialize_args(request: LaunchRequest):
    """Turn ArgPayloads into the positional args the C launcher expects.

    Returns:
      device_args: list passed straight to launcher.launch as the kernel args.
                   Tensor args become int (raw NPU pointer); scalars pass through.
      live_tensors: list of torch tensors to keep alive until D2H is done.
                    Indexed the same as request.args so we can D2H by position.
    """
    device_args = []
    live_tensors = []
    for arg in request.args:
        if arg.kind == "tensor":
            t = to_npu_tensor(arg.value)
            live_tensors.append(t)
            device_args.append(int(t.data_ptr()))
        else:
            live_tensors.append(None)
            device_args.append(arg.value)
    return device_args, live_tensors


def _handle_launch(request: LaunchRequest, cache: _KernelCache) -> LaunchResponse:
    import torch  # noqa: F401
    import torch_npu  # noqa: F401

    from triton.backends.ascend.driver import NPUDriver
    driver = NPUDriver()
    device = driver.get_current_device()
    stream = driver.get_current_stream(device)

    entry = cache.get_or_load(request, device)
    device_args, live_tensors = _materialize_args(request)

    # See make_launcher in driver.py for the format string. Hooks are None
    # (no profiling in v1); launch_metadata is None.
    entry.launcher(
        request.grid[0], request.grid[1], request.grid[2],
        stream, entry.function,
        dict(request.packed_metadata),  # the C launcher mutates with shape data; defensive copy
        None,                            # launch_metadata
        None,                            # launch_enter_hook
        None,                            # launch_exit_hook
        *device_args,
    )

    # Block until done so D2H sees final values.
    torch.npu.synchronize()

    # D2H every tensor arg. Scalars are echoed unchanged.
    response_args = []
    for src_arg, live in zip(request.args, live_tensors):
        if src_arg.kind == "tensor":
            response_args.append(ArgPayload(kind="tensor", value=from_npu_tensor(live)))
        else:
            response_args.append(src_arg)
    return LaunchResponse(args=response_args)


# --------- server bootstrap ---------

class _ThreadingServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def serve_forever(host: str, port: int) -> None:
    cache = _KernelCache()

    def _handler(request: LaunchRequest) -> LaunchResponse:
        try:
            _log.info("launch: kernel=%s grid=%s args=%d",
                      request.kernel_name, request.grid, len(request.args))
            return _handle_launch(request, cache)
        except Exception:
            _log.exception("launch failed")
            raise

    handler_cls = make_request_handler(_handler)
    server = _ThreadingServer((host, port), handler_cls)
    _log.info("triton-ascend remote server listening on %s:%d", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _log.info("shutdown requested")
    finally:
        server.server_close()


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Triton-Ascend remote execution server")
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address (default: 127.0.0.1; v1 has no auth)")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    serve_forever(args.host, args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
