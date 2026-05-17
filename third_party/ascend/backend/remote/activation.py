# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""Switch Triton to remote-NPU mode when TRITON_REMOTE_NPU is set.

Eagerly imported from ``triton.backends.ascend.__init__`` so the driver
swap happens before any code path triggers NPUDriver instantiation (which
requires CANN on this host).

Two activation modes:
  * DirectConfig: TRITON_REMOTE_NPU=host:port → connect to a pre-running server.
  * BootstrapConfig: TRITON_REMOTE_NPU=ssh://... (and friends) → we spawn the
    server ourselves via SshBootstrap, then connect to a local-forwarded port.

When activated, ``driver.active`` is replaced with ``RemoteNPUDriver`` and
``CompiledKernel._init_handles`` is patched to construct ``RemoteNPULauncher``
with the binary attached (bypassing ``load_binary`` entirely on the client).
"""
from __future__ import annotations

import logging
import threading

_log = logging.getLogger(__name__)
_activated = False
_lock = threading.Lock()


def maybe_enable_remote_mode() -> bool:
    """Activate remote mode iff config resolves to anything actionable.

    Returns whether remote mode is now (or was already) active.
    """
    global _activated
    from .config import resolve, BootstrapConfig, DirectConfig

    cfg = resolve()
    if cfg is None:
        return False
    with _lock:
        if _activated:
            return True
        if isinstance(cfg, DirectConfig):
            _install_direct(cfg)
        elif isinstance(cfg, BootstrapConfig):
            _install_bootstrap(cfg)
        else:
            raise RuntimeError(f"unexpected config type {type(cfg).__name__}")
        _activated = True
    return True


def _install_direct(cfg) -> None:
    from .transport import HttpTransport
    transport = HttpTransport(host=cfg.host, port=cfg.port)
    _log.info("triton-ascend remote mode: direct connect to %s:%d", cfg.host, cfg.port)
    _install_driver_and_patch(transport)


def _install_bootstrap(cfg) -> None:
    from .bootstrap import SshBootstrap
    from .transport import HttpTransport
    boot = SshBootstrap(cfg)
    host, port = boot.start()
    transport = HttpTransport(host=host, port=port)
    _log.info(
        "triton-ascend remote mode: bootstrapped %s (container=%s) → 127.0.0.1:%d",
        cfg.ssh_host, cfg.container or "<none>", port,
    )
    _install_driver_and_patch(transport)


def _install_driver_and_patch(transport) -> None:
    # 1. Swap the active driver.
    from triton.runtime.driver import driver
    from .client_driver import RemoteNPUDriver, RemoteNPULauncher

    driver.set_active(RemoteNPUDriver(transport=transport))

    # 2. Patch CompiledKernel._init_handles so the launcher is constructed
    # with the binary attached, bypassing load_binary entirely.
    from triton.compiler import compiler as _compiler_mod

    _original_init_handles = _compiler_mod.CompiledKernel._init_handles

    def _remote_init_handles(self):
        if self.module is not None:
            return
        self._run = RemoteNPULauncher(
            src=self.src,
            metadata=self.metadata,
            binary=self.kernel,
            packed_metadata=self.packed_metadata,
            transport=transport,
        )
        # Faked values. The client never invokes load_binary, so these
        # handles are opaque sentinels; the runner only forwards them.
        self.module = "<remote>"
        self.function = "<remote>"
        self.n_regs = 0
        self.n_spills = 0
        self.n_max_threads = 1 << 30

    _compiler_mod.CompiledKernel._init_handles = _remote_init_handles
    _compiler_mod.CompiledKernel._original_init_handles = _original_init_handles
