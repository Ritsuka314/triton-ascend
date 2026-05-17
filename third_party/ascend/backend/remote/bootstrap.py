# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""Spawn and tear down the remote server over SSH.

Strategy:
  1. Pick a local TCP port (free, see ``_pick_free_port``). Use the same number
     as the remote bind port so the launch command and the ``-L`` tunnel agree.
  2. Build the remote shell command:
        [docker start <container> >/dev/null 2>&1 || true; ]
        docker exec -i <container> bash -c <quoted launch_command>
     If no container is configured, run the launch command directly.
  3. ``ssh -L <port>:127.0.0.1:<port> [opts] <ssh_host> <remote_cmd>``
  4. Poll the local port until it accepts a connection (or timeout).
  5. Register an ``atexit`` hook that SIGTERM-then-SIGKILL the ssh process,
     which cascades to docker exec → server.

Errors during bring-up surface immediately. Server-side stderr is left
attached to the parent so the user sees real diagnostics in the terminal.

The ``Bootstrap`` ABC keeps room for a future ``ParamikoBootstrap`` etc.
"""
from __future__ import annotations

import atexit
import logging
import os
import shlex
import signal
import socket
import subprocess
import time
from abc import ABC, abstractmethod
from typing import List, Optional

from .config import BootstrapConfig


_log = logging.getLogger(__name__)

STARTUP_TIMEOUT_S = 60.0
SHUTDOWN_TIMEOUT_S = 5.0


def _pick_free_port() -> int:
    # Ask the kernel for a free ephemeral port, then close immediately. The
    # tiny race (someone else grabs it before ssh binds) is acceptable for v1.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _wait_for_port(host: str, port: int, deadline: float, proc: subprocess.Popen) -> None:
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"ssh process exited with code {proc.returncode} before server came up"
            )
        if _port_open(host, port):
            return
        time.sleep(0.25)
    raise RuntimeError(
        f"timed out after {STARTUP_TIMEOUT_S:.0f}s waiting for remote server on "
        f"{host}:{port}; check the ssh stderr above for what the container is doing"
    )


class Bootstrap(ABC):
    @abstractmethod
    def start(self) -> tuple[str, int]:
        """Bring the server up. Returns (local_host, local_port) for the client."""

    @abstractmethod
    def stop(self) -> None:
        """Tear it down. Safe to call repeatedly."""


class SshBootstrap(Bootstrap):
    def __init__(self, cfg: BootstrapConfig):
        self.cfg = cfg
        self._proc: Optional[subprocess.Popen] = None
        self._port: int = 0
        self._stopped = False

    def start(self) -> tuple[str, int]:
        if self._proc is not None:
            return "127.0.0.1", self._port

        self._port = self._choose_port()
        launch = self.cfg.launch_command.format(port=self._port)
        remote_cmd = self._build_remote_command(launch)
        argv = self._build_ssh_argv(remote_cmd)

        _log.info("remote bootstrap: %s", " ".join(shlex.quote(a) for a in argv))

        # Stdin opened so SSH considers it a non-tty session; stdout/stderr
        # inherit so the user sees server boot output as it happens.
        self._proc = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=None,
            stderr=None,
            # New process group so a SIGTERM to the leader kills the ssh client
            # cleanly without our process group also taking the signal.
            start_new_session=True,
        )
        atexit.register(self.stop)

        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        _wait_for_port("127.0.0.1", self._port, deadline, self._proc)
        _log.info("remote server reachable at 127.0.0.1:%d", self._port)
        return "127.0.0.1", self._port

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        try:
            # Signal the whole session so ssh, docker exec, and the python
            # server all get the message — not just the ssh client.
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            return
        try:
            proc.wait(timeout=SHUTDOWN_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    # ----- helpers -----

    def _choose_port(self) -> int:
        # If the user pinned a port, honor it; else find a free one.
        if self.cfg.port and self.cfg.port > 0:
            return self.cfg.port
        return _pick_free_port()

    def _build_remote_command(self, launch_command: str) -> str:
        parts: List[str] = []
        container = self.cfg.container
        if container:
            if self.cfg.ensure_running:
                parts.append(f"docker start {shlex.quote(container)} >/dev/null 2>&1 || true")
            parts.append(
                f"docker exec -i {shlex.quote(container)} bash -c {shlex.quote(launch_command)}"
            )
        else:
            parts.append(launch_command)
        return "; ".join(parts)

    def _build_ssh_argv(self, remote_cmd: str) -> List[str]:
        argv = [
            "ssh",
            "-p", str(self.cfg.ssh_port),
            "-o", "ExitOnForwardFailure=yes",
            "-o", "ServerAliveInterval=30",
            "-L", f"{self._port}:127.0.0.1:{self._port}",
        ]
        argv.extend(self.cfg.ssh_options)
        argv.append(self.cfg.ssh_host)
        argv.append(remote_cmd)
        return argv
