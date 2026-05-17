# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""Resolve remote-execution config from (file, env vars).

The on-disk config and the env vars overlap by design: the file holds the
stable defaults (which host, which container, the long launch command);
env vars override per-invocation (e.g. switch host for one run). Both are
optional — direct ``TRITON_REMOTE_NPU=host:port`` still works without any
of this.

File location: ``$TRITON_REMOTE_NPU_CONFIG`` if set, else
``~/.triton/remote.json``. JSON (no extra deps); TOML is a v2 nicety.

Env-var precedence (highest wins):
    TRITON_REMOTE_NPU                  scheme + host (ssh://user@host[:port] or host:port)
    TRITON_REMOTE_NPU_CONTAINER        docker container name
    TRITON_REMOTE_NPU_LAUNCH           launch command template (uses {port})
    TRITON_REMOTE_NPU_PORT             remote port the server binds to
    TRITON_REMOTE_NPU_SSH_OPTIONS      extra ssh args (shell-split)
    TRITON_REMOTE_NPU_ENSURE_RUNNING   '1' / '0'
"""
from __future__ import annotations

import json
import os
import shlex
from dataclasses import dataclass, field
from typing import List, Optional
from urllib.parse import urlparse


DEFAULT_CONFIG_PATH = "~/.triton/remote.json"
DEFAULT_PORT = 8001


@dataclass
class DirectConfig:
    """A pre-running server we just connect to. No bootstrap."""
    host: str
    port: int


@dataclass
class BootstrapConfig:
    """An ssh+docker bring-up we own end-to-end."""
    ssh_host: str                    # 'user@host' form
    ssh_port: int = 22
    ssh_options: List[str] = field(default_factory=list)

    container: Optional[str] = None  # if None, run launch_command directly on the host
    ensure_running: bool = True      # docker start <container> before docker exec

    launch_command: str = ""         # template; {port} is substituted
    port: int = DEFAULT_PORT         # bound in container; also the local tunnel port


def _load_file(path: str) -> dict:
    expanded = os.path.expanduser(path)
    if not os.path.isfile(expanded):
        return {}
    try:
        with open(expanded) as f:
            data = json.load(f)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"failed to read remote config {expanded!r}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"remote config {expanded!r} must be a JSON object")
    return data


def _split_ssh_endpoint(spec: str) -> tuple[str, int]:
    """Parse 'ssh://[user@]host[:port]' into ('user@host', port). The user@
    prefix is optional; default port 22."""
    if not spec.startswith("ssh://"):
        raise ValueError(f"expected ssh:// URL, got {spec!r}")
    parsed = urlparse(spec)
    host = parsed.hostname
    if not host:
        raise ValueError(f"missing host in {spec!r}")
    user_at = f"{parsed.username}@" if parsed.username else ""
    return f"{user_at}{host}", parsed.port or 22


def resolve():
    """Return one of (None, DirectConfig, BootstrapConfig).

    None means remote mode is off (no env var, no config file with anything
    actionable). DirectConfig means TRITON_REMOTE_NPU was a plain host:port.
    BootstrapConfig means we have to bring the server up ourselves.
    """
    raw_endpoint = os.environ.get("TRITON_REMOTE_NPU", "").strip()
    file_data = _load_file(os.environ.get("TRITON_REMOTE_NPU_CONFIG", DEFAULT_CONFIG_PATH))

    # Bootstrap path: ssh:// URL, OR file specifies ssh_host, OR any
    # bootstrap-only env var is set (which only makes sense in bootstrap mode).
    bootstrap_signal = (
        raw_endpoint.startswith("ssh://")
        or "ssh_host" in file_data
        or any(os.environ.get(k) for k in (
            "TRITON_REMOTE_NPU_CONTAINER",
            "TRITON_REMOTE_NPU_LAUNCH",
        ))
    )

    if bootstrap_signal:
        return _resolve_bootstrap(raw_endpoint, file_data)

    # Direct mode: just a host:port, or nothing.
    if not raw_endpoint:
        return None
    if ":" not in raw_endpoint:
        raise RuntimeError(
            f"TRITON_REMOTE_NPU={raw_endpoint!r} must be 'host:port' or 'ssh://...'"
        )
    host, port_s = raw_endpoint.rsplit(":", 1)
    return DirectConfig(host=host, port=int(port_s))


def _resolve_bootstrap(raw_endpoint: str, file_data: dict) -> BootstrapConfig:
    # File defaults, then env-var overrides.
    cfg = BootstrapConfig(
        ssh_host=file_data.get("ssh_host", ""),
        ssh_port=int(file_data.get("ssh_port", 22)),
        ssh_options=list(file_data.get("ssh_options", [])),
        container=file_data.get("container"),
        ensure_running=bool(file_data.get("ensure_running", True)),
        launch_command=file_data.get("launch_command", ""),
        port=int(file_data.get("port", DEFAULT_PORT)),
    )

    if raw_endpoint.startswith("ssh://"):
        cfg.ssh_host, cfg.ssh_port = _split_ssh_endpoint(raw_endpoint)

    if (v := os.environ.get("TRITON_REMOTE_NPU_CONTAINER")) is not None:
        cfg.container = v or None
    if (v := os.environ.get("TRITON_REMOTE_NPU_LAUNCH")) is not None:
        cfg.launch_command = v
    if (v := os.environ.get("TRITON_REMOTE_NPU_PORT")) is not None:
        cfg.port = int(v)
    if (v := os.environ.get("TRITON_REMOTE_NPU_SSH_OPTIONS")) is not None:
        cfg.ssh_options = shlex.split(v)
    if (v := os.environ.get("TRITON_REMOTE_NPU_ENSURE_RUNNING")) is not None:
        cfg.ensure_running = v.strip() not in ("0", "false", "False", "")

    if not cfg.ssh_host:
        raise RuntimeError(
            "remote bootstrap requires an ssh host. Set TRITON_REMOTE_NPU=ssh://user@host "
            "or put 'ssh_host' in the config file."
        )
    if not cfg.launch_command:
        raise RuntimeError(
            "remote bootstrap requires a launch command. Set TRITON_REMOTE_NPU_LAUNCH "
            "or put 'launch_command' in the config file. The string may include {port} "
            "which is substituted with the server port."
        )
    return cfg
