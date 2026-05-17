# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""Transport layer for remote kernel execution.

v1 uses HTTP with pickle bodies. Both peers must be trusted Python processes
sharing a compatible torch version (no untrusted callers — pickle is unsafe
in the general case). For v1 this means: bind the server to localhost and
reach it via SSH tunnel, or run client+server on the same trusted subnet.

The ``Transport`` ABC is the seam where a future msgpack/gRPC/in-process
implementation can drop in.
"""
from __future__ import annotations

import http.client
import http.server
import pickle
from abc import ABC, abstractmethod
from typing import Callable

from .protocol import LaunchRequest, LaunchResponse


HTTP_PATH = "/launch"
DEFAULT_TIMEOUT_S = 300.0


class Transport(ABC):
    @abstractmethod
    def launch(self, request: LaunchRequest) -> LaunchResponse: ...


class HttpTransport(Transport):
    def __init__(self, host: str, port: int, timeout: float = DEFAULT_TIMEOUT_S):
        self.host = host
        self.port = port
        self.timeout = timeout

    def launch(self, request: LaunchRequest) -> LaunchResponse:
        body = pickle.dumps(request)
        conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        try:
            conn.request(
                "POST", HTTP_PATH, body=body,
                headers={"Content-Type": "application/octet-stream",
                         "Content-Length": str(len(body))},
            )
            resp = conn.getresponse()
            if resp.status != 200:
                raise RuntimeError(
                    f"remote launch HTTP {resp.status}: {resp.reason}"
                )
            payload = resp.read()
        finally:
            conn.close()
        response = pickle.loads(payload)
        if not isinstance(response, LaunchResponse):
            raise RuntimeError(f"unexpected response type {type(response).__name__}")
        if response.error is not None:
            raise RuntimeError(f"remote launch error: {response.error}")
        return response


def parse_endpoint(spec: str) -> tuple[str, int]:
    """Parse 'host:port' (the value of TRITON_REMOTE_NPU)."""
    if ":" not in spec:
        raise ValueError(f"TRITON_REMOTE_NPU must be host:port, got {spec!r}")
    host, port_s = spec.rsplit(":", 1)
    return host, int(port_s)


# ----- server-side helper: handler factory that calls a user-supplied
# callable. Keeps server.py free of HTTP glue.

def make_request_handler(handler: Callable[[LaunchRequest], LaunchResponse]):

    class _Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            # Quieter default. Server.py installs its own logger.
            pass

        def do_POST(self):  # noqa: N802 (BaseHTTPRequestHandler API)
            if self.path != HTTP_PATH:
                self.send_error(404, "unknown path")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                request = pickle.loads(body)
                if not isinstance(request, LaunchRequest):
                    raise RuntimeError(
                        f"unexpected request type {type(request).__name__}"
                    )
                response = handler(request)
            except Exception as exc:  # noqa: BLE001 — surface anything
                import traceback
                response = LaunchResponse(
                    error=f"{exc}\n{traceback.format_exc()}"
                )
            data = pickle.dumps(response)
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return _Handler
