# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
Remote kernel execution for Triton-Ascend.

When TRITON_REMOTE_NPU=host:port is set in the environment, kernel launches
are routed to a remote agent that holds the actual CANN runtime and NPU
hardware. The local process only compiles; the remote process loads the
binary, marshals tensor data H2D, launches, syncs, and ships D2H bytes back.

v1 scope (POC):
  - one client, one server
  - HTTP transport with pickle wire format
  - all kernel-arg tensors treated as bidirectional (shuttle every call)
  - no persistent tensor handles, no auth, no binary caching on server

Activation: import this package's ``activation`` module, which is gated on the
TRITON_REMOTE_NPU env var. See ``activation.maybe_enable_remote_mode``.
"""
