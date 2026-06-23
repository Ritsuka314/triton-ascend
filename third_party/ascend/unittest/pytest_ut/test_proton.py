# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""
Tests for proton instrumentation support on Ascend:

- ``is_a5`` / ``triton_support_simt`` arch classification (host-only, run anywhere).
- The ``make_ttir`` guard that rejects proton ops outside ``compile_mode='simt_only'``
  and accepts them in pure SIMT. The guard tests need a proton-enabled build and
  A5 / 910_95 hardware, so they are gated accordingly.
"""

import pytest
import torch
import triton
import triton.language as tl

from triton.tools.get_ascend_devices import is_a5, is_compile_on_910_95
from triton.backends.ascend.utils import triton_support_simt

# A5 / 910_95-only gate, matching the xfail convention in test_simd_simt.py.
a5_only = pytest.mark.xfail(not is_compile_on_910_95,
                            reason="proton on Ascend requires SIMT-capable A5 / 910_95 hardware", run=False)

# proton is only importable when built with TRITON_BUILD_PROTON=ON.
try:
    import triton.profiler as proton
    import triton.profiler.language as pl
    HAS_PROTON = True
except ImportError:
    HAS_PROTON = False

proton_required = pytest.mark.skipif(not HAS_PROTON, reason="proton not built (TRITON_BUILD_PROTON=OFF)")


# ---------------------------------------------------------------------------
# Host-only: arch classification. Pure functions, run on any machine.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("arch, expected", [
    ("Ascend910_95xx", True),    # A5 production silicon
    ("Ascend910_9579", True),    # A5 SOC number form
    ("Ascend950PR_x", True),     # A5 pre-release
    ("Ascend950DT_y", True),     # A5 pre-release
    ("Ascend910_93xx", False),   # A3 shares the Ascend910_9 prefix — must NOT match
    ("Ascend910_9381", False),   # A3 SOC number form
    ("Ascend910B1", False),      # A2
    ("Ascend310B1", False),      # 310
    ("", False),                 # empty (no device / rtGetSocVersion failed)
])
def test_is_a5(arch, expected):
    assert is_a5(arch) is expected


def test_triton_support_simt_returns_bool():
    assert isinstance(triton_support_simt(), bool)


# ---------------------------------------------------------------------------
# The make_ttir guard. Needs proton + A5 hardware.
# ---------------------------------------------------------------------------
def _add_with_scope():

    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr):
        idx = tl.arange(0, BLOCK)
        with pl.scope("load_add"):
            x = tl.load(x_ptr + idx)
            y = tl.load(y_ptr + idx)
        tl.store(out_ptr + idx, x + y)

    return add_kernel


@a5_only
@proton_required
@pytest.mark.parametrize("compile_mode", ["simd", "simd_simt", "simt_template"])
def test_proton_rejected_outside_simt_only(tmp_path, compile_mode):
    pl.enable_semantic("triton")
    add_kernel = _add_with_scope()
    x = torch.randn(64, device="npu", dtype=torch.float32)
    y = torch.randn(64, device="npu", dtype=torch.float32)
    out = torch.empty_like(x)

    proton.start(str(tmp_path / "prof"), backend="instrumentation")
    try:
        with pytest.raises(RuntimeError, match="compile_mode='simt_only'"):
            add_kernel[(1, )](x, y, out, BLOCK=64, compile_mode=compile_mode)
    finally:
        proton.finalize()


@a5_only
@proton_required
def test_proton_accepted_in_simt_only(tmp_path):
    pl.enable_semantic("triton")
    add_kernel = _add_with_scope()
    x = torch.randn(64, device="npu", dtype=torch.float32)
    y = torch.randn(64, device="npu", dtype=torch.float32)
    out = torch.empty_like(x)

    proton.start(str(tmp_path / "prof"), backend="instrumentation")
    try:
        # Should compile + run without the guard firing.
        add_kernel[(1, )](x, y, out, BLOCK=64, compile_mode="simt_only")
        torch.testing.assert_close(out, x + y, rtol=1e-5, atol=1e-5)
    finally:
        proton.finalize()
