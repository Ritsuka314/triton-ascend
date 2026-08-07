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

import pytest
import torch
import torch_npu  # noqa: F401 -- registers the NPU device with torch
import triton
import triton.language as tl
from triton.backends.ascend.utils import is_compile_on_910_95

pytestmark = pytest.mark.skipif(not is_compile_on_910_95(), reason="SIMT-only compilation requires A5")

INT32_INPUTS = (
    -2147483648,
    -2147483647,
    -2109734912,  # Regression: this value produced -100662384 for `% 1000`.
    # Fast-div carry boundaries for divisors 3 and 1000.
    -1073741825,
    -1073741824,
    -1073741823,
    -100663297,
    -100663296,
    -100663295,
    -2001,
    -2000,
    -1999,
    -1001,
    -1000,
    -999,
    -257,
    -256,
    -255,
    -129,
    -128,
    -127,
    -7,
    -6,
    -5,
    -4,
    -3,
    -2,
    -1,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    127,
    128,
    129,
    255,
    256,
    257,
    999,
    1000,
    1001,
    1999,
    2000,
    2001,
    1073741823,
    1073741824,
    # Fast-div signed-shift boundaries for divisors 3 and 1000.
    1610612735,
    1610612736,
    1610612737,
    2097151999,
    2097152000,
    2097152001,
    2147483646,
    2147483647,
)

# Zero is invalid, and -1 is omitted because the inputs include INT32_MIN.
DIVISORS = (
    pytest.param(3, id="divisor_3"),
    pytest.param(128, id="divisor_128"),
    pytest.param(1000, id="divisor_1000"),
    pytest.param(2147483647, id="divisor_int32_max"),
    pytest.param(-3, id="divisor_neg_3"),
    pytest.param(-128, id="divisor_neg_128"),
    pytest.param(-1000, id="divisor_neg_1000"),
)

BLOCK_SIZE = 64


@triton.jit
def simt_div_kernel(x_ptr, divisor: tl.constexpr, output_ptr, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    output = x // divisor
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def simt_mod_kernel(x_ptr, divisor: tl.constexpr, output_ptr, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    output = x % divisor
    tl.store(output_ptr + offsets, output, mask=mask)


@pytest.mark.parametrize("divisor", DIVISORS)
def test_simt_int32_div(divisor):
    x_cpu = torch.tensor(INT32_INPUTS, dtype=torch.int32)
    x = x_cpu.to("npu")
    actual = torch.empty_like(x)

    simt_div_kernel[(1, )](x, divisor, actual, len(INT32_INPUTS), BLOCK_SIZE, compile_mode="simt_only")

    expected = torch.div(x_cpu, divisor, rounding_mode="trunc")
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("divisor", DIVISORS)
def test_simt_int32_mod(divisor):
    x_cpu = torch.tensor(INT32_INPUTS, dtype=torch.int32)
    x = x_cpu.to("npu")
    actual = torch.empty_like(x)

    simt_mod_kernel[(1, )](x, divisor, actual, len(INT32_INPUTS), BLOCK_SIZE, compile_mode="simt_only")

    quotient = torch.div(x_cpu, divisor, rounding_mode="trunc")
    expected = x_cpu - quotient * divisor
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
