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

INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1
UINT32_MAX = 2**32 - 1

SIGNED_DIVISORS = (
    INT32_MIN,
    -1000,
    -128,
    -7,
    -3,
    -2,
    -1,
    1,
    2,
    3,
    7,
    128,
    1000,
    INT32_MAX,
)

SIGNED_BASE_VALUES = (
    INT32_MIN,
    -2109734912,
    -1001,
    -1000,
    -999,
    -1,
    0,
    1,
    999,
    1000,
    1001,
    INT32_MAX,
)

UNSIGNED_DIVISORS = (
    1,
    2,
    3,
    7,
    128,
    1000,
    0x7FFFFFFF,
    0x80000000,
    0x80000001,
    0xFFFFFFFE,
    0xFFFFFFFF,
)

UNSIGNED_BASE_VALUES = (
    0,
    1,
    2,
    0x7FFFFFFF,
    0x80000000,
    0x82400000,
    0xFFFFFFFE,
    0xFFFFFFFF,
)


def trunc_div(value, divisor):
    quotient = abs(value) // abs(divisor)
    return -quotient if (value < 0) != (divisor < 0) else quotient


def signed_remainder(value, divisor):
    return value - trunc_div(value, divisor) * divisor


def to_int32_bits(value):
    value &= UINT32_MAX
    return value if value < 0x80000000 else value - 0x100000000


def signed_values(divisor):
    values = set(SIGNED_BASE_VALUES)
    magnitude = abs(divisor)
    values.update(value for value in (
        -magnitude - 1,
        -magnitude,
        -magnitude + 1,
        magnitude - 1,
        magnitude,
        magnitude + 1,
    ) if INT32_MIN <= value <= INT32_MAX)
    return tuple(sorted(values))


def unsigned_values(divisor):
    values = set(UNSIGNED_BASE_VALUES)
    values.update(value for value in (divisor - 1, divisor, divisor + 1) if 0 <= value <= UINT32_MAX)
    return tuple(sorted(values))


@triton.jit
def simt_div_kernel(x_ptr, output_ptr, DIVISOR: tl.constexpr):
    pid = tl.program_id(0)
    x = tl.load(x_ptr + pid)
    divisor = tl.full((), DIVISOR, tl.int32)
    tl.store(output_ptr + pid, x // divisor)


@triton.jit
def simt_mod_kernel(x_ptr, output_ptr, DIVISOR: tl.constexpr):
    pid = tl.program_id(0)
    x = tl.load(x_ptr + pid)
    divisor = tl.full((), DIVISOR, tl.int32)
    tl.store(output_ptr + pid, x % divisor)


@triton.jit
def simt_udiv_kernel(x_ptr, output_ptr, DIVISOR: tl.constexpr):
    pid = tl.program_id(0)
    x = tl.load(x_ptr + pid).to(tl.uint32)
    divisor = tl.full((), DIVISOR, tl.uint32)
    result = x // divisor
    tl.store(output_ptr + pid, result.to(tl.int32))


@triton.jit
def simt_umod_kernel(x_ptr, output_ptr, DIVISOR: tl.constexpr):
    pid = tl.program_id(0)
    x = tl.load(x_ptr + pid).to(tl.uint32)
    divisor = tl.full((), DIVISOR, tl.uint32)
    result = x % divisor
    tl.store(output_ptr + pid, result.to(tl.int32))


@pytest.mark.parametrize("divisor", SIGNED_DIVISORS, ids=lambda divisor: f"divisor_{divisor}")
def test_simt_int32_div(divisor):
    values = signed_values(divisor)
    if divisor == -1:
        values = tuple(value for value in values if value != INT32_MIN)
    x_cpu = torch.tensor(values, dtype=torch.int32)
    expected = torch.tensor([trunc_div(value, divisor) for value in values], dtype=torch.int32)
    x = x_cpu.to("npu")
    actual = torch.empty_like(x)

    simt_div_kernel[(x.numel(), )](x, actual, divisor, compile_mode="simt_only")

    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("divisor", SIGNED_DIVISORS, ids=lambda divisor: f"divisor_{divisor}")
def test_simt_int32_mod(divisor):
    values = signed_values(divisor)
    x_cpu = torch.tensor(values, dtype=torch.int32)
    expected = torch.tensor([signed_remainder(value, divisor) for value in values], dtype=torch.int32)
    x = x_cpu.to("npu")
    actual = torch.empty_like(x)

    simt_mod_kernel[(x.numel(), )](x, actual, divisor, compile_mode="simt_only")

    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("divisor", UNSIGNED_DIVISORS, ids=lambda divisor: f"divisor_{divisor:#x}")
def test_simt_uint32_div(divisor):
    values = unsigned_values(divisor)
    x_cpu = torch.tensor([to_int32_bits(value) for value in values], dtype=torch.int32)
    expected = torch.tensor([to_int32_bits(value // divisor) for value in values], dtype=torch.int32)
    x = x_cpu.to("npu")
    actual = torch.empty_like(x)

    simt_udiv_kernel[(x.numel(), )](x, actual, divisor, compile_mode="simt_only")

    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("divisor", UNSIGNED_DIVISORS, ids=lambda divisor: f"divisor_{divisor:#x}")
def test_simt_uint32_mod(divisor):
    values = unsigned_values(divisor)
    x_cpu = torch.tensor([to_int32_bits(value) for value in values], dtype=torch.int32)
    expected = torch.tensor([to_int32_bits(value % divisor) for value in values], dtype=torch.int32)
    x = x_cpu.to("npu")
    actual = torch.empty_like(x)

    simt_umod_kernel[(x.numel(), )](x, actual, divisor, compile_mode="simt_only")

    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def test_unsigned_oracle_mutation_killers():
    assert 0xFFFFFFFE // 3 == 1431655764
    assert 0xFFFFFFFE % 3 == 2
    assert 0xFFFFFFFE // 1000 == 4294967
    assert 0xFFFFFFFE % 1000 == 294
