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
"""Unit tests for the SIMT stack-limit resolver in triton.backends.ascend.utils.

Resolution priority: per-kernel `simt_stack_limit` > TRITON_SIMT_STACK_LIMIT env
var > built-in default 1152.
"""

import pytest

from triton.backends.ascend.utils import (
    DEFAULT_SIMT_STACK_LIMIT,
    _get_simt_stack_limit,
)


def test_default_when_neither_set(monkeypatch):
    monkeypatch.delenv("TRITON_SIMT_STACK_LIMIT", raising=False)
    assert _get_simt_stack_limit(None) == DEFAULT_SIMT_STACK_LIMIT
    assert DEFAULT_SIMT_STACK_LIMIT == 1152


@pytest.mark.parametrize(
    "env_value, expected",
    [
        ("8192", 8192),
        ("0x800", 2048),
        ("0X400", 1024),
        ("0", 0),
        ("-1", -1),
        ("  4096  ", 4096),
    ],
)
def test_env_var_parsed(monkeypatch, env_value, expected):
    monkeypatch.setenv("TRITON_SIMT_STACK_LIMIT", env_value)
    assert _get_simt_stack_limit(None) == expected


@pytest.mark.parametrize(
    "per_kernel",
    [0, 1, 1024, 8192, -1, -999],
)
def test_per_kernel_wins_over_env(monkeypatch, per_kernel):
    monkeypatch.setenv("TRITON_SIMT_STACK_LIMIT", "9999")
    assert _get_simt_stack_limit(per_kernel) == per_kernel


def test_per_kernel_wins_over_default(monkeypatch):
    monkeypatch.delenv("TRITON_SIMT_STACK_LIMIT", raising=False)
    assert _get_simt_stack_limit(4096) == 4096
    assert _get_simt_stack_limit(0) == 0  # 0 is not None — it wins


def test_invalid_env_raises(monkeypatch):
    monkeypatch.setenv("TRITON_SIMT_STACK_LIMIT", "not-a-number")
    with pytest.raises(ValueError, match="TRITON_SIMT_STACK_LIMIT"):
        _get_simt_stack_limit(None)


def test_invalid_env_ignored_when_per_kernel_set(monkeypatch):
    """Per-kernel value short-circuits the env-var parse — invalid env
    must not raise when the user provided a per-kernel value."""
    monkeypatch.setenv("TRITON_SIMT_STACK_LIMIT", "garbage")
    assert _get_simt_stack_limit(2048) == 2048
