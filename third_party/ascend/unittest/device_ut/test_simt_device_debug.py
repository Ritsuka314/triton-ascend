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
"""SIMT device debug runtime tests.

Run this module with ``pytest --forked`` so every case starts before the parent
pytest process initializes NPU runtime state.
"""

import ast
import ctypes
import os
import random
import re
from collections import defaultdict

os.environ["TRITON_DEVICE_PRINT"] = "1"
os.environ["TRITON_DEBUG"] = "1"
os.environ["TRITON_ENABLE_TASKQUEUE"] = "0"

import pytest  # noqa: E402
import torch  # noqa: E402
import torch_npu  # noqa: E402, F401
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from triton.backends.ascend.driver import NPUUtils  # noqa: E402

_libc = ctypes.CDLL(None)

# ========== test harness


@pytest.fixture(autouse=True)
def set_random_seed():
    random.seed(42)


def flush_native_stdio():
    _libc.fflush(None)


def capture_kernel_outerr(capfd, launch, debug):
    exception = None
    try:
        launch()
        torch.npu.synchronize()
    except Exception as error:
        exception = error

    flush_native_stdio()
    outerr = capfd.readouterr()

    if debug:
        print(outerr.out)
        print(outerr.err)
        print(exception)

    return outerr, exception


HEADER = "HiIPU Print"

_SEP_RE = re.compile(r"-{3,}\Z")
_RECORD_SPLIT_RE = re.compile(r"(?m)^=>\s*Vec\s+(\d+)\b[^\n]*$")
_ASSERTION_RE = re.compile(r"(?m)^\*\*\* Assertion Failure "
                           r"\(Vec,\s*Block ID\s*=\s*(\d+)\):\s*(.*?)\s*$")
_FIELD_SPLIT_RE = re.compile(r"(?m)^([^\n]*):\s*$")


def normalize_device_print(text):
    return "\n".join(
        line
        for raw in text.splitlines()
        if (line := raw.strip())
        and line != HEADER
        and not _SEP_RE.fullmatch(line)
    )


class _BoolSub(ast.NodeTransformer):

    def visit_Name(self, node):
        if node.id == "T": return ast.Constant(value=True)
        if node.id == "F": return ast.Constant(value=False)
        return node


def parse_default_value(value):
    tree = ast.parse(value.strip(), mode="eval")
    tree = _BoolSub().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.literal_eval(tree)


def parse_hex_ptr(value):
    return int(value, 16)


def parse_record(vec, body, value_parsers):
    assertion_matches = _ASSERTION_RE.findall(body)
    body = _ASSERTION_RE.sub("", body)

    grouped_fields = defaultdict(list)
    field_parts = iter(_FIELD_SPLIT_RE.split(body)[1:])
    for name, value in zip(field_parts, field_parts):
        name = name.strip()
        value = "".join(line.strip() for line in value.splitlines())
        parser = value_parsers.get(name, parse_default_value)
        grouped_fields[name].append(parser(value))

    record = {"vec": int(vec)}
    record.update(
        (name, values[0] if len(values) == 1 else values)
        for name, values in grouped_fields.items()
    )

    if assertion_matches:
        assertions = [{
            "block_id": int(block_id),
            "message": message.strip(),
        } for block_id, message in assertion_matches]
        record["assertion"] = (assertions[0] if len(assertions) == 1 else assertions)

    return record


def parse_device_print_records(text, value_parsers, debug):
    record_parts = iter(
        _RECORD_SPLIT_RE.split(normalize_device_print(text))[1:]
    )
    records = [
        parse_record(vec, body, value_parsers)
        for vec, body in zip(record_parts, record_parts)
    ]

    if debug:
        print(records)

    return records


def round_val(obj, ndigits=6):
    if isinstance(obj, torch.Tensor):
        if obj.is_floating_point():
            return obj.round(ndigits=ndigits)
        return obj
    if isinstance(obj, (list, tuple)):
        return type(obj)(round_val(v, ndigits) for v in obj)
    if isinstance(obj, dict):
        return {k: round_val(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, float):
        return round(obj, ndigits)
    return obj


def check_strict_equal(actual, expected, path="root"):
    if type(actual) is not type(expected):
        return (f"{path}: type mismatch\n"
                f"  actual:   {actual!r} ({type(actual).__name__})\n"
                f"  expected: {expected!r} ({type(expected).__name__})")

    elif isinstance(actual, dict):
        actual_keys = set(actual)
        expected_keys = set(expected)

        if actual_keys != expected_keys:
            return (f"{path}: dictionary keys differ\n"
                    f"  missing: {expected_keys - actual_keys!r}\n"
                    f"  extra:   {actual_keys - expected_keys!r}")

        for key in actual:
            mismatch = check_strict_equal(
                actual[key],
                expected[key],
                f"{path}[{key!r}]",
            )
            if mismatch is not None:
                return mismatch
        return None

    elif isinstance(actual, (list, tuple)):
        if len(actual) != len(expected):
            return (f"{path}: length mismatch\n"
                    f"  actual:   {len(actual)}\n"
                    f"  expected: {len(expected)}")

        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected)
        ):
            mismatch = check_strict_equal(
                actual_item,
                expected_item,
                f"{path}[{index}]",
            )
            if mismatch is not None:
                return mismatch
        return None

    elif actual != expected:
        return (f"{path}: value mismatch\n"
                f"  actual:   {actual!r}\n"
                f"  expected: {expected!r}")

    else:
        return None


def assert_kernel_output(
    capfd,
    launch,
    *,
    value_parsers=None,
    expected=None,
    exception=None,
    debug=False,
):
    outerr, exc = capture_kernel_outerr(capfd, launch, debug)
    records = parse_device_print_records(
        outerr.out, value_parsers or {}, debug
    )

    if exception is None:
        if exc is not None:
            raise exc
    else:
        assert isinstance(exc, exception), (
            f"Expected {exception}, got {type(exc)}"
        )

    if expected is not None:
        mismatch = check_strict_equal(round_val(records), round_val(expected))
        if mismatch:
            raise AssertionError(f"{mismatch}\n\n"
                                 f"full output:\n" + outerr.out + "\n" + outerr.err + "\n" + f"expected:\n"
                                 f"{expected}")

    return outerr, records, exc


SCALARS = [
    (42,        torch.int32,            "int32"),
    (2**31 + 1, torch.uint32,           "uint32"),
    (2**63 + 1, torch.uint64,           "uint64"),
    (True,      torch.bool,             "bool"),
    (0.375,     torch.float16,          "fp16"), # torch half
    (0.375,     torch.float32,          "fp32"),
    # device print does not support
    # (0.375,     torch.float64,          "fp64"), # torch double
    (0.375,     torch.bfloat16,         "bf16"),
    (0.375,     torch.float8_e4m3fn,    "fp8e4m3fn"),
    # device print does not support
    # (0.375,     torch.float8_e4m3fnuz,  "float8e4b8"),
    (0.375,     torch.float8_e5m2,      "fp8e5m2"),
    # device print does not support
    # (0.375,     torch.float8_e5m2fnuz,  "fp8e5b16"),
]

# ========== tests


@triton.jit
def print_kernel_arg_constexpr(x: tl.constexpr):
    tl.device_print("const:", x)


@pytest.mark.parametrize("value", [
    pytest.param(value, id=id,)
    for value, _, id in SCALARS])
def test_print_kernel_arg_constexpr(capfd, value):
    assert_kernel_output(capfd, launch=lambda: print_kernel_arg_constexpr[(1, )](value, compile_mode="simt_only"),
                         expected=[{'vec': 0, 'const': value}])


# ==========


@triton.jit
def print_kernel_hex(x: tl.constexpr):
    tl.device_print("hex:", x, hex=True)


def test_print_kernel_hex(capfd):
    value = 42
    assert_kernel_output(capfd, launch=lambda: print_kernel_hex[(1, )](value, compile_mode="simt_only"), value_parsers={
        "hex": parse_hex_ptr,
    }, expected=[{'vec': 0, 'hex': value}])


# ==========


@triton.jit
def print_kernel_no_arg():
    tl.device_print("print")


@pytest.mark.skip(reason="waiting for TA to support")
def test_print_kernel_no_arg(capfd):
    assert_kernel_output(
        capfd,
        launch=lambda: print_kernel_no_arg[(1, )](compile_mode="simt_only", debug=True),
        expected=[{'vec': 0, 'print': None}],
    )


# ==========


@triton.jit
def print_kernel_arg_scalar(x):
    tl.device_print("scalar:", x)


@pytest.mark.parametrize("value", [pytest.param(value, id=id) for value, _, id in SCALARS])
def test_print_kernel_arg_scalar(capfd, value):
    assert_kernel_output(
        capfd,
        launch=lambda: print_kernel_arg_scalar[(1, )](value, compile_mode="simt_only"),
        expected=[{'vec': 0, 'scalar': value}],
    )


@triton.jit
def assert_kernel_arg_scalar(x):
    tl.device_assert(x, "x is false")
    tl.device_print("x", x)


@pytest.mark.parametrize("value", [True, False])
def test_assert_kernel_arg_scalar(capfd, value):
    if value:
        expected = [{'vec': 0, 'x': True}]
        exception = None
    else:
        expected = [{'vec': 0, 'assertion': {'block_id': 0, 'message': 'x is false'}}]
        exception = RuntimeError

    assert_kernel_output(
        capfd,
        launch=lambda: assert_kernel_arg_scalar[(1,)](
            value, compile_mode="simt_only", debug=True),
        expected=expected,
        exception=exception,
    )


# ==========


@triton.jit
def print_kernel_scalar(x_ptr):
    tl.device_print("scalar:", tl.load(x_ptr))


@pytest.mark.parametrize("value,dtype", [pytest.param(value, dtype, id=id) for value, dtype, id in SCALARS])
def test_print_kernel_scalar(capfd, value, dtype):
    x = torch.tensor([value], dtype=dtype, device="npu")

    assert_kernel_output(
        capfd,
        launch=lambda: print_kernel_scalar[(1, )](x, compile_mode="simt_only"),
        expected=[{"vec": 0, "scalar": value}],
    )


# ==========


@triton.jit
def print_kernel_arg_pointer(x):
    tl.device_print("ptr x:", x)
    tl.device_print("ptr x+1:", x + 1)


def test_print_kernel_arg_pointer(capfd):
    n = 4
    x = torch.arange(n, dtype=torch.int32, device="npu")

    assert_kernel_output(capfd, launch=lambda: print_kernel_arg_pointer[(1, )](x, compile_mode="simt_only"),
                         value_parsers={
                             "ptr x": parse_hex_ptr,
                             "ptr x+1": parse_hex_ptr,
                         }, expected=[{'vec': 0, 'ptr x': x.data_ptr(), 'ptr x+1': x[1:].data_ptr()}])


# ==========


@triton.jit
def print_pid():
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    tl.device_print("pid_m:", pid_m)
    tl.device_print("pid_n:", pid_n)


def test_print_pid(capfd):
    grid = (2, 2)

    assert_kernel_output(
        capfd, launch=lambda: print_pid[grid](compile_mode="simt_only"),
        expected=[{'vec': 0, 'pid_n': 0, 'pid_m': 0}, {'vec': 1, 'pid_n': 0, 'pid_m': 1},
                  {'vec': 2, 'pid_n': 1, 'pid_m': 0}, {'vec': 3, 'pid_n': 1, 'pid_m': 1}])


# ==========


@triton.jit
def print_kernel_blockify():
    pid = tl.program_id(0)
    tl.device_print("p", pid)


def test_print_kernel_blockify(capfd):
    num_physical_blocks = NPUUtils().get_aivector_core_num()

    grid = (num_physical_blocks * 2, )

    assert_kernel_output(capfd, launch=lambda: print_kernel_blockify[grid]
                         (compile_mode="simt_only", enable_auto_blockify=1),
                         expected=[{'vec': i, 'p': [i * 2, i * 2 + 1]} for i in range(num_physical_blocks)])


# ==========


@triton.jit
def print_second_superblock_task():
    pid = tl.program_id(0)
    if pid == 1:
        tl.device_print("logical_pid:", pid)


def test_device_print_runs_for_second_superblock_task(capfd, monkeypatch):
    monkeypatch.setenv("TRITON_ALL_BLOCKS_PARALLEL", "1")

    physical_blocks = NPUUtils().get_aivector_core_num()
    logical_blocks = physical_blocks * 2

    # The driver launches physical_blocks blocks. Auto-blockify maps two
    # logical tasks onto each block when the superblock factor is two, so PID 1
    # is led by physical thread 32 in the first block.
    outerr, records, _ = assert_kernel_output(
        capfd,
        launch=lambda: print_second_superblock_task[(logical_blocks, )](
            compile_mode="simt_only",
            enable_auto_blockify=True,
            superblock_factor=2,
            num_warps=1,
        ),
    )
    printed_pids = sorted(record["logical_pid"] for record in records if "logical_pid" in record)

    assert printed_pids == [1], ("the second logical task in an F=2 superblock must execute device_print exactly once; "
                                 f"expected PID [1], got {printed_pids}\n"
                                 f"parsed records:\n{records}\n"
                                 f"stdout:\n{outerr.out}\n"
                                 f"stderr:\n{outerr.err}")


# ==========


@triton.jit
def print_kernel_1d(x_ptr, M, BLOCK_M: tl.constexpr):
    pid_m = tl.program_id(0)
    idx = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = idx < M
    x = tl.load(x_ptr + idx, mask=mask)

    tl.device_print("pid_m:", pid_m)
    tl.device_print("idx:", idx)
    tl.device_print("values:", x)


def expected_arange_1d_records(x, block_m, grid):
    records = []
    for pid_m in range(grid[0]):
        values = x[pid_m * block_m:(pid_m + 1) * block_m]
        records.append({
            "vec": pid_m,
            "pid_m": pid_m,
            "idx": list(range(pid_m * block_m, (pid_m + 1) * block_m)),
            "values": values.tolist() if hasattr(values, 'tolist') else values,
        })
    return records


@pytest.mark.parametrize("dtype,values", [
    (torch.int32, lambda n: torch.randn(n, dtype=torch.int32, device="npu")),
    (torch.float32, lambda n: torch.randn(n, dtype=torch.float32, device="npu")),
    (torch.bool, lambda n: torch.randn(n, device="npu") < 0),
], ids=["int32", "float32", "bool"])
def test_print_kernel_arg_tensor_1d(capfd, dtype, values):
    M = 8
    x = values(M)

    print(x)

    BLOCK_M = 2
    grid = (triton.cdiv(M, BLOCK_M), )  # (4,)

    assert_kernel_output(
        capfd,
        launch=lambda: print_kernel_1d[grid](
            x,
            M,
            BLOCK_M=BLOCK_M,
            compile_mode="simt_only",
        ),
        expected=expected_arange_1d_records(
            x,
            BLOCK_M,
            grid,
        ),
    )


# ==========


@triton.jit
def print_kernel_2d(x_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    row = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]  # [BM, 1]
    col = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]  # [1, BN]
    offsets = row * N + col  # [BM, BN], row-major
    mask = (row < M) & (col < N)
    x = tl.load(x_ptr + offsets, mask=mask)

    tl.device_print("pid_m:", pid_m)
    tl.device_print("pid_n:", pid_n)
    tl.device_print("row:", row)
    tl.device_print("col:", col)
    tl.device_print("idx:", offsets)
    tl.device_print("values:", x)


def expected_arange_2d_records(x, block_m, block_n, grid, n_cols):
    records = []
    for pid_n in range(grid[1]):
        for pid_m in range(grid[0]):
            row_start = pid_m * block_m
            col_start = pid_n * block_n
            values = x[row_start:row_start + block_m, col_start:col_start + block_n]

            vec = pid_m + grid[0] * pid_n
            row_list = [[r] for r in range(row_start, row_start + block_m)]
            col_list = [list(range(col_start, col_start + block_n))]
            offsets = [[r * n_cols + c for c in col_list[0]] for r in [row[0] for row in row_list]]

            records.append({
                "vec": vec,
                "pid_m": pid_m,
                "pid_n": pid_n,
                "row": row_list,
                "col": col_list,
                "idx": offsets,
                "values": values.tolist(),
            })
    return records


@pytest.mark.parametrize("dtype,values", [
    (torch.int32, lambda m, n: torch.randn(m * n, dtype=torch.int32, device="npu")),
    (torch.float32, lambda m, n: torch.randn(m * n, dtype=torch.float32, device="npu")),
    (torch.bool, lambda m, n: torch.randn(m * n, device="npu") < 0),
], ids=["int32", "float32", "bool"])
def test_print_kernel_arg_tensor_2d(capfd, dtype, values):
    M, N = 4, 8
    x = values(M, N).reshape(M, N)

    BLOCK_M, BLOCK_N = 2, 4
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))  # (2, 2)

    assert_kernel_output(
        capfd,
        launch=lambda: print_kernel_2d[grid](x, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, compile_mode="simt_only"),
        expected=expected_arange_2d_records(
            x,
            BLOCK_M,
            BLOCK_N,
            grid,
            N,
        ),
    )


# ==========

# @triton.jit
# def print_kernel_2d_mix(x_ptr, y_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
#     pid_m = tl.program_id(0)
#     pid_n = tl.program_id(1)

#     with al.scope(vec_mode="simd"):
#         row = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]   # [BM, 1]
#         col = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]   # [1, BN]
#         offsets = row * N + col                                  # [BM, BN], row-major
#         mask = (row < M) & (col < N)
#         x = tl.load(x_ptr + offsets, mask=mask)

#         tl.device_print("== SIMD ==", 1)
#         tl.device_print("pid_m:", pid_m)
#         tl.device_print("pid_n:", pid_n)
#         tl.device_print("row:", row)
#         tl.device_print("col:", col)
#         tl.device_print("values:", x)

#     with al.scope(vec_mode="simt"):
#         row = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]   # [BM, 1]
#         col = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]   # [1, BN]
#         offsets = row * N + col                                  # [BM, BN], row-major
#         mask = (row < M) & (col < N)
#         y = tl.load(y_ptr + offsets, mask=mask)

#         tl.device_print("== SIMT ==", 2)
#         tl.device_print("pid_m:", pid_m)
#         tl.device_print("pid_n:", pid_n)
#         tl.device_print("row:", row)
#         tl.device_print("col:", col)
#         tl.device_print("values:", y)

# M, N = 4, 8
# x = torch.arange(M * N, dtype=torch.int64, device="npu").reshape(M, N)
# y = torch.arange(M * N, dtype=torch.int64, device="npu").reshape(M, N) * 2

# BLOCK_M, BLOCK_N = 2, 4
# grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))  # (2, 2)

# print_kernel_2d_mix[grid](x, y, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, compile_mode="simd_simt", debug=True)
