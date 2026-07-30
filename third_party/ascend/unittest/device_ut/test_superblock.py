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

import multiprocessing
import os
import queue
import signal
import time
import traceback

import pytest
import torch
import torch_npu
import triton
import triton.language as tl
from triton import knobs
from triton.backends.ascend.driver import NPUUtils

_COMPILE_TIMEOUT_SECONDS = 300
_RUNTIME_TIMEOUT_SECONDS = 60
_CHILD_EXIT_TIMEOUT_SECONDS = 30
_CHILD_STOP_TIMEOUT_SECONDS = 10
_SENTINEL = -777777


@triton.jit
def count_launches_without_program_id(counter):
    lanes = tl.arange(0, 32)
    tl.atomic_add(counter, 1, mask=lanes == 0)


@triton.jit
def task_local_barrier_tail(scratch, out, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    base = pid * BLOCK
    linear = base + offsets

    tl.store(scratch + linear, linear + 1)
    tl.debug_barrier()
    mirrored = tl.load(scratch + base + (BLOCK - 1 - offsets))

    # Different logical tasks deliberately execute different barrier counts.
    # A physical-block barrier can pair the even task's conditional barrier
    # with the odd task's common barrier and then deadlock the even task.
    if (pid % 2) == 0:
        tl.debug_barrier()
    tl.debug_barrier()
    tl.debug_barrier()

    tl.store(out + linear, mirrored)


@triton.jit
def per_task_cross_warp_reduction(out, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    values = (pid + 1) * 1000 + offsets
    total = tl.sum(values, axis=0)
    tl.store(out + pid, total)


@triton.jit
def contended_repeated_barrier(scratch, out, BLOCK: tl.constexpr, STEPS: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    base = pid * BLOCK
    linear = base + offsets
    value = linear

    for _ in range(STEPS):
        tl.store(scratch + linear, value)
        tl.debug_barrier()
        value = tl.load(scratch + base + ((offsets + 1) % BLOCK))
        tl.debug_barrier()

    tl.store(out + linear, value)


@triton.jit
def one_warp_fence_boundary(out):
    pid = tl.program_id(0)
    lanes = tl.arange(0, 16)
    tl.debug_barrier()
    tl.store(out + pid, pid + 1, mask=lanes == 0)


def _compile_and_launch(
    kernel,
    grid,
    args,
    options,
    result_queue,
    *,
    expected_auto_blockified,
    expected_block_count,
):
    compiled = kernel.warmup(*args, grid=grid, **options)
    metadata = compiled.metadata
    assert metadata.auto_blockified is expected_auto_blockified
    assert metadata.auto_blockify_block_count == expected_block_count
    assert metadata.superblock_factor == options["superblock_factor"]
    assert metadata.num_warps == options["num_warps"]
    assert metadata.warp_size == options["warp_size"]

    # Build the launcher and load the binary under the compilation/setup
    # timeout, before the device-execution deadline begins.
    _ = compiled.run
    result_queue.put(("compiled", ""))
    kernel[grid](*args, **options)
    torch_npu.npu.synchronize()


def _run_feedback_case(result_queue):
    os.environ["TRITON_ALL_BLOCKS_PARALLEL"] = "1"
    physical_blocks = NPUUtils().get_aivector_core_num()
    assert physical_blocks > 0
    logical_blocks = physical_blocks + 3
    counter = torch.zeros((1,), dtype=torch.int32, device="npu")
    options = {
        "compile_mode": "simt_only",
        "enable_auto_blockify": True,
        "superblock_factor": 1,
        "num_warps": 1,
        "warp_size": 32,
        # The atomic gives this no-PID kernel an observable per-launch side
        # effect. Override Triton-Ascend's conservative blacklist so the
        # compiler's negative AutoBlockify feedback is exercised.
        "has_auto_blockify_blacklist_op": False,
    }

    _compile_and_launch(
        count_launches_without_program_id,
        (logical_blocks,),
        (counter,),
        options,
        result_queue,
        expected_auto_blockified=False,
        expected_block_count=0,
    )

    actual = counter.cpu().item()
    assert actual == logical_blocks, (
        "AutoBlockify was not applied, so launch feedback must preserve the "
        f"logical launch count; expected {logical_blocks}, got {actual}"
    )


def _run_tail_barrier_case(result_queue):
    factor = 2
    num_warps = 2
    block = 64
    physical_blocks = NPUUtils().get_aivector_core_num()
    assert physical_blocks > 0
    logical_blocks = physical_blocks * factor - 1
    num_elements = logical_blocks * block
    padded_elements = (logical_blocks + 1) * block
    scratch = torch.full((padded_elements,), _SENTINEL, dtype=torch.int32, device="npu")
    out = torch.full_like(scratch, _SENTINEL)
    options = {
        "compile_mode": "simt_only",
        # A factor greater than one must override an explicit false value.
        "enable_auto_blockify": False,
        "superblock_factor": factor,
        "num_warps": num_warps,
        "warp_size": 32,
    }

    _compile_and_launch(
        task_local_barrier_tail,
        (logical_blocks,),
        (scratch, out, block),
        options,
        result_queue,
        expected_auto_blockified=True,
        expected_block_count=physical_blocks,
    )

    expected = torch.arange(1, num_elements + 1, dtype=torch.int32).reshape(logical_blocks, block).flip(1)
    actual = out.cpu()
    torch.testing.assert_close(actual[:num_elements].reshape(logical_blocks, block), expected)
    torch.testing.assert_close(
        actual[num_elements:],
        torch.full((block,), _SENTINEL, dtype=torch.int32),
    )
    torch.testing.assert_close(
        scratch.cpu()[num_elements:],
        torch.full((block,), _SENTINEL, dtype=torch.int32),
    )


def _run_product_64_case(result_queue):
    factor = 16
    num_warps = 4
    block = 128
    steps = 4
    physical_blocks = NPUUtils().get_aivector_core_num()
    assert physical_blocks > 0
    logical_blocks = physical_blocks * factor
    num_elements = logical_blocks * block
    scratch = torch.empty((num_elements,), dtype=torch.int32, device="npu")
    out = torch.empty_like(scratch)
    options = {
        "compile_mode": "simt_only",
        "enable_auto_blockify": False,
        "superblock_factor": factor,
        "num_warps": num_warps,
        "warp_size": 32,
    }

    _compile_and_launch(
        contended_repeated_barrier,
        (logical_blocks,),
        (scratch, out, block, steps),
        options,
        result_queue,
        expected_auto_blockified=True,
        expected_block_count=physical_blocks,
    )

    expected = torch.arange(num_elements, dtype=torch.int32).reshape(logical_blocks, block).roll(-steps, dims=1)
    torch.testing.assert_close(out.cpu().reshape(logical_blocks, block), expected)


def _run_shared_partition_case(result_queue):
    factor = 2
    num_warps = 2
    block = 64
    physical_blocks = NPUUtils().get_aivector_core_num()
    assert physical_blocks > 0
    logical_blocks = physical_blocks * factor
    out = torch.empty((logical_blocks,), dtype=torch.int32, device="npu")
    options = {
        "compile_mode": "simt_only",
        "enable_auto_blockify": False,
        "superblock_factor": factor,
        "num_warps": num_warps,
        "warp_size": 32,
    }

    _compile_and_launch(
        per_task_cross_warp_reduction,
        (logical_blocks,),
        (out, block),
        options,
        result_queue,
        expected_auto_blockified=True,
        expected_block_count=physical_blocks,
    )

    pids = torch.arange(1, logical_blocks + 1, dtype=torch.int32)
    expected = pids * (1000 * block) + block * (block - 1) // 2
    torch.testing.assert_close(out.cpu(), expected)


def _run_one_warp_2048_case(result_queue):
    factor = 128
    physical_blocks = NPUUtils().get_aivector_core_num()
    assert physical_blocks > 0
    out = torch.empty((factor,), dtype=torch.int32, device="npu")
    options = {
        "compile_mode": "simt_only",
        "enable_auto_blockify": False,
        "superblock_factor": factor,
        "num_warps": 1,
        "warp_size": 16,
    }

    _compile_and_launch(
        one_warp_fence_boundary,
        (factor,),
        (out,),
        options,
        result_queue,
        expected_auto_blockified=True,
        expected_block_count=physical_blocks,
    )

    expected = torch.arange(1, factor + 1, dtype=torch.int32)
    torch.testing.assert_close(out.cpu(), expected)


_CASES = {
    "feedback-no-program-id": _run_feedback_case,
    "task-local-barrier-tail": _run_tail_barrier_case,
    "shared-partition": _run_shared_partition_case,
    "product-64-yield": _run_product_64_case,
    "one-warp-2048": _run_one_warp_2048_case,
}


def _child_main(case_name, device_id, cache_dir, result_queue):
    try:
        os.setsid()
        result_queue.put(("session-started", ""))
        os.environ["TRITON_ALL_BLOCKS_PARALLEL"] = "0"
        os.environ["TRITON_DEVICE_PRINT"] = "0"
        os.environ["TRITON_DEBUG"] = "0"
        os.environ["TRITON_ENABLE_TASKQUEUE"] = "0"
        os.environ["TRITON_ALWAYS_COMPILE"] = "0"
        os.environ["TRITON_CACHE_DIR"] = cache_dir
        knobs.cache.dir = cache_dir
        knobs.compilation.always_compile = False
        knobs.runtime.debug = False
        torch.npu.set_device(device_id)
        _CASES[case_name](result_queue)
    except BaseException:
        result_queue.put(("error", traceback.format_exc()))
    else:
        result_queue.put(("ok", ""))


def _stop_process(process, terminate_group=False):
    if not process.is_alive():
        process.join(timeout=0)
        return True

    if terminate_group:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    else:
        process.terminate()
    process.join(timeout=_CHILD_STOP_TIMEOUT_SECONDS)
    if process.is_alive():
        if terminate_group:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        process.join(timeout=_CHILD_STOP_TIMEOUT_SECONDS)
    return not process.is_alive()


def _start_with_sanitized_environment(process, cache_dir):
    child_environment = {
        "TRITON_ALL_BLOCKS_PARALLEL": "0",
        "TRITON_ALWAYS_COMPILE": "0",
        "TRITON_CACHE_DIR": cache_dir,
        "TRITON_COMPILE_ONLY": "0",
        "TRITON_DEBUG": "0",
        "TRITON_DEVICE_PRINT": "0",
        "TRITON_ENABLE_TASKQUEUE": "0",
        "TRITON_INTERPRET": "0",
        "TRITON_KERNEL_OVERRIDE": "0",
    }
    previous = {name: os.environ.get(name) for name in child_environment}
    try:
        os.environ.update(child_environment)
        process.start()
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _run_case_with_timeout(case_name, tmp_path):
    device_id = torch.npu.current_device()
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    cache_dir = str(tmp_path / f"triton-cache-{case_name}")
    process = context.Process(
        target=_child_main,
        args=(case_name, device_id, cache_dir, result_queue),
    )
    _start_with_sanitized_environment(process, cache_dir)

    child_session_started = False
    try:
        phase = "compilation and runtime setup"
        deadline = time.monotonic() + _COMPILE_TIMEOUT_SECONDS
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                stopped = _stop_process(process, terminate_group=child_session_started)
                suffix = "" if stopped else "; child could not be stopped"
                pytest.fail(f"{case_name} timed out during {phase}{suffix}")

            try:
                status, payload = result_queue.get(timeout=min(0.5, remaining))
            except queue.Empty:
                if process.is_alive():
                    continue
                process.join(timeout=0)
                pytest.fail(f"{case_name} exited with code {process.exitcode} without reporting a result")

            if status == "session-started":
                child_session_started = True
                continue

            if status == "compiled":
                phase = "device execution"
                deadline = time.monotonic() + _RUNTIME_TIMEOUT_SECONDS
                continue

            process.join(timeout=_CHILD_EXIT_TIMEOUT_SECONDS)
            child_still_alive = process.is_alive()
            if child_still_alive:
                _stop_process(process, terminate_group=child_session_started)

            if status == "error":
                pytest.fail(payload)
            assert status == "ok", f"unexpected child status: {status}"
            assert not child_still_alive, f"{case_name} reported success but did not exit"
            assert process.exitcode == 0
            return
    finally:
        _stop_process(process, terminate_group=child_session_started)
        result_queue.cancel_join_thread()
        result_queue.close()
        if not process.is_alive():
            process.close()


def test_auto_blockify_feedback_preserves_grid_without_program_id(tmp_path):
    _run_case_with_timeout("feedback-no-program-id", tmp_path)


def test_superblock_factor_forces_task_local_barrier_for_tail(tmp_path):
    _run_case_with_timeout("task-local-barrier-tail", tmp_path)


def test_superblock_partitions_cross_warp_reduction_state(tmp_path):
    _run_case_with_timeout("shared-partition", tmp_path)


def test_one_warp_superblock_allows_2048_threads(tmp_path):
    _run_case_with_timeout("one-warp-2048", tmp_path)


def test_soft_barrier_product_64_yields_and_completes(tmp_path):
    # Keep the highest-contention case last: a device-side deadlock can poison
    # the device context even after the host child process is terminated.
    _run_case_with_timeout("product-64-yield", tmp_path)
