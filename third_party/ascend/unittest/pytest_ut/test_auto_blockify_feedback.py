import importlib
import importlib.util
import json
import os
import subprocess
import sys
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

_ASCEND_PACKAGE_NAME = "triton.backends.ascend"
_ASCEND_DEPENDENCIES = (
    (f"{_ASCEND_PACKAGE_NAME}.backend_register", "backend_register", "backend_register.py"),
    (f"{_ASCEND_PACKAGE_NAME}.utils", "utils", "utils.py"),
    (f"{_ASCEND_PACKAGE_NAME}.driver", "driver", "driver.py"),
)
_COMPILER_MODULE_NAME = "ascend_compiler_auto_blockify_under_test"
_CHILD_MODE = "--auto-blockify-feedback-child"


def _ascend_module_state():
    return {
        name: module
        for name, module in sys.modules.items()
        if name == _ASCEND_PACKAGE_NAME or name.startswith(f"{_ASCEND_PACKAGE_NAME}.")
    }


def _load_compiler_module():
    backend_dir = Path(__file__).resolve().parents[2] / "backend"
    importlib.import_module("triton")
    ascend_package = importlib.import_module(_ASCEND_PACKAGE_NAME)

    for name, attribute, filename in _ASCEND_DEPENDENCIES:
        dependency_spec = importlib.util.spec_from_file_location(name, backend_dir / filename)
        dependency = importlib.util.module_from_spec(dependency_spec)
        assert dependency_spec.loader is not None
        sys.modules[name] = dependency
        setattr(ascend_package, attribute, dependency)
        dependency_spec.loader.exec_module(dependency)

    spec = importlib.util.spec_from_file_location(_COMPILER_MODULE_NAME, backend_dir / "compiler.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeModule:

    def __str__(self):
        return "module {}"


def _make_fake_run(payload, seen_command):

    def fake_run(command, **kwargs):
        seen_command.extend(command)
        output_base = Path(command[command.index("-o") + 1])
        output_base.with_suffix(".o").write_bytes(b"npubin")
        metadata_arg = next(arg for arg in command if arg.startswith("--triton-metadata-output="))
        metadata_path = Path(metadata_arg.split("=", 1)[1])
        metadata_path.write_text(json.dumps(payload))
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    return fake_run


def _compile(compiler, options, payload=None, has_blacklist=False):
    seen_command = []
    compiler.subprocess.run = _make_fake_run({} if payload is None else payload, seen_command)
    metadata = {
        "bisheng_options": None,
        "has_auto_blockify_blacklist_op": has_blacklist,
    }
    result = compiler.ttir_to_npubin(_FakeModule(), metadata, options)
    return result, metadata, seen_command


def _run_auto_blockify_feedback_child():
    compiler = _load_compiler_module()
    compiler._parse_ttir_metadata = lambda code, metadata: metadata
    compiler.get_common_bishengir_compile_options = lambda metadata: []
    compiler._get_npucompiler_path = lambda: ("bishengir-compile", {})

    class UnlimitedNPUUtils:

        def has_device_limit(self):
            return False

    compiler.NPUUtils = UnlimitedNPUUtils
    os.environ.pop("NPU_DEVICE_LIMIT", None)
    compiler._is_auto_map_parallel_blocks_enabled = lambda: True
    compiler.get_cann_version_file_hash = lambda: "fixed-cann"
    default_options = compiler.NPUOptions(compile_mode="simt_only")
    assert default_options.enable_auto_blockify is True
    assert compiler.NPUOptions(compile_mode="simd").enable_auto_blockify is None
    result, metadata, command = _compile(
        compiler,
        default_options,
        {"auto_blockified": True, "auto_blockify_block_count": 64},
    )
    assert result == b"npubin"
    assert command.count("--enable-auto-blockify-loop") == 1
    assert metadata["auto_blockified"] is True
    assert metadata["auto_blockify_block_count"] == 64

    disabled_options = compiler.NPUOptions(
        compile_mode="simt_only",
        enable_auto_blockify=False,
        superblock_factor=1,
    )
    assert disabled_options.enable_auto_blockify is False
    _, metadata, command = _compile(compiler, disabled_options)
    assert "--enable-auto-blockify-loop" not in command
    assert not any(arg.startswith("--super-block-factor=") for arg in command)
    assert metadata["auto_blockified"] is False
    assert metadata["auto_blockify_block_count"] == 0

    blacklisted_options = compiler.NPUOptions(
        compile_mode="simt_only",
        enable_auto_blockify=True,
    )
    _, _, command = _compile(compiler, blacklisted_options, has_blacklist=True)
    assert "--enable-auto-blockify-loop" not in command

    compiler._is_auto_map_parallel_blocks_enabled = lambda: False
    env_disabled_options = compiler.NPUOptions(compile_mode="simt_only")
    assert env_disabled_options.enable_auto_blockify is False
    assert env_disabled_options.hash() != default_options.hash()
    explicit_options = compiler.NPUOptions(
        compile_mode="simt_only",
        enable_auto_blockify=True,
    )
    _, _, command = _compile(
        compiler,
        explicit_options,
        {"auto_blockified": True, "auto_blockify_block_count": 64},
    )
    assert command.count("--enable-auto-blockify-loop") == 1
    _, metadata, command = _compile(
        compiler,
        explicit_options,
        {"auto_blockified": False, "auto_blockify_block_count": 0},
    )
    assert command.count("--enable-auto-blockify-loop") == 1
    assert metadata["auto_blockified"] is False
    assert metadata["auto_blockify_block_count"] == 0
    try:
        _compile(compiler, explicit_options)
    except ValueError as error:
        assert "does not provide SIMT auto blockify feedback" in str(error)
    else:
        raise AssertionError("requested auto blockify without compiler feedback must be rejected")

    os.environ["NPU_DEVICE_LIMIT"] = "20,40"
    limited_options = compiler.NPUOptions(
        compile_mode="simt_only",
        enable_auto_blockify=True,
    )
    assert limited_options.hash() != explicit_options.hash()
    assert str(limited_options) != str(explicit_options)
    assert "npu_device_limit_snapshot='20,40'" in str(limited_options)
    metadata_type = namedtuple("KernelMetadata", limited_options.__dict__.keys())
    metadata_type(**limited_options.__dict__)

    class LimitedNPUUtils:

        def has_device_limit(self):
            return False

        def get_aicore_num(self):
            return 20

        def get_aivector_core_num(self):
            return 40

    compiler.NPUUtils = LimitedNPUUtils
    _, _, command = _compile(
        compiler,
        limited_options,
        {"auto_blockified": True, "auto_blockify_block_count": 40},
    )
    assert command.count("--custom-aic-number=20") == 1
    assert command.count("--custom-aiv-number=40") == 1
    os.environ.pop("NPU_DEVICE_LIMIT")
    compiler.NPUUtils = UnlimitedNPUUtils

    superblock_options = compiler.NPUOptions(
        compile_mode="simt_only",
        enable_auto_blockify=False,
        superblock_factor=2,
    )
    assert superblock_options.enable_auto_blockify is True
    _, _, command = _compile(
        compiler,
        superblock_options,
        {"auto_blockified": True, "auto_blockify_block_count": 64},
    )
    assert command.count("--enable-auto-blockify-loop") == 1
    assert command.count("--super-block-factor=2") == 1

    try:
        _compile(
            compiler,
            superblock_options,
            {"auto_blockified": False, "auto_blockify_block_count": 0},
        )
    except ValueError as error:
        assert "confirm successful SIMT auto blockify" in str(error)
    else:
        raise AssertionError("superblocking without compiler confirmation must be rejected")

    try:
        _compile(compiler, superblock_options, has_blacklist=True)
    except ValueError as error:
        assert "superblock_factor=2" in str(error)
    else:
        raise AssertionError("blacklisted superblocking must be rejected")

    invalid_payloads = (
        ([], "JSON object"),
        ({"auto_blockified": "true", "auto_blockify_block_count": 64}, "must be a boolean"),
        ({"auto_blockified": True, "auto_blockify_block_count": True}, "non-negative integer"),
        ({"auto_blockified": False, "auto_blockify_block_count": 64}, "are inconsistent"),
    )
    for payload, expected_message in invalid_payloads:
        try:
            _compile(compiler, explicit_options, payload)
        except ValueError as error:
            assert expected_message in str(error)
        else:
            raise AssertionError(f"invalid metadata payload was accepted: {payload!r}")


def test_auto_blockify_feedback_contract():
    original_ascend_modules = _ascend_module_state()
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), _CHILD_MODE],
        capture_output=True,
        text=True,
        check=False,
    )
    restored_ascend_modules = _ascend_module_state()

    assert restored_ascend_modules.keys() == original_ascend_modules.keys()
    assert all(restored_ascend_modules[name] is module for name, module in original_ascend_modules.items())
    assert completed.returncode == 0, ("auto-blockify feedback child failed\n"
                                       f"stdout:\n{completed.stdout}\n"
                                       f"stderr:\n{completed.stderr}")


if __name__ == "__main__":
    if sys.argv[1:] != [_CHILD_MODE]:
        raise SystemExit(f"expected private child mode {_CHILD_MODE}")
    _run_auto_blockify_feedback_child()
