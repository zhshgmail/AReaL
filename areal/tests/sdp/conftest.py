"""Pytest configuration for Segment-wise Decoupled PPO tests.

This module handles platform-specific compatibility issues.

IMPORTANT: This file is loaded by pytest BEFORE test collection, so mocks must be
set up early to handle module-level imports in the areal codebase.
"""
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

# Mock platform-specific or optional dependencies FIRST, before any areal imports
#
# uvloop: doesn't support Windows, used at module level
#   - cli_args.py calls uvloop.install() at import time
#   - workflow_api.py calls uvloop.run() to run async event loops
#   - MagicMock is sufficient because tests never call uvloop.run()
#
# ray: optional distributed computing framework
#   - utils/name_resolve.py imports ray at module level
#   - Used for model name resolution in distributed settings
#   - Tests don't use ray features, so MagicMock is sufficient
#
# megatron: NVIDIA's large-scale training library
#   - workflow_api.py imports megatron.core.parallel_state
#   - Used for distributed model parallelism
#   - Tests don't use megatron features, so MagicMock is sufficient
#
# swanlab: optional experiment tracking library
#   - utils/stats_logger.py imports swanlab
#   - Tests don't use swanlab features, so MagicMock is sufficient
#
if sys.platform == "win32":
    for module in ["uvloop", "ray", "numba", "swanlab", "wandb", "tensorboardX"]:
        if module not in sys.modules:
            sys.modules[module] = MagicMock()

    # tabulate is special - it's a module with a tabulate function
    if "tabulate" not in sys.modules:
        tabulate_mock = MagicMock()
        tabulate_mock.tabulate = MagicMock(return_value="")
        tabulate_mock.__spec__ = MagicMock()
        tabulate_mock.__spec__.name = "tabulate"
        sys.modules["tabulate"] = tabulate_mock

    # Mock megatron.core and its submodules
    if "megatron" not in sys.modules:
        megatron_mock = MagicMock()
        megatron_core_mock = MagicMock()
        parallel_state_mock = MagicMock()

        megatron_core_mock.parallel_state = parallel_state_mock
        megatron_mock.core = megatron_core_mock

        sys.modules["megatron"] = megatron_mock
        sys.modules["megatron.core"] = megatron_core_mock
        sys.modules["megatron.core.parallel_state"] = parallel_state_mock


# Fix cross-drive path issues on Windows for Hydra config loading
@pytest.fixture
def tmp_path(tmp_path):
    """Override tmp_path to create temp directories on the same drive as the code.

    This fixes Hydra config loading issues on Windows where os.path.relpath()
    fails when tmp_path is on C: drive but code is on D: drive.
    """
    if sys.platform == "win32":
        # Get the drive of the current working directory
        cwd_drive = Path.cwd().drive
        tmp_drive = tmp_path.drive

        # If on different drives, create temp dir on code's drive
        if cwd_drive != tmp_drive:
            # Create temp dir in project's directory
            project_tmp = Path.cwd() / ".pytest_tmp"
            project_tmp.mkdir(exist_ok=True)

            # Create unique temp dir for this test
            import tempfile
            test_tmp = Path(tempfile.mkdtemp(dir=project_tmp))
            yield test_tmp

            # Cleanup
            import shutil
            shutil.rmtree(test_tmp, ignore_errors=True)
            return

    # For non-Windows or same-drive, use default tmp_path
    yield tmp_path
