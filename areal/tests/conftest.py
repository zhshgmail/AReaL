"""Global pytest configuration and fixtures for areal tests."""

import sys
from unittest.mock import MagicMock

# Mock uvloop for platforms where it's not available (e.g., Windows)
# This must happen BEFORE any imports of areal modules
if "uvloop" not in sys.modules:
    mock_uvloop = MagicMock()
    mock_uvloop.install = MagicMock()
    sys.modules["uvloop"] = mock_uvloop

# Mock megatron for platforms where it's not installed (e.g., Windows, CPU-only)
# This must happen BEFORE any imports that use megatron
if "megatron" not in sys.modules:
    mock_megatron = MagicMock()
    mock_megatron.core = MagicMock()
    mock_megatron.core.parallel_state = MagicMock()
    sys.modules["megatron"] = mock_megatron
    sys.modules["megatron.core"] = mock_megatron.core
    sys.modules["megatron.core.parallel_state"] = mock_megatron.core.parallel_state
