"""Shared fixtures for crucible tests."""

from __future__ import annotations

import sys
from pathlib import Path

# Add scripts dir to path so we can import the modules under test
SCRIPTS_DIR = Path(__file__).parent.parent / "skills" / "crucible" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
