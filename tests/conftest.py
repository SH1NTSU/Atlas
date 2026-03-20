"""
Pytest configuration for Atlas tests.

Adds src/ to path so imports work correctly.
"""

import sys
from pathlib import Path

# Add src/ to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
