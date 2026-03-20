"""
Run all Atlas tests with detailed output.

Usage:
    python scripts/run_tests.py          # Run all tests
    python scripts/run_tests.py --quick  # Run only fast tests (skip e2e)
    python scripts/run_tests.py --gpu    # Include GPU-specific tests
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run Atlas tests")
    parser.add_argument("--quick", action="store_true", help="Skip slow e2e tests")
    parser.add_argument("--gpu", action="store_true", help="Include GPU tests")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent

    cmd = [
        sys.executable, "-m", "pytest",
        str(project_root / "tests"),
        "-v",
        "--tb=short",
        "-x",  # Stop on first failure
    ]

    if args.quick:
        cmd.extend(["--ignore", str(project_root / "tests" / "test_e2e.py")])

    if not args.gpu:
        cmd.extend(["-k", "not gpu"])

    print(f"Running: {' '.join(cmd)}")
    print("=" * 60)

    result = subprocess.run(cmd, cwd=str(project_root))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
