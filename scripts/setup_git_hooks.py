"""
SIGNOVA - Automated Git Pre-Commit Hook Installer
Sets up automatic quality, linting, and testing verification before every git commit.
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parents[1]
HOOKS_DIR = ROOT_DIR / ".git" / "hooks"
PRE_COMMIT_HOOK = HOOKS_DIR / "pre-commit"

HOOK_CONTENT = """#!/bin/sh
# SIGNOVA Automated Pre-Commit Quality Gate
# Automatically executes linting, guardrails, and test suite before committing.

echo ""
echo "=================================================="
echo "⚡ SIGNOVA: Running Automated Pre-Commit Checks..."
echo "=================================================="

# 1. Code Style & Linting
echo "[1/4] Checking code quality with Ruff..."
python -m ruff check .
if [ $? -ne 0 ]; then
    echo "❌ Ruff linting failed. Please fix issues before committing."
    exit 1
fi

# 2. Code Formatting
echo "[2/4] Checking formatting with Ruff..."
python -m ruff format --check .
if [ $? -ne 0 ]; then
    echo "❌ Formatting check failed. Run 'python -m ruff format .' to auto-format."
    exit 1
fi

# 3. Domain Guardrails, Secrets & ADR-001
echo "[3/4] Running project quality & linguistic guardrails..."
python scripts/ci/check_project_quality.py
if [ $? -ne 0 ]; then
    echo "❌ Project quality guardrails failed. Commit aborted."
    exit 1
fi

# 4. Automated Tests
echo "[4/4] Executing test suite with Pytest..."
python -m pytest -v
if [ $? -ne 0 ]; then
    echo "❌ Test suite failed. Commit aborted."
    exit 1
fi

echo "=================================================="
echo "✅ All automated checks passed! Proceeding with commit."
echo "=================================================="
echo ""
exit 0
"""


def install_hook():
    if not HOOKS_DIR.exists():
        print(f"Error: Git hooks directory not found at {HOOKS_DIR}")
        sys.exit(1)

    PRE_COMMIT_HOOK.write_text(HOOK_CONTENT, encoding="utf-8")

    # On POSIX systems, ensure executable permission
    try:
        current_mode = PRE_COMMIT_HOOK.stat().st_mode
        PRE_COMMIT_HOOK.chmod(current_mode | 0o755)
    except OSError:
        pass

    print("✅ Automated pre-commit hook successfully installed at:")
    print(f"   {PRE_COMMIT_HOOK}")
    print("\nFrom now on, 'git commit' will automatically run all quality checks and tests!")


if __name__ == "__main__":
    install_hook()
