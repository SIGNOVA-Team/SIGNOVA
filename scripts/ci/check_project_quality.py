"""
SIGNOVA - Automated Project Quality, Hygiene & Domain Guardrails Checker
This script executes comprehensive quality checks across the entire repository:
  1. Secret & Credential Leakage Detection
  2. Git Hygiene & Forbidden Binary/Video File Detection
  3. Jupyter Notebook Health & JSON Integrity Check
  4. Linguistic Guardrails (ADR-001 ArSL != ESL enforcement)
  5. Core Dataset & Documentation Deliverables Completeness Check
"""

import csv
import json
import os
import re
import sys
from pathlib import Path

# Color styling for CLI output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

ROOT_DIR = Path(__file__).resolve().parents[2]


def print_header(title: str):
    print(f"\n{BOLD}{CYAN}=== {title} ==={RESET}")


def check_secrets_and_credentials() -> tuple[bool, list[str]]:
    """Scan tracked repository files for exposed secrets, private keys, or .env files."""
    errors = []
    suspicious_patterns = [
        re.compile(r"-----BEGIN (RSA|OPENSSH|EC|DSA|PRIVATE) KEY-----"),
        re.compile(
            r"(api[_-]?key|secret[_-]?key|auth[_-]?token)\s*=\s*['\"][A-Za-z0-9_\-]{20,}['\"]",
            re.IGNORECASE,
        ),
        re.compile(r"ghp_[A-Za-z0-9]{36}"),
        re.compile(r"AIzaSy[A-Za-z0-9_-]{33}"),  # Google API key pattern
    ]

    # Forbidden files
    forbidden_filenames = {".env", ".env.local", "id_rsa", "id_ed25519"}

    for root, dirs, files in os.walk(ROOT_DIR):
        # Skip git, cache, and virtual environments
        dirs[:] = [
            d
            for d in dirs
            if d not in {".git", "__pycache__", ".pytest_cache", ".venv", "venv", "node_modules"}
        ]

        for f in files:
            if f in forbidden_filenames or f.endswith((".pem", ".key", ".pfx")):
                errors.append(f"Forbidden sensitive file found: {Path(root) / f}")

            fpath = Path(root) / f
            # Only scan text/code files
            if fpath.suffix.lower() in {".py", ".md", ".json", ".yaml", ".yml", ".sh", ".toml"}:
                try:
                    content = fpath.read_text(encoding="utf-8", errors="ignore")
                    for pattern in suspicious_patterns:
                        if pattern.search(content):
                            errors.append(
                                f"Potential secret/key match in: {fpath.relative_to(ROOT_DIR)}"
                            )
                            break
                except OSError as e:
                    errors.append(f"Failed to read {fpath}: {e}")

    return len(errors) == 0, errors


def check_git_hygiene_and_binaries() -> tuple[bool, list[str]]:
    """Verify no large video files (*.mp4) or binary archives (*.7z, *.npy) are tracked."""
    errors = []
    forbidden_extensions = {
        ".mp4",
        ".7z",
        ".avi",
        ".mov",
        ".npy",
        ".npz",
        ".pth",
        ".pt",
        ".bin",
        ".h5",
    }
    max_allowed_file_size = 50 * 1024 * 1024  # 50 MB max for any committed file

    for root, dirs, files in os.walk(ROOT_DIR):
        dirs[:] = [
            d for d in dirs if d not in {".git", "__pycache__", ".pytest_cache", ".venv", "venv"}
        ]

        for f in files:
            fpath = Path(root) / f
            ext = fpath.suffix.lower()

            if ext in forbidden_extensions:
                errors.append(
                    f"Forbidden raw binary/media file committed: {fpath.relative_to(ROOT_DIR)}"
                )

            try:
                size = fpath.stat().st_size
                if size > max_allowed_file_size:
                    errors.append(
                        f"File exceeds maximum allowed size (50MB): {fpath.relative_to(ROOT_DIR)} ({size / (1024 * 1024):.1f} MB)"
                    )
            except OSError as e:
                errors.append(f"Unable to stat file {fpath}: {e}")

    return len(errors) == 0, errors


def check_notebooks_integrity() -> tuple[bool, list[str]]:
    """Verify that all Jupyter notebooks are valid JSON and not corrupt."""
    errors = []
    notebook_files = list(ROOT_DIR.glob("**/*.ipynb"))
    notebook_files = [nb for nb in notebook_files if ".ipynb_checkpoints" not in str(nb)]

    if not notebook_files:
        errors.append("No notebooks found to check.")
        return False, errors

    for nb_path in notebook_files:
        rel_path = nb_path.relative_to(ROOT_DIR)
        try:
            with open(nb_path, encoding="utf-8") as f:
                data = json.load(f)

            if "cells" not in data:
                errors.append(f"Notebook {rel_path} is missing 'cells' array.")
            if "metadata" not in data:
                errors.append(f"Notebook {rel_path} is missing 'metadata' dictionary.")
            if not isinstance(data.get("cells"), list):
                errors.append(f"Notebook {rel_path} 'cells' is not a list.")

        except json.JSONDecodeError as e:
            errors.append(f"Corrupt notebook JSON in {rel_path}: {e}")
        except OSError as e:
            errors.append(f"Error accessing notebook {rel_path}: {e}")

    return len(errors) == 0, errors


def check_linguistic_guardrails() -> tuple[bool, list[str]]:
    """Enforce ADR-001 rule: ArabSign and KArSL must strictly be ArSL and NEVER labeled ESL."""
    errors = []

    # Check ArabSign Canonical Metadata
    arab_meta = ROOT_DIR / "data" / "metadata" / "arabsign_canonical_metadata.csv"
    if arab_meta.exists():
        with open(arab_meta, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader, 1):
                if row.get("sign_language") != "ArSL":
                    errors.append(
                        f"ArabSign sample {row.get('sample_id', i)} has invalid sign_language: '{row.get('sign_language')}' (must be ArSL)"
                    )
                    break
                if "DIALECT_NOT_ESL" not in row.get("qc_flags", ""):
                    errors.append(
                        f"ArabSign sample {row.get('sample_id', i)} is missing required 'DIALECT_NOT_ESL' flag"
                    )
                    break
    else:
        errors.append("Missing arabsign_canonical_metadata.csv")

    # Check KArSL Metadata
    karsl_meta = ROOT_DIR / "karsl" / "data" / "metadata" / "karsl_metadata.csv"
    if karsl_meta.exists():
        with open(karsl_meta, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # Sample first 500 rows for rapid assertion
            for i, row in enumerate(reader):
                if i > 500:
                    break
                if row.get("sign_language") != "ArSL":
                    errors.append(
                        f"KArSL sample {row.get('sample_id', i)} has invalid sign_language: '{row.get('sign_language')}' (must be ArSL)"
                    )
                    break
    else:
        errors.append("Missing karsl_metadata.csv")

    return len(errors) == 0, errors


def check_core_deliverables() -> tuple[bool, list[str]]:
    """Verify presence and non-emptiness of core artifacts."""
    errors = []
    required_paths = [
        ROOT_DIR / "README.md",
        ROOT_DIR / "requirements.txt",
        ROOT_DIR / "requirements-dev.txt",
        ROOT_DIR / "pyproject.toml",
        ROOT_DIR / "docs" / "dataset_card_how2sign.md",
        ROOT_DIR / "docs" / "dataset_card_arabsign.md",
        ROOT_DIR / "docs" / "decisions" / "ADR_001_p2_datasets_evaluation.md",
        ROOT_DIR / "karsl" / "README.md",
        ROOT_DIR / "karsl" / "docs" / "karsl_dataset_card.md",
        ROOT_DIR / "karsl" / "docs" / "karsl_handoff.md",
        ROOT_DIR / "karsl" / "data" / "reports" / "karsl_benchmark_summary.md",
    ]

    for p in required_paths:
        if not p.exists():
            errors.append(f"Missing core deliverable: {p.relative_to(ROOT_DIR)}")
        elif p.stat().st_size == 0:
            errors.append(f"Empty core deliverable: {p.relative_to(ROOT_DIR)}")

    return len(errors) == 0, errors


def main() -> int:
    print(f"\n{BOLD}{CYAN}======================================================{RESET}")
    print(f"{BOLD}{CYAN}       SIGNOVA - Automated Project Quality Gate       {RESET}")
    print(f"{BOLD}{CYAN}======================================================{RESET}")

    all_passed = True
    checks = [
        ("Secrets & Credentials Scan", check_secrets_and_credentials),
        ("Git Hygiene & Binary Files", check_git_hygiene_and_binaries),
        ("Notebook Health & JSON Syntax", check_notebooks_integrity),
        ("Linguistic Guardrails (ADR-001)", check_linguistic_guardrails),
        ("Core Deliverables & Schema Presence", check_core_deliverables),
    ]

    results_table = []

    for name, func in checks:
        passed, errors = func()
        status_text = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        results_table.append((name, passed, errors))
        print(f"[{status_text}] {name}")
        if not passed:
            all_passed = False
            for err in errors:
                print(f"       {RED}* {err}{RESET}")

    print(f"\n{BOLD}{CYAN}------------------------------------------------------{RESET}")
    if all_passed:
        print(f"{BOLD}{GREEN}ALL QUALITY & DOMAIN GUARDRAILS PASSED (5/5){RESET}")
        print(f"{BOLD}{CYAN}======================================================\n{RESET}")
        return 0
    else:
        print(f"{BOLD}{RED}QUALITY CHECKS FAILED - PLEASE REVIEW ERRORS ABOVE{RESET}")
        print(f"{BOLD}{CYAN}======================================================\n{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
