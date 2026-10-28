"""Inspect a MATLAB .mat file and report nested structures without modifying it."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat


def describe(value: Any, path: str = "root", depth: int = 0, max_depth: int = 5) -> None:
    indent = "  " * depth
    if depth > max_depth:
        print(f"{indent}{path}: <maximum display depth reached>")
        return

    if isinstance(value, dict):
        for key, item in value.items():
            if not key.startswith("__"):
                describe(item, f"{path}.{key}", depth, max_depth)
        return

    if isinstance(value, np.ndarray):
        print(f"{indent}{path}: ndarray shape={value.shape}, dtype={value.dtype}")
        if value.dtype == object and value.size and depth < max_depth:
            for i, item in enumerate(value.flat[:3]):
                describe(item, f"{path}[{i}]", depth + 1, max_depth)
        elif value.dtype.names and value.size and depth < max_depth:
            record = value.flat[0]
            for name in value.dtype.names:
                describe(record[name], f"{path}[0].{name}", depth + 1, max_depth)
        return

    if hasattr(value, "_fieldnames"):
        print(f"{indent}{path}: MATLAB struct fields={value._fieldnames}")
        if depth < max_depth:
            for name in value._fieldnames:
                describe(getattr(value, name), f"{path}.{name}", depth + 1, max_depth)
        return

    if isinstance(value, (list, tuple)):
        print(f"{indent}{path}: {type(value).__name__}, length={len(value)}")
        for i, item in enumerate(value[:3]):
            describe(item, f"{path}[{i}]", depth + 1, max_depth)
        return

    if isinstance(value, (str, bytes)):
        print(f"{indent}{path}: {type(value).__name__}, value={value[:80]!r}")
    elif np.isscalar(value):
        print(f"{indent}{path}: {type(value).__name__}, value={value!r}")
    else:
        print(f"{indent}{path}: {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mat_file", type=Path, help="Path to a source .mat file")
    parser.add_argument("--max-depth", type=int, default=5, help="Maximum nested display depth")
    args = parser.parse_args()

    if not args.mat_file.is_file():
        parser.error(f"File not found: {args.mat_file}")
    if args.mat_file.suffix.lower() != ".mat":
        parser.error("Expected a .mat file")

    data = loadmat(args.mat_file, struct_as_record=False, squeeze_me=True)
    print(f"File: {args.mat_file}")
    print("Top-level variables:")
    for key, value in data.items():
        if not key.startswith("__"):
            describe(value, key, max_depth=args.max_depth)


if __name__ == "__main__":
    main()
