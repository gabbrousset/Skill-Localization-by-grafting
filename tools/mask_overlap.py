#!/usr/bin/env python
"""Compute pairwise overlap for saved graft mask checkpoints."""

import argparse
import csv
import math
import sys
from pathlib import Path

import torch


def parse_mask_arg(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("mask arguments must have the form LABEL=PATH")
    label, path = value.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("mask label cannot be empty")
    return label, Path(path)


def infer_label(path, root):
    rel = path.relative_to(root)
    parts = rel.parts
    if len(parts) >= 5:
        return "/".join(parts[:4] + (path.stem,))
    return str(rel.with_suffix(""))


def load_binary_mask(path, sigmoid_bias):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        tensors = [obj]
    elif isinstance(obj, dict):
        tensors = [obj[key] for key in sorted(obj)]
    elif isinstance(obj, (list, tuple)):
        tensors = list(obj)
    else:
        raise TypeError(f"Unsupported mask checkpoint type in {path}: {type(obj)!r}")

    pieces = []
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Mask checkpoint {path} contains a non-tensor entry: {type(tensor)!r}")
        binary = torch.round(torch.sigmoid(tensor.float() - sigmoid_bias)).to(torch.bool)
        pieces.append(binary.reshape(-1))
    if not pieces:
        raise ValueError(f"Mask checkpoint {path} did not contain any tensors")
    return torch.cat(pieces)


def write_rows(rows, output_csv):
    if output_csv == "-":
        writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        return

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mask",
        action="append",
        default=[],
        type=parse_mask_arg,
        metavar="LABEL=PATH",
        help="Saved graft mask checkpoint. Can be passed multiple times.",
    )
    parser.add_argument(
        "--mask-dir",
        action="append",
        default=[],
        type=Path,
        help="Directory to scan recursively for .pt mask checkpoints.",
    )
    parser.add_argument("--pattern", default="*.pt", help="Glob pattern used with --mask-dir.")
    parser.add_argument("--sigmoid-bias", type=float, default=10.0)
    parser.add_argument("--output-csv", default="-", help="Output CSV path, or '-' for stdout.")
    args = parser.parse_args()

    masks = list(args.mask)
    for root in args.mask_dir:
        for path in sorted(root.rglob(args.pattern)):
            masks.append((infer_label(path, root), path))

    if not masks:
        raise SystemExit("No masks were provided. Use --mask LABEL=PATH or --mask-dir DIR.")

    loaded = []
    expected_size = None
    for label, path in masks:
        if not path.exists():
            raise FileNotFoundError(path)
        binary = load_binary_mask(path, args.sigmoid_bias)
        if expected_size is None:
            expected_size = binary.numel()
        elif binary.numel() != expected_size:
            raise ValueError(
                f"Mask {path} has {binary.numel()} parameters, expected {expected_size}. "
                "Only compare masks from the same model architecture and trainable-parameter setting."
            )
        loaded.append((label, path, binary))

    rows = []
    for row_label, row_path, row_mask in loaded:
        row_active = int(row_mask.sum().item())
        for col_label, col_path, col_mask in loaded:
            col_active = int(col_mask.sum().item())
            intersection = int(torch.logical_and(row_mask, col_mask).sum().item())
            overlap = math.nan if col_active == 0 else intersection / float(col_active)
            rows.append(
                {
                    "row_mask": row_label,
                    "column_mask": col_label,
                    "row_path": str(row_path),
                    "column_path": str(col_path),
                    "row_active": row_active,
                    "column_active": col_active,
                    "intersection": intersection,
                    "overlap_fraction": overlap,
                }
            )

    write_rows(rows, args.output_csv)


if __name__ == "__main__":
    main()
