"""
Merge ClArTTS parquet shards (from the notebook HF download) into the paths this repo expects:

  clartts/combined.parquet        — all train-* shards concatenated
  clartts/clartts_train.parquet   — random split from combined (default 90%%)
  clartts/clartts_val.parquet
  clartts/clartts_test.parquet    — all test-* shards concatenated (if any)

Run from repository root, e.g.:
  python tacotron2/combine_clartts_parquets.py
  python tacotron2/combine_clartts_parquets.py --input-dir clartts --val-fraction 0.1
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _sort_train(paths: list[Path]) -> list[Path]:
    def key(p: Path) -> tuple[int, str]:
        m = re.search(r"train-(\d+)-of-", p.name)
        return (int(m.group(1)), p.name) if m else (10**9, p.name)

    return sorted(paths, key=key)


def _sort_test(paths: list[Path]) -> list[Path]:
    def key(p: Path) -> tuple[int, str]:
        m = re.search(r"test-(\d+)-of-", p.name)
        return (int(m.group(1)), p.name) if m else (10**9, p.name)

    return sorted(paths, key=key)


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine ClArTTS parquet shards into standard filenames.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory containing train-*.parquet and test-*.parquet (default: <repo>/clartts)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to write outputs (default: same as input-dir)",
    )
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Fraction for clartts_val (from train shards).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split.")
    parser.add_argument("--skip-combined", action="store_true", help="Do not write combined.parquet.")
    parser.add_argument("--skip-train-val", action="store_true", help="Do not write clartts_train / clartts_val.")
    parser.add_argument("--skip-test", action="store_true", help="Do not write clartts_test.parquet.")
    args = parser.parse_args()

    repo = _repo_root()
    in_dir = Path(args.input_dir) if args.input_dir else repo / "clartts"
    if not in_dir.is_dir():
        print(f"Input directory not found: {in_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else in_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_files = _sort_train(sorted(in_dir.glob("train-*.parquet")))
    test_files = _sort_test(sorted(in_dir.glob("test-*.parquet")))

    if not train_files:
        print(f"No train-*.parquet files under {in_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {len(train_files)} train shard(s)…")
    train_parts = [pd.read_parquet(f) for f in train_files]
    full_train = pd.concat(train_parts, ignore_index=True)
    print(f"  Combined train rows: {len(full_train)}")

    if not args.skip_combined:
        combined_path = out_dir / "combined.parquet"
        full_train.to_parquet(combined_path, index=False)
        print(f"Wrote {combined_path}")

    if not args.skip_train_val:
        if not (0 < args.val_fraction < 1):
            print("--val-fraction must be in (0, 1)", file=sys.stderr)
            sys.exit(1)
        tr_df, va_df = train_test_split(
            full_train,
            test_size=args.val_fraction,
            random_state=args.seed,
        )
        train_out = out_dir / "clartts_train.parquet"
        val_out = out_dir / "clartts_val.parquet"
        tr_df.to_parquet(train_out, index=False)
        va_df.to_parquet(val_out, index=False)
        print(f"Wrote {train_out} ({len(tr_df)} rows), {val_out} ({len(va_df)} rows)")

    if not args.skip_test and test_files:
        print(f"Reading {len(test_files)} test shard(s)…")
        test_parts = [pd.read_parquet(f) for f in test_files]
        full_test = pd.concat(test_parts, ignore_index=True)
        test_out = out_dir / "clartts_test.parquet"
        full_test.to_parquet(test_out, index=False)
        print(f"Wrote {test_out} ({len(full_test)} rows)")
    elif not test_files and not args.skip_test:
        print("No test-*.parquet shards found; skipping clartts_test.parquet")

    print("Done.")


if __name__ == "__main__":
    main()
