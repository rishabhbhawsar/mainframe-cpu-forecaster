"""Recompute pooled out-of-fold metrics from training_summary.csv and check published figures.

Usage: python scripts/calculate_metrics.py [--summary-path PATH]
Exit status: 0 all checks pass, 1 a check failed, 2 input missing or malformed.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd

PUBLISHED = {"mae": 1.564, "rmse": 1.968}
PUBLISHED_ROUNDED = {"mae": 1.56, "rmse": 1.97}
TOLERANCE = 5e-4
REQUIRED_COLUMNS = ("box", "system", "service_class", "status", "n_rows", "n_folds", "mae", "rmse")
WIDTH = 78


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "src").is_dir():
            return candidate
    return start


def load_summary(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    frame["status"] = frame["status"].astype(str).str.lower()
    return frame


def pooled_metrics(passed: pd.DataFrame) -> Tuple[float, float, str]:
    """Row-weighted pooled MAE and RMSE; RMSE pools through squared error."""
    if "n_oof_rows" in passed.columns:
        weights, basis = passed["n_oof_rows"].to_numpy(dtype="float64"), "n_oof_rows"
    else:
        weights, basis = passed["n_rows"].to_numpy(dtype="float64"), "n_rows (proxy)"
    mae = float((passed["mae"].to_numpy() * weights).sum() / weights.sum())
    mse = float(((passed["rmse"].to_numpy() ** 2) * weights).sum() / weights.sum())
    return mae, math.sqrt(mse), basis


def render(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    line = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    fmt = lambda r: "|" + "|".join(f" {c.ljust(w)} " for c, w in zip(r, widths)) + "|"
    return "\n".join([line, fmt(headers), line, *(fmt(r) for r in rows), line])


def main(argv: Optional[Sequence[str]] = None) -> int:
    root = find_repo_root(Path(__file__).resolve().parent)
    parser = argparse.ArgumentParser(description="Verify pooled metrics from training_summary.csv.")
    parser.add_argument("--summary-path", default=str(root / "artifacts" / "models" / "training_summary.csv"))
    args = parser.parse_args(argv)

    path = Path(args.summary_path)
    print("=" * WIDTH + "\nPIPELINE METRIC VERIFICATION (source: training_summary.csv)\n" + "=" * WIDTH)
    if not path.is_file():
        print(f"Not found: {path}\nGenerate it with: python -m scripts.run_pipeline")
        return 2
    try:
        frame = load_summary(path)
    except (ValueError, pd.errors.ParserError) as exc:
        print(f"Malformed summary: {exc}")
        return 2

    passed = frame[frame["status"] == "passed"]
    rejected = frame[frame["status"] != "passed"]
    if passed.empty:
        print("No gate-passing series in summary; nothing to verify.")
        return 2

    print(f"file            : {path}")
    print(f"series          : {len(frame)} total, {len(passed)} passed, {len(rejected)} not exported\n")
    print(
        render(
            ["Box", "System", "Service Class", "Status", "Rows", "MAE", "RMSE"],
            [
                [str(r.box), str(r.system), str(r.service_class), r.status.upper(),
                 str(int(r.n_rows)), f"{r.mae:.6f}", f"{r.rmse:.6f}"]
                for r in frame.itertuples()
            ],
        )
    )

    mae, rmse, basis = pooled_metrics(passed)
    unweighted_mae = float(passed["mae"].mean())
    equal_weights = passed["n_rows"].nunique() == 1
    print(f"\npooling weights : {basis}{'' if equal_weights else '  (unequal: approximation)'}")
    print(f"pooled MAE      : {mae:.9f}  (unweighted mean {unweighted_mae:.9f})")
    print(f"pooled RMSE     : {rmse:.9f}")

    checks: List[Tuple[str, bool, str]] = [
        ("MAE matches published 1.564", abs(mae - PUBLISHED["mae"]) <= TOLERANCE, f"{mae:.4f}"),
        ("RMSE matches published 1.968", abs(rmse - PUBLISHED["rmse"]) <= TOLERANCE, f"{rmse:.4f}"),
        ("MAE rounds to resume figure 1.56", round(mae, 2) == PUBLISHED_ROUNDED["mae"], f"{round(mae, 2)}"),
        ("RMSE rounds to resume figure 1.97", round(rmse, 2) == PUBLISHED_ROUNDED["rmse"], f"{round(rmse, 2)}"),
        ("Every series metric finite and positive",
         bool(((passed[["mae", "rmse"]] > 0) & passed[["mae", "rmse"]].notna()).all().all()), ""),
        ("RMSE >= MAE for every series", bool((passed["rmse"] >= passed["mae"]).all()), ""),
    ]
    artifacts = sorted(path.parent.glob("*.joblib"))
    checks.append(
        ("One .joblib artifact per passed series", len(artifacts) == len(passed),
         f"{len(artifacts)} files, {len(passed)} passed")
    )

    print("\n" + render(["Status", "Check", "Detail"],
                        [["PASS" if ok else "FAIL", name, detail] for name, ok, detail in checks]))
    print(
        "\nOut of scope: train-serve skew and the 52-check gateway matrix are verified by\n"
        "scripts/test_gateway.py (parity check + verdict table), not by this file."
    )

    ok = all(c[1] for c in checks)
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())