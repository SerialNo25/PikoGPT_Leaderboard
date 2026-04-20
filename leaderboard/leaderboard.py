#!/usr/bin/env python3
"""
leaderboard.py

Build a simple leaderboard by scanning overview JSON files that contain
{"benchmarks": [...]}.

Current layout:
  Results/<submission>/<checkpoint_name>/*.json

Usage:
  python leaderboard.py
  python leaderboard.py --rank-by public_avg
  python leaderboard.py --rank-by overall_avg
  python leaderboard.py --save-json leaderboard.json --save-csv leaderboard.csv
  python leaderboard.py --root .
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PUBLIC_BENCHES = ["hellaswag", "winogrande", "openbookqa", "lambada"]


@dataclass
class RunRow:
    run_name: str                      # relative run directory (e.g. Results/Group1-test/best)
    output_dir: str                    # from overview json
    checkpoint: str
    limit: Optional[int]

    public_avg: Optional[float]
    overall_avg: Optional[float]

    # Optional extras
    invalid_total: int
    total_total: int

    # Per-benchmark accuracies (dynamic; stored separately)
    per_bench_acc: Dict[str, Optional[float]]


def is_overview_json(obj: Any) -> bool:
    return isinstance(obj, dict) and isinstance(obj.get("benchmarks"), list)


def find_overview_jsons(root: Path) -> tuple[List[Path], List[str]]:
    """
    Find all overview JSON files.

    Preferred layout:
      Results/<submission>/<checkpoint_name>/*.json

    Legacy layout support:
      top-level directories named Results*
    """
    found: list[Path] = []
    skipped: list[str] = []

    results_root = root / "Results"
    if results_root.is_dir():
        for p in sorted(results_root.rglob("*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if is_overview_json(data):
                found.append(p)

    # legacy compatibility with Results-* folders at project root
    for d in sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("Results") and p.name != "Results"]):
        candidates = sorted(d.glob("*.json"))
        ov = None
        for p in candidates:
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if is_overview_json(data):
                ov = p
                break
        if ov is None:
            skipped.append(d.name)
            continue
        found.append(ov)

    # de-duplicate while preserving deterministic order
    uniq = sorted(set(found))
    return uniq, skipped


def mean(values: List[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def extract_run_row(ov_path: Path, root: Path) -> Optional[RunRow]:
    overview = json.loads(ov_path.read_text(encoding="utf-8"))
    run_dir = ov_path.parent

    # Basic metadata
    checkpoint = str(overview.get("checkpoint", ""))
    limit = overview.get("limit", None)
    output_dir = str(overview.get("output_dir", run_dir.name))

    # Per benchmark accuracies
    per_bench: Dict[str, Optional[float]] = {}
    invalid_total = 0
    total_total = 0

    for b in overview.get("benchmarks", []):
        name = b.get("benchmark")
        acc = b.get("accuracy_pct", None)
        if isinstance(name, str):
            per_bench[name] = float(acc) if acc is not None else None
        invalid_total += int(b.get("invalid", 0) or 0)
        total_total += int(b.get("total", 0) or 0)

    public_avg = mean([per_bench.get(b) for b in PUBLIC_BENCHES if per_bench.get(b) is not None])

    # overall: average over ALL benchmarks that have accuracy_pct
    overall_avg = mean([v for v in per_bench.values() if v is not None])

    return RunRow(
        run_name=str(run_dir.resolve().relative_to(root)),
        output_dir=output_dir,
        checkpoint=checkpoint,
        limit=limit if isinstance(limit, int) else None,
        public_avg=public_avg,
        overall_avg=overall_avg,
        invalid_total=invalid_total,
        total_total=total_total,
        per_bench_acc=per_bench,
    )


def format_float(x: Optional[float]) -> str:
    if x is None:
        return "-"
    return f"{x:.2f}"


def shorten_ckpt(s: str, max_len: int = 42) -> str:
    if len(s) <= max_len:
        return s
    return "…" + s[-(max_len - 1):]


def print_table(rows: List[RunRow], rank_by: str) -> None:
    # build columns
    headers = [
        "#",
        "Run",
        f"{rank_by}",
        "public_avg",
        "overall_avg",
        "invalid/total",
        "checkpoint",
    ]

    table: List[List[str]] = []
    for i, r in enumerate(rows, start=1):
        rank_val = getattr(r, rank_by)
        table.append([
            str(i),
            r.run_name,
            format_float(rank_val),
            format_float(r.public_avg),
            format_float(r.overall_avg),
            f"{r.invalid_total}/{r.total_total}",
            shorten_ckpt(r.checkpoint),
        ])

    # compute widths
    widths = [len(h) for h in headers]
    for row in table:
        for j, cell in enumerate(row):
            widths[j] = max(widths[j], len(cell))

    def line(sep: str = "-") -> str:
        return "+".join(sep * (w + 2) for w in widths)

    def render_row(cells: List[str]) -> str:
        return "|".join(f" {c.ljust(widths[i])} " for i, c in enumerate(cells))

    print(line("-"))
    print(render_row(headers))
    print(line("="))
    for row in table:
        print(render_row(row))
        print(line("-"))


def save_json(path: Path, rows: List[RunRow], rank_by: str) -> None:
    payload = {
        "rank_by": rank_by,
        "rows": [
            {
                **{k: v for k, v in asdict(r).items() if k != "per_bench_acc"},
                "per_bench_acc": r.per_bench_acc,
            }
            for r in rows
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_csv(path: Path, rows: List[RunRow]) -> None:
    # Collect all benchmarks seen to create stable columns
    all_benches = sorted({b for r in rows for b in r.per_bench_acc.keys()})

    fieldnames = [
        "rank",
        "run_name",
        "checkpoint",
        "limit",
        "public_avg",
        "overall_avg",
        "invalid_total",
        "total_total",
        *[f"acc_{b}" for b in all_benches],
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, r in enumerate(rows, start=1):
            row = {
                "rank": i,
                "run_name": r.run_name,
                "checkpoint": r.checkpoint,
                "limit": r.limit,
                "public_avg": r.public_avg,
                "overall_avg": r.overall_avg,
                "invalid_total": r.invalid_total,
                "total_total": r.total_total,
            }
            for b in all_benches:
                row[f"acc_{b}"] = r.per_bench_acc.get(b)
            w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(description="Create a leaderboard from saved benchmark overview JSON files.")
    ap.add_argument("--root", type=Path, default=Path("."), help="Project root (default: .)")
    ap.add_argument(
        "--rank-by",
        choices=["public_avg", "overall_avg"],
        default="public_avg",
        help="Ranking metric (default: public_avg)",
    )
    ap.add_argument("--save-json", type=Path, default=None, help="Optional path to save leaderboard JSON")
    ap.add_argument("--save-csv", type=Path, default=None, help="Optional path to save leaderboard CSV")
    args = ap.parse_args()

    root = args.root.resolve()

    rows: List[RunRow] = []
    overview_paths, skipped = find_overview_jsons(root)

    for ov_path in overview_paths:
        row = extract_run_row(ov_path, root)
        if row is None:
            continue
        rows.append(row)

    # Sort (descending). Nones go last.
    def sort_key(r: RunRow) -> Tuple[int, float]:
        val = getattr(r, args.rank_by)
        if val is None:
            return (1, 0.0)  # push to end
        return (0, -float(val))

    rows.sort(key=sort_key)

    if not rows:
        print("No valid evaluation overviews were found.")
        if skipped:
            print("Skipped (legacy Results-* without overview): " + ", ".join(skipped))
        raise SystemExit(1)

    print_table(rows, args.rank_by)

    if skipped:
        print("\nSkipped (legacy Results-* without overview): " + ", ".join(skipped))

    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        save_json(args.save_json, rows, args.rank_by)
        print(f"\nSaved JSON leaderboard to: {args.save_json}")

    if args.save_csv:
        args.save_csv.parent.mkdir(parents=True, exist_ok=True)
        save_csv(args.save_csv, rows)
        print(f"Saved CSV leaderboard to: {args.save_csv}")


if __name__ == "__main__":
    main()
