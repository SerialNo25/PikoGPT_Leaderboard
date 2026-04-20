# uv run python benchmarks/preprocessing/preprocess_hellaswag.py

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset


# ----------------------------
# Cleaning helpers
# ----------------------------
_ws_re = re.compile(r"[ \t]+")


def clean_text(s: str) -> str:
    """Basic normalization: strip, collapse tabs/spaces, normalize newlines."""
    if s is None:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.strip()
    # collapse repeated spaces in each line
    s = "\n".join(_ws_re.sub(" ", line).strip() for line in s.split("\n"))
    # remove accidental empty lines at start/end
    s = s.strip()
    return s


# ----------------------------
# Output schema
# ----------------------------
@dataclass
class CleanMCExample:
    benchmark: str
    split: str
    id: str
    prompt: str
    context: str
    choices: List[str]
    answer_index: int
    answer_letter: str
    answer_text: str
    source: Dict[str, Any]


def build_prompt(ctx: str, choices: List[str]) -> str:
    letters = ["A", "B", "C", "D"]
    lines = [f"Context: {ctx}"]
    for i, ch in enumerate(choices):
        lines.append(f"{letters[i]}) {ch}")
    lines.append("Answer:")
    return "\n".join(lines)


def stable_id(source_id: str | None, ind: int | None, fallback_i: int) -> str:
    """
    Build a stable-ish id. Prefer (source_id, ind) from the dataset, else fallback.
    """
    if source_id and ind is not None:
        # make filesystem/json friendly
        sid = source_id.replace("~", "_").replace("/", "_")
        return f"hellaswag_val_{sid}_{ind}"
    return f"hellaswag_val_{fallback_i:08d}"


def main() -> None:
    out_dir = Path("benchmarks/hellaswag/cleaned")
    out_dir.mkdir(parents=True, exist_ok=True)

    # IMPORTANT: use validation for local benchmarking (test labels are hidden)
    hf_split = "validation"
    ds = load_dataset("Rowan/hellaswag", split=hf_split)

    letters = ["A", "B", "C", "D"]
    out_path = out_dir / "validation.jsonl"

    n_written = 0
    n_skipped = 0

    with out_path.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            ctx = clean_text(ex.get("ctx", ""))

            endings = ex.get("endings", [])
            label = ex.get("label", None)

            if not ctx or not isinstance(endings, list) or len(endings) != 4:
                n_skipped += 1
                continue

            try:
                answer_index = int(label)  # in this dataset version label is a string "0".."3"
            except (TypeError, ValueError):
                n_skipped += 1
                continue

            if answer_index < 0 or answer_index > 3:
                n_skipped += 1
                continue

            choices = [clean_text(x) for x in endings]
            prompt = build_prompt(ctx, choices)

            record = CleanMCExample(
                benchmark="hellaswag",
                split="validation",
                id=stable_id(ex.get("source_id"), ex.get("ind"), i),
                prompt=prompt,
                context=ctx,
                choices=choices,
                answer_index=answer_index,
                answer_letter=letters[answer_index],
                answer_text=choices[answer_index],
                source={
                    "hf_dataset": "Rowan/hellaswag",
                    "hf_split": hf_split,
                    "ind": ex.get("ind"),
                    "source_id": ex.get("source_id"),
                    "activity_label": ex.get("activity_label"),
                    "split_type": ex.get("split_type"),
                },
            )

            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            n_written += 1

    # Write metadata
    meta = {
        "benchmark": "hellaswag",
        "split": "validation",
        "hf_dataset": "Rowan/hellaswag",
        "hf_split": hf_split,
        "rows_in_split": len(ds),
        "written": n_written,
        "skipped": n_skipped,
        "output_file": str(out_path),
        "schema": "CleanMCExample",
        "note": "HellaSwag test labels are hidden; validation split is used for local benchmarking.",
    }
    (out_dir / "validation_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[hellaswag] hf_split rows: {len(ds)}")
    print(f"[hellaswag] written: {n_written}, skipped: {n_skipped}")
    print(f"[hellaswag] saved: {out_path}")


if __name__ == "__main__":
    main()