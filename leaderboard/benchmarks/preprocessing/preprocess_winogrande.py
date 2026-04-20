# uv run python benchmarks/preprocessing/preprocess_winogrande.py

from __future__ import annotations

import hashlib
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
    s = "\n".join(_ws_re.sub(" ", line).strip() for line in s.split("\n"))
    return s.strip()


def stable_id(split: str, i: int, ex: dict) -> str:
    """
    Stable ID strategy:
      1) Use HF 'id' field if present and is a simple scalar (not dict/list).
      2) Else hash sentence+options (stable across runs).
      3) Else fallback to index.
    """
    raw_id = ex.get("id")

    if isinstance(raw_id, (str, int)) and str(raw_id).strip():
        safe = str(raw_id).replace("/", "_").replace(" ", "_")
        return f"winogrande_{split}_{safe}"

    sentence = clean_text(ex.get("sentence", ""))
    option1 = clean_text(ex.get("option1", ""))
    option2 = clean_text(ex.get("option2", ""))

    if sentence and option1 and option2:
        canonical = f"{sentence}\nA) {option1}\nB) {option2}"
        h = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]
        return f"winogrande_{split}_sha1_{h}"

    return f"winogrande_{split}_{i:08d}"


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


def build_prompt(sentence: str, choices: List[str]) -> str:
    letters = ["A", "B"]
    lines = [f"Context: {sentence}"]
    for i, ch in enumerate(choices):
        lines.append(f"{letters[i]}) {ch}")
    lines.append('Respond with only "A" or "B".')
    lines.append("Answer:")
    return "\n".join(lines)


def main() -> None:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    out_dir = REPO_ROOT / "benchmarks" / "winogrande" / "cleaned"
    out_dir.mkdir(parents=True, exist_ok=True)

    hf_dataset = "allenai/winogrande"
    hf_config = "winogrande_xl"
    hf_split = "validation"

    ds = load_dataset(hf_dataset, hf_config, split=hf_split)

    out_path = out_dir / "validation.jsonl"
    letters = ["A", "B"]

    n_written = 0
    n_skipped = 0

    with out_path.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            sentence = clean_text(ex.get("sentence", ""))
            option1 = clean_text(ex.get("option1", ""))
            option2 = clean_text(ex.get("option2", ""))
            answer = ex.get("answer", None)  # usually "1" or "2"

            if not sentence or not option1 or not option2:
                n_skipped += 1
                continue

            try:
                answer_index = int(answer) - 1  # "1"/"2" -> 0/1
            except (TypeError, ValueError):
                n_skipped += 1
                continue

            if answer_index not in (0, 1):
                n_skipped += 1
                continue

            choices = [option1, option2]
            prompt = build_prompt(sentence, choices)

            record = CleanMCExample(
                benchmark="winogrande",
                split="validation",
                id=stable_id("val", i, ex),
                prompt=prompt,
                context=sentence,
                choices=choices,
                answer_index=answer_index,
                answer_letter=letters[answer_index],
                answer_text=choices[answer_index],
                source={
                    "hf_dataset": hf_dataset,
                    "hf_config": hf_config,
                    "hf_split": hf_split,
                    "id": ex.get("id"),
                },
            )

            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            n_written += 1

    meta = {
        "benchmark": "winogrande",
        "split": "validation",
        "hf_dataset": hf_dataset,
        "hf_config": hf_config,
        "hf_split": hf_split,
        "rows_in_split": len(ds),
        "written": n_written,
        "skipped": n_skipped,
        "output_file": str(out_path),
        "schema": "CleanMCExample",
        "note": "Winogrande is binary multiple choice; model is expected to output A or B.",
    }
    (out_dir / "validation_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[winogrande] hf_split rows: {len(ds)}")
    print(f"[winogrande] written: {n_written}, skipped: {n_skipped}")
    print(f"[winogrande] saved: {out_path}")


if __name__ == "__main__":
    main()