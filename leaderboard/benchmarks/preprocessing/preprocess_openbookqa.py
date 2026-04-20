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
    s = s.replace("\r\n", "\n").replace("\r", "\n").strip()
    s = "\n".join(_ws_re.sub(" ", line).strip() for line in s.split("\n"))
    return s.strip()


# ----------------------------
# Output schema (same as your MC schema)
# ----------------------------
@dataclass
class CleanMCExample:
    benchmark: str
    split: str
    id: str
    prompt: str
    context: str  # for OpenBookQA this is the question text
    choices: List[str]
    answer_index: int
    answer_letter: str
    answer_text: str
    source: Dict[str, Any]


def build_prompt(question: str, choices: List[str]) -> str:
    letters = ["A", "B", "C", "D"]
    lines = [f"Question: {question}"]
    for i, ch in enumerate(choices):
        lines.append(f"{letters[i]}) {ch}")
    lines.append("Answer:")
    return "\n".join(lines)


def stable_id(split_tag: str, i: int, ex: dict) -> str:
    """
    Stable ID strategy:
      1) Use HF 'id' if it's a simple scalar (str/int).
      2) Else SHA1 hash of question + normalized choices.
      3) Else fallback to index.
    """
    raw_id = ex.get("id")
    if isinstance(raw_id, (str, int)) and str(raw_id).strip():
        safe = str(raw_id).replace("/", "_").replace(" ", "_")
        return f"openbookqa_{split_tag}_{safe}"

    q = clean_text(ex.get("question_stem", ""))
    ch = ex.get("choices", {})
    labels = ch.get("label", []) if isinstance(ch, dict) else []
    texts = ch.get("text", []) if isinstance(ch, dict) else []

    if q and isinstance(texts, list) and texts:
        canonical = q + "\n" + "\n".join(f"{l}:{clean_text(t)}" for l, t in zip(labels, texts))
        h = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]
        return f"openbookqa_{split_tag}_sha1_{h}"

    return f"openbookqa_{split_tag}_{i:08d}"


def answer_to_index(answer_key: Any) -> int | None:
    """
    OpenBookQA answerKey is usually 'A'/'B'/'C'/'D'.
    Some variants may use '1'..'4'. Handle both.
    """
    if answer_key is None:
        return None
    ak = str(answer_key).strip().upper()
    letters = ["A", "B", "C", "D"]
    if ak in letters:
        return letters.index(ak)
    if ak.isdigit():
        n = int(ak)
        if 1 <= n <= 4:
            return n - 1
    return None


def main() -> None:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    out_dir = REPO_ROOT / "benchmarks" / "openbookqa" / "cleaned"
    out_dir.mkdir(parents=True, exist_ok=True)

    hf_dataset = "allenai/openbookqa"
    hf_config = "main"
    hf_split = "validation"  # good for local benchmarking

    ds = load_dataset(hf_dataset, hf_config, split=hf_split)

    out_path = out_dir / "validation.jsonl"
    letters = ["A", "B", "C", "D"]

    n_written = 0
    n_skipped = 0

    with out_path.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            question = clean_text(ex.get("question_stem", ""))
            ch = ex.get("choices", {})
            answer_key = ex.get("answerKey", None)

            if not question or not isinstance(ch, dict):
                n_skipped += 1
                continue

            labels = ch.get("label", [])
            texts = ch.get("text", [])
            if not isinstance(labels, list) or not isinstance(texts, list) or len(texts) != 4:
                n_skipped += 1
                continue

            # Ensure choices are in A/B/C/D order (some datasets already are, but make it explicit)
            label_to_text = {str(l).strip().upper(): clean_text(t) for l, t in zip(labels, texts)}
            if not all(k in label_to_text for k in letters):
                n_skipped += 1
                continue

            choices = [label_to_text[l] for l in letters]

            answer_index = answer_to_index(answer_key)
            if answer_index is None or answer_index not in (0, 1, 2, 3):
                n_skipped += 1
                continue

            prompt = build_prompt(question, choices)

            record = CleanMCExample(
                benchmark="openbookqa",
                split="validation",
                id=stable_id("val", i, ex),
                prompt=prompt,
                context=question,
                choices=choices,
                answer_index=answer_index,
                answer_letter=letters[answer_index],
                answer_text=choices[answer_index],
                source={
                    "hf_dataset": hf_dataset,
                    "hf_config": hf_config,
                    "hf_split": hf_split,
                    "id": ex.get("id"),
                    "answerKey": ex.get("answerKey"),
                    "choices_labels": labels,
                },
            )

            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            n_written += 1

    meta = {
        "benchmark": "openbookqa",
        "split": "validation",
        "hf_dataset": hf_dataset,
        "hf_config": hf_config,
        "hf_split": hf_split,
        "rows_in_split": len(ds),
        "written": n_written,
        "skipped": n_skipped,
        "output_file": str(out_path),
        "schema": "CleanMCExample",
        "note": "OpenBookQA is 4-way multiple choice; model is expected to output A/B/C/D.",
    }
    (out_dir / "validation_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[openbookqa] hf_split rows: {len(ds)}")
    print(f"[openbookqa] written: {n_written}, skipped: {n_skipped}")
    print(f"[openbookqa] saved: {out_path}")


if __name__ == "__main__":
    main()