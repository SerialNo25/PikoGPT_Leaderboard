# uv run python benchmarks/preprocessing/preprocess_lambada.py

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

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


def stable_id(split: str, text: str) -> str:
    """Stable ID from SHA1(text)."""
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    return f"lambada_{split}_sha1_{h}"


def split_last_word(text: str) -> Optional[tuple[str, str]]:
    """
    LAMBADA target = last word of the passage.

    Returns (context, last_word) or None if not splittable.
    """
    text = clean_text(text)
    if not text:
        return None

    # rsplit on whitespace once
    parts = text.rsplit(None, 1)
    if len(parts) != 2:
        return None

    context, last_word = parts[0], parts[1]
    if not context or not last_word:
        return None

    return context, last_word


# ----------------------------
# Output schema
# ----------------------------
@dataclass
class CleanNextWordExample:
    benchmark: str
    split: str
    id: str
    prompt: str        # context + trailing space
    context: str       # context only (no trailing space)
    answer_text: str   # gold last word
    source: Dict[str, Any]


def main() -> None:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    out_dir = REPO_ROOT / "benchmarks" / "lambada" / "cleaned"
    out_dir.mkdir(parents=True, exist_ok=True)

    hf_dataset = "EleutherAI/lambada_openai"
    hf_config = "en"   # English subset
    hf_split = "test"  # dataset provides test split with gold last words

    ds = load_dataset(hf_dataset, hf_config, split=hf_split)

    out_path = out_dir / f"{hf_split}.jsonl"

    n_written = 0
    n_skipped = 0

    with out_path.open("w", encoding="utf-8") as f:
        for ex in ds:
            raw_text = ex.get("text", "")
            split_res = split_last_word(raw_text)
            if split_res is None:
                n_skipped += 1
                continue

            context, last_word = split_res

            # prompt = context + trailing space so next token starts as the next word
            prompt = context + " "

            record = CleanNextWordExample(
                benchmark="lambada",
                split=hf_split,
                id=stable_id(hf_split, clean_text(raw_text)),
                prompt=prompt,
                context=context,
                answer_text=last_word,
                source={
                    "hf_dataset": hf_dataset,
                    "hf_config": hf_config,
                    "hf_split": hf_split,
                },
            )

            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            n_written += 1

    meta = {
        "benchmark": "lambada",
        "split": hf_split,
        "hf_dataset": hf_dataset,
        "hf_config": hf_config,
        "hf_split": hf_split,
        "rows_in_split": len(ds),
        "written": n_written,
        "skipped": n_skipped,
        "output_file": str(out_path),
        "schema": "CleanNextWordExample",
        "note": "Prompt is the passage WITHOUT the last word, plus a trailing space. Gold is the last word (answer_text).",
    }
    (out_dir / f"{hf_split}_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[lambada] hf_split rows: {len(ds)}")
    print(f"[lambada] written: {n_written}, skipped: {n_skipped}")
    print(f"[lambada] saved: {out_path}")


if __name__ == "__main__":
    main()