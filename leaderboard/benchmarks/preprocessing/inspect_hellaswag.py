# uv run python benchmarks/preprocessing/inspect_hellaswag.py

from __future__ import annotations

import json
from collections import Counter

from datasets import load_dataset


def preview_row(ex: dict, max_len: int = 200) -> dict:
    """Make a row readable in terminal by truncating long strings."""
    out = {}
    for k, v in ex.items():
        if isinstance(v, str):
            out[k] = v[:max_len] + ("..." if len(v) > max_len else "")
        elif isinstance(v, list):
            # truncate list elements if they are strings
            vv = []
            for item in v[:10]:
                if isinstance(item, str):
                    vv.append(item[:max_len] + ("..." if len(item) > max_len else ""))
                else:
                    vv.append(item)
            out[k] = vv + (["..."] if len(v) > 10 else [])
        else:
            out[k] = v
    return out


def main() -> None:
    ds = load_dataset("Rowan/hellaswag", split="validation")

    print("============================================================")
    print("HellaSwag (Rowan/hellaswag) — split: validation")
    print("============================================================\n")

    # Dataset features (schema)
    print("---- Features / schema ----")
    print(ds.features)
    print()

    # Column names
    print("---- Columns ----")
    print(ds.column_names)
    print()

    # Size
    print("---- Size ----")
    print("num_rows:", len(ds))
    print()

    # Show a couple raw examples
    print("---- Raw examples (first 2) ----")
    for i in range(2):
        ex = ds[i]
        print(f"\nExample {i}:")
        print(json.dumps(preview_row(ex), indent=2, ensure_ascii=False))

    # Check label distribution (if present)
    if "label" in ds.column_names:
        print("\n---- Label distribution (first 10k rows or full if smaller) ----")
        n = min(len(ds), 10_000)
        labels = []
        for i in range(n):
            labels.append(ds[i]["label"])
        cnt = Counter(labels)
        print("count:", dict(cnt))
        print("note: labels might be strings or ints depending on dataset version")
    print()

    # Sanity checks of fields you plan to use
    print("---- Field sanity checks (first 100 rows) ----")
    n = min(len(ds), 100)
    missing_ctx = 0
    wrong_endings_len = 0
    bad_label = 0

    for i in range(n):
        ex = ds[i]
        ctx = ex.get("ctx", "")
        endings = ex.get("endings", [])
        label = ex.get("label", None)

        if not isinstance(ctx, str) or not ctx.strip():
            missing_ctx += 1

        if not isinstance(endings, list) or len(endings) != 4:
            wrong_endings_len += 1

        try:
            li = int(label)
            if li < 0 or li > 3:
                bad_label += 1
        except Exception:
            bad_label += 1

    print(f"missing/empty ctx: {missing_ctx}/{n}")
    print(f"endings not len 4: {wrong_endings_len}/{n}")
    print(f"label not int in [0..3]: {bad_label}/{n}")
    print("\nDone.")


if __name__ == "__main__":
    main()