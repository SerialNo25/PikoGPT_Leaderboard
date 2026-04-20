# 1. HellaSwag Benchmark (Validation Split)
**locally benchmarkable version of the HellaSwag dataset**, preprocessed from the HuggingFace dataset `Rowan/hellaswag`.
HellaSwag is a **commonsense reasoning benchmark** where a model must choose the most plausible continuation of a short real-world scenario.

---

## Dataset overview

**Benchmark:** HellaSwag  
**Source:** HuggingFace – `Rowan/hellaswag`  
**Split used:** `validation`  
**Number of examples:** 10,042  

> The official `test` split does **not** include gold labels and cannot be used for local evaluation.  
> Therefore, the `validation` split is used for all local benchmarking.

---

## Task description

Each example consists of:
- a short **context** describing a real-world situation
- **four candidate endings**
- exactly **one correct ending**

The task is a **4-way multiple-choice sentence completion** problem that tests:
- commonsense reasoning
- physical and temporal plausibility
- understanding of human actions and intentions

---

## Raw dataset structure (validation split)

Each row in the original dataset contains the following fields:

| Field | Type | Description |
|------|-----|-------------|
| `ind` | int | Example index |
| `activity_label` | string | Coarse activity category |
| `ctx_a` | string | Main context description |
| `ctx_b` | string | Context continuation prefix |
| `ctx` | string | Combined context (`ctx_a + ctx_b`) |
| `endings` | list[string] | Four candidate endings |
| `source_id` | string | Original video/source identifier |
| `split` | string | Dataset split (`val`) |
| `split_type` | string | `indomain` or `zeroshot` |
| `label` | string | Correct ending index (`"0"`–`"3"`) |

### Label distribution (validation)
Labels are approximately uniform:
0: 2505
1: 2475
2: 2573
3: 2447

---

## Preprocessing (`preprocess_hellaswag.py`)

The script `benchmarks/preprocessing/preprocess_hellaswag.py` converts the raw HuggingFace dataset into a **clean, model-agnostic JSONL format** suitable for local benchmarking.

For each example in the validation split:

1. **Loads the dataset**
   - Uses `load_dataset("Rowan/hellaswag", split="validation")`

2. **Cleans text**
   - Strips whitespace
   - Normalizes newlines
   - Collapses repeated spaces

3. **Builds a standardized prompt**
   ```text
   Context: <context>
   A) <choice 0>
   B) <choice 1>
   C) <choice 2>
   D) <choice 3>
   Answer:

    ```
4.	**Extracts gold answers**
- Converts label → answer_index 
- Maps to answer_letter (A–D)
- Stores the correct answer text

5.	**Creates stable example IDs**
- Uses source_id and ind when available

6.	**Writes output files**
- validation.jsonl: one cleaned example per line
- validation_meta.json: metadata and preprocessing info

## Processed Output format
Each line in validation.jsonl has the following structure:
```json
{
  "benchmark": "hellaswag",
  "split": "validation",
  "id": "hellaswag_val_<source_id>_<ind>",
  "prompt": "...",
  "context": "...",
  "choices": [
    "...",
    "...",
    "...",
    "..."
  ],
  "answer_index": 3,
  "answer_letter": "D",
  "answer_text": "...",
  "source": {
    "hf_dataset": "Rowan/hellaswag",
    "hf_split": "validation",
    "activity_label": "...",
    "split_type": "indomain"
  }
}
```

