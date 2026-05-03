#!/usr/bin/env python3
"""Leaderboard adapter for PegasusGPT.

Implements the CLI signature expected by leaderboard/run_benchmarks.py and
delegates inference to the project's GPTInferenceService. In --leaderboard
mode, only the generated continuation is written to stdout.
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
# Add the project source root for imports like `from domain...`.
# The leaderboard submission is self-contained under `src/`, while the original
# project layout may have `domain/` directly under an ancestor directory.
SOURCE_ROOT = next(
    (
        p
        for p in (HERE / "src", HERE, *HERE.parents)
        if (p / "domain" / "inference" / "inference_service.py").is_file()
    ),
    HERE / "src",
)
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


# Fallback model hyperparameters used only if the checkpoint does not embed a
# model_config. Mirrors configs/inference.toml for the current run.
DEFAULT_MODEL_CONFIG: dict = dict(
    architecture="llama3",
    vocab_size=50257,
    max_position_embeddings=1024,
    hidden_size=320,
    num_layers=22,
    num_attention_heads=16,
    tie_word_embeddings=True,
    mlp_hidden_size=None,
    qkv_bias=False,
    dropout=0.0,
    n_kv_heads=1,
    intermediate_size=853,
    rope_theta=10000.0,
)


def _silence_third_party_logging() -> None:
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    logging.getLogger().setLevel(logging.ERROR)
    # Windows console defaults to cp1252; tolerate any chars the model emits.
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass


def _seed_all(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(name: str) -> str:
    import torch
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PegasusGPT leaderboard adapter")
    p.add_argument("--stage", default="inference")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-tokens", dest="max_tokens", type=int, required=True)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--leaderboard", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--top-k", dest="top_k", type=int, default=50)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.stage != "inference":
        sys.stderr.write(f"unsupported stage: {args.stage}\n")
        return 2

    _silence_third_party_logging()
    _seed_all(args.seed)

    from domain.inference.inference_service import GPTInferenceService
    from transformers import GPT2TokenizerFast

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = (Path.cwd() / checkpoint_path).resolve()

    service = GPTInferenceService()
    result = service.run(
        checkpoint_path=str(checkpoint_path),
        model_config=SimpleNamespace(**DEFAULT_MODEL_CONFIG),
        input_text=args.prompt,
        max_new_tokens=args.max_tokens,
        device_name=_resolve_device(args.device),
        vocab_size=DEFAULT_MODEL_CONFIG["vocab_size"],
        temperature=args.temperature,
        top_k=args.top_k,
    )

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    continuation_ids = result.generated_token_ids[len(result.input_token_ids):]
    continuation = tokenizer.decode(continuation_ids)

    sys.stdout.write(continuation)
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
