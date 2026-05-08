from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import GPT2TokenizerFast

from domain.inference.inference_service import GPTInferenceService
from domain.scoring.base import NormalizedMultipleChoicePrompt
from domain.scoring.multiple_choice_scorer import ChoiceScore, MultipleChoiceScorer
from domain.scoring.registry import BenchmarkRegistry, default_registry


@dataclass(frozen=True)
class MultipleChoiceScoringResult:
    benchmark: str
    reply: str
    scores: tuple[ChoiceScore, ...]


class MultipleChoiceScoringService:
    def __init__(self, registry: BenchmarkRegistry | None = None) -> None:
        self.registry = registry or default_registry()

    def detect_and_normalize(self, prompt: str) -> NormalizedMultipleChoicePrompt | None:
        adapter = self.registry.detect(prompt)
        if adapter is None:
            return None
        return adapter.normalize(prompt)

    def run(
        self,
        *,
        checkpoint_path: str,
        model_config,
        input_text: str,
        device_name: str,
        vocab_size: int | None = None,
    ) -> MultipleChoiceScoringResult | None:
        normalized = self.detect_and_normalize(input_text)
        if normalized is None:
            return None

        inference_service = GPTInferenceService()
        model, built_config = inference_service.load_model(
            checkpoint_path=checkpoint_path,
            model_config=model_config,
            vocab_size=vocab_size,
        )

        model.to(device_name)
        was_training = model.training
        model.eval()
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        scorer = MultipleChoiceScorer(
            model=model,
            tokenizer=tokenizer,
            device_name=device_name,
            max_position_embeddings=built_config.max_position_embeddings,
        )

        try:
            with torch.no_grad():
                scores = tuple(scorer.score(candidate) for candidate in normalized.candidates)
        finally:
            if was_training:
                model.train()

        best = max(scores, key=lambda score: score.log_likelihood)
        return MultipleChoiceScoringResult(
            benchmark=normalized.benchmark,
            reply=best.letter,
            scores=scores,
        )
