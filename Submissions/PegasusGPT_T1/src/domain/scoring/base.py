from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ChoiceCandidate:
    letter: str
    text: str
    scoring_prefix: str
    scoring_continuation: str


@dataclass(frozen=True)
class NormalizedMultipleChoicePrompt:
    benchmark: str
    original_prompt: str
    candidates: tuple[ChoiceCandidate, ...]
    scoring_pre_prompt: str = ""


class BenchmarkAdapter(Protocol):
    name: str
    scoring_pre_prompt: str

    def detect(self, prompt: str) -> bool:
        ...

    def normalize(self, prompt: str) -> NormalizedMultipleChoicePrompt:
        ...


def prepend_to_prompt(pre_prompt: str, scoring_prefix: str) -> str:
    pre_prompt = pre_prompt.strip()
    if not pre_prompt:
        return scoring_prefix
    return f"{pre_prompt}\n\n{scoring_prefix}"


def build_scored_text(candidate: ChoiceCandidate, pre_prompt: str = "") -> str:
    return prepend_to_prompt(
        pre_prompt,
        candidate.scoring_prefix,
    ) + candidate.scoring_continuation
