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


class BenchmarkAdapter(Protocol):
    name: str

    def detect(self, prompt: str) -> bool:
        ...

    def normalize(self, prompt: str) -> NormalizedMultipleChoicePrompt:
        ...
