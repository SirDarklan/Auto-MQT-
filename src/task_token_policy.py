from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TaskTokenPolicy:
    default_budget: int
    budgets: dict[str, int]
    keyword_rules: dict[str, list[str]]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TaskTokenPolicy":
        with Path(path).open("r", encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f)
        return cls(
            default_budget=int(raw["default_budget"]),
            budgets={k: int(v) for k, v in raw["budgets"].items()},
            keyword_rules={
                task: [str(term).lower() for term in terms]
                for task, terms in raw.get("keyword_rules", {}).items()
            },
        )

    def classify_prompt(self, prompt: str) -> str:
        text = prompt.lower()
        for task, terms in self.keyword_rules.items():
            if any(term in text for term in terms):
                return task
        return "vqa"

    def budget_for(self, prompt: str, task: str | None = None) -> int:
        chosen_task = task or self.classify_prompt(prompt)
        return self.budgets.get(chosen_task, self.default_budget)
