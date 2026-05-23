from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class RouterConfig:
    budgets: list[int]
    prompt_dim: int
    image_dim: int
    hidden_dim: int
    dropout: float
    confidence_fallback_threshold: float = 0.0
    cross_attn_heads: int = 4
    cross_attn_query_tokens: int = 4
    cross_attn_layers: int = 1


class LateFusionRouter(nn.Module):
    def __init__(self, config: RouterConfig, mode: str = "multimodal") -> None:
        super().__init__()
        if mode not in {"prompt", "image", "multimodal", "cross_attention"}:
            raise ValueError(f"Unknown router mode: {mode}")
        self.config = config
        self.mode = mode

        if mode == "cross_attention":
            self.prompt_proj = nn.Linear(config.prompt_dim, config.hidden_dim)
            self.image_proj = nn.Linear(config.image_dim, config.hidden_dim)
            self.query_tokens = nn.Parameter(
                torch.randn(config.cross_attn_query_tokens, config.hidden_dim) * 0.02
            )
            self.cross_blocks = nn.ModuleList(
                [
                    nn.MultiheadAttention(
                        embed_dim=config.hidden_dim,
                        num_heads=config.cross_attn_heads,
                        dropout=config.dropout,
                        batch_first=True,
                    )
                    for _ in range(config.cross_attn_layers)
                ]
            )
            self.cross_norms = nn.ModuleList([nn.LayerNorm(config.hidden_dim) for _ in range(config.cross_attn_layers)])
            self.net = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, len(config.budgets)),
            )
            return

        input_dim = 0
        if mode in {"prompt", "multimodal"}:
            input_dim += config.prompt_dim
        if mode in {"image", "multimodal"}:
            input_dim += config.image_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, len(config.budgets)),
        )

    def forward(self, prompt_features: torch.Tensor, image_features: torch.Tensor) -> torch.Tensor:
        if self.mode == "cross_attention":
            prompt_token = self.prompt_proj(prompt_features)
            image_token = self.image_proj(image_features)
            modal_tokens = torch.stack([prompt_token, image_token], dim=1)
            queries = self.query_tokens.unsqueeze(0).expand(prompt_features.size(0), -1, -1)
            for attention, norm in zip(self.cross_blocks, self.cross_norms):
                attended, _ = attention(queries, modal_tokens, modal_tokens, need_weights=False)
                queries = norm(queries + attended)
            fused = queries.mean(dim=1)
            return self.net(fused)

        pieces = []
        if self.mode in {"prompt", "multimodal"}:
            pieces.append(prompt_features)
        if self.mode in {"image", "multimodal"}:
            pieces.append(image_features)
        return self.net(torch.cat(pieces, dim=-1))


def budget_to_index(budget: int, budgets: list[int]) -> int:
    try:
        return budgets.index(int(budget))
    except ValueError as exc:
        raise ValueError(f"Budget {budget} not in configured budgets {budgets}") from exc


def predict_budget(
    logits: torch.Tensor,
    budgets: list[int],
    confidence_fallback_threshold: float = 0.0,
    cost_bias: float = 0.0,
    cost_power: float = 1.0,
) -> tuple[int, float]:
    adjusted_logits = logits
    if cost_bias != 0.0:
        budget_tensor = torch.tensor(budgets, dtype=logits.dtype, device=logits.device)
        normalized_cost = (budget_tensor / budget_tensor.max()).pow(cost_power)
        adjusted_logits = logits - float(cost_bias) * normalized_cost

    probs = torch.softmax(adjusted_logits, dim=-1)
    confidence, index = torch.max(probs, dim=-1)
    chosen_index = int(index.item())
    if confidence_fallback_threshold and confidence.item() < confidence_fallback_threshold:
        chosen_index = min(chosen_index + 1, len(budgets) - 1)
    return int(budgets[chosen_index]), float(confidence.item())
