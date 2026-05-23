from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

from data_utils import feature_from_example, filter_rows_by_dataset, load_jsonl, parse_dataset_names
from router_model import LateFusionRouter, RouterConfig, budget_to_index, predict_budget


def soft_target_distribution(
    row: dict[str, Any],
    budgets: list[int],
    oracle_index: int,
    temperature: float,
    cost_weight: float,
) -> list[float]:
    if temperature <= 0:
        temperature = 1.0

    default = [0.0 for _ in budgets]
    default[oracle_index] = 1.0
    budget_results = row.get("budget_results")
    if not isinstance(budget_results, list) or not budget_results:
        return default

    score_key = str(row.get("oracle_score_key", "relaxed_match"))
    score_by_budget: dict[int, float] = {}
    for item in budget_results:
        if not isinstance(item, dict) or "visual_tokens" not in item:
            continue
        budget = int(item["visual_tokens"])
        if budget not in budgets:
            continue
        if score_key in item:
            score = float(item[score_key])
        elif "score" in item:
            score = float(item["score"])
        elif "relaxed_match" in item:
            score = float(item["relaxed_match"])
        elif "exact_match" in item:
            score = float(item["exact_match"])
        else:
            score = 0.0
        score_by_budget[budget] = score

    if not score_by_budget:
        return default

    max_budget = float(max(budgets))
    values: list[float] = []
    for budget in budgets:
        score = score_by_budget.get(int(budget), 0.0)
        normalized_cost = float(budget) / max_budget
        utility = (score / temperature) - cost_weight * normalized_cost
        values.append(utility)

    values_tensor = torch.tensor(values, dtype=torch.float32)
    probs = torch.softmax(values_tensor, dim=-1).tolist()
    return [float(v) for v in probs]


def score_vector_from_budget_results(row: dict[str, Any], budgets: list[int]) -> list[float]:
    score_key = str(row.get("oracle_score_key", "dataset_score"))
    budget_results = row.get("budget_results")
    if not isinstance(budget_results, list):
        return [0.0 for _ in budgets]

    score_by_budget: dict[int, float] = {}
    for item in budget_results:
        if not isinstance(item, dict) or "visual_tokens" not in item:
            continue
        budget = int(item["visual_tokens"])
        if score_key in item:
            score = float(item[score_key])
        elif "score" in item:
            score = float(item["score"])
        elif "dataset_score" in item:
            score = float(item["dataset_score"])
        elif "relaxed_match" in item:
            score = float(item["relaxed_match"])
        elif "exact_match" in item:
            score = float(item["exact_match"])
        else:
            score = 0.0
        score_by_budget[budget] = score
    return [float(score_by_budget.get(int(budget), 0.0)) for budget in budgets]


class OracleBudgetDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        config: RouterConfig,
        soft_target_temperature: float,
        soft_target_cost_weight: float,
    ) -> None:
        self.rows = rows
        self.config = config
        self.label_indices = [budget_to_index(row["oracle_budget"], config.budgets) for row in rows]
        self.oracle_budgets = [int(row["oracle_budget"]) for row in rows]
        self.score_vectors = [score_vector_from_budget_results(row, config.budgets) for row in rows]
        self.soft_targets = [
            soft_target_distribution(
                row=row,
                budgets=config.budgets,
                oracle_index=self.label_indices[index],
                temperature=soft_target_temperature,
                cost_weight=soft_target_cost_weight,
            )
            for index, row in enumerate(rows)
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        prompt_feature = feature_from_example(
            row,
            field="prompt_embedding",
            fallback_text=row["prompt"],
            dim=self.config.prompt_dim,
        )
        image_feature = feature_from_example(
            row,
            field="image_embedding",
            fallback_text=row["image"],
            dim=self.config.image_dim,
        )
        label = self.label_indices[index]
        oracle_budget = self.oracle_budgets[index]
        return {
            "prompt_features": torch.tensor(prompt_feature, dtype=torch.float32),
            "image_features": torch.tensor(image_feature, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
            "oracle_budget": torch.tensor(float(oracle_budget), dtype=torch.float32),
            "soft_target": torch.tensor(self.soft_targets[index], dtype=torch.float32),
            "score_vector": torch.tensor(self.score_vectors[index], dtype=torch.float32),
        }


def load_router_config(path: str | Path) -> tuple[RouterConfig, dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    config = RouterConfig(
        budgets=[int(v) for v in raw["budgets"]],
        prompt_dim=int(raw["prompt_dim"]),
        image_dim=int(raw["image_dim"]),
        hidden_dim=int(raw["hidden_dim"]),
        dropout=float(raw["dropout"]),
        confidence_fallback_threshold=float(raw.get("confidence_fallback_threshold", 0.0)),
        cross_attn_heads=int(raw.get("cross_attn_heads", 4)),
        cross_attn_query_tokens=int(raw.get("cross_attn_query_tokens", 4)),
        cross_attn_layers=int(raw.get("cross_attn_layers", 1)),
    )
    return config, raw


def infer_embedding_dims(rows: list[dict[str, Any]], config: RouterConfig) -> RouterConfig:
    prompt_dim = config.prompt_dim
    image_dim = config.image_dim

    for row in rows:
        embedding = row.get("prompt_embedding")
        if isinstance(embedding, list) and embedding:
            prompt_dim = int(len(embedding))
            break

    for row in rows:
        embedding = row.get("image_embedding")
        if isinstance(embedding, list) and embedding:
            image_dim = int(len(embedding))
            break

    if prompt_dim == config.prompt_dim and image_dim == config.image_dim:
        return config

    print(
        "override dims from labels:"
        f" prompt_dim {config.prompt_dim} -> {prompt_dim},"
        f" image_dim {config.image_dim} -> {image_dim}"
    )
    return RouterConfig(
        budgets=config.budgets,
        prompt_dim=prompt_dim,
        image_dim=image_dim,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        confidence_fallback_threshold=config.confidence_fallback_threshold,
        cross_attn_heads=config.cross_attn_heads,
        cross_attn_query_tokens=config.cross_attn_query_tokens,
        cross_attn_layers=config.cross_attn_layers,
    )


def stratified_split_indices(
    labels: list[int],
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    rng = torch.Generator().manual_seed(seed)
    indices_by_label: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        indices_by_label[int(label)].append(idx)

    train_indices: list[int] = []
    val_indices: list[int] = []
    for label, indices in sorted(indices_by_label.items()):
        if len(indices) == 1:
            train_indices.extend(indices)
            continue

        permutation = torch.randperm(len(indices), generator=rng).tolist()
        shuffled = [indices[i] for i in permutation]

        val_count = max(1, int(round(len(shuffled) * val_fraction)))
        if val_count >= len(shuffled):
            val_count = len(shuffled) - 1

        val_indices.extend(shuffled[:val_count])
        train_indices.extend(shuffled[val_count:])

    if not val_indices and train_indices:
        val_indices.append(train_indices.pop())

    return sorted(train_indices), sorted(val_indices)


def class_counts(labels: list[int], num_classes: int) -> list[int]:
    counter = Counter(labels)
    return [int(counter.get(class_index, 0)) for class_index in range(num_classes)]


def build_class_weights(
    counts: list[int],
    device: torch.device,
    mode: str,
    effective_beta: float,
) -> torch.Tensor | None:
    if mode == "none":
        return None

    counts_tensor = torch.tensor(counts, dtype=torch.float32, device=device)
    eps = 1e-8
    nonzero_mask = counts_tensor > 0

    if mode == "inverse":
        weights = torch.zeros_like(counts_tensor)
        weights[nonzero_mask] = 1.0 / counts_tensor[nonzero_mask]
    elif mode == "effective":
        beta = float(effective_beta)
        if beta <= 0.0 or beta >= 1.0:
            raise ValueError(f"class_balance_effective_beta must be in (0, 1), got {beta}")
        weights = torch.zeros_like(counts_tensor)
        numerator = 1.0 - beta
        weights[nonzero_mask] = numerator / (1.0 - torch.pow(beta, counts_tensor[nonzero_mask]) + eps)
    else:
        raise ValueError(f"Unknown class balance mode: {mode}")

    weights_sum = weights[nonzero_mask].sum()
    if weights_sum <= 0:
        return None
    weights = weights / (weights_sum / max(1, int(nonzero_mask.sum().item())))
    return weights


def build_weighted_sampler(labels: list[int], mode: str) -> WeightedRandomSampler | None:
    if mode == "none":
        return None

    counts = Counter(labels)
    sample_weights = [1.0 / float(counts[label]) for label in labels]
    weight_tensor = torch.tensor(sample_weights, dtype=torch.double)
    return WeightedRandomSampler(weight_tensor, num_samples=len(weight_tensor), replacement=True)


def cost_aware_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    soft_targets: torch.Tensor,
    score_vectors: torch.Tensor,
    budgets: list[int],
    lambda_cost: float,
    cost_power: float,
    class_weights: torch.Tensor | None,
    label_smoothing: float,
    focal_gamma: float,
    soft_target_alpha: float,
    hard_label_alpha: float,
    lambda_utility: float,
    gamma_calibration: float,
) -> torch.Tensor:
    ce_per = nn.functional.cross_entropy(
        logits,
        labels,
        weight=class_weights,
        reduction="none",
        label_smoothing=label_smoothing,
    )
    if focal_gamma > 0.0:
        base_ce = nn.functional.cross_entropy(
            logits,
            labels,
            reduction="none",
            label_smoothing=label_smoothing,
        )
        p_t = torch.exp(-base_ce)
        focal_weight = (1.0 - p_t).pow(focal_gamma)
        ce = (focal_weight * ce_per).mean()
    else:
        ce = ce_per.mean()

    soft_target_alpha = max(0.0, min(1.0, float(soft_target_alpha)))
    if soft_target_alpha > 0.0:
        log_probs = torch.log_softmax(logits, dim=-1)
        soft_ce = -(soft_targets * log_probs).sum(dim=-1).mean()
        ce = (1.0 - soft_target_alpha) * ce + soft_target_alpha * soft_ce

    probs = torch.softmax(logits, dim=-1)
    budget_tensor = torch.tensor(budgets, dtype=torch.float32, device=logits.device)
    normalized_cost = (budget_tensor / budget_tensor.max()).pow(cost_power)
    expected_cost = (probs * normalized_cost).sum(dim=-1).mean()
    expected_score = (probs * score_vectors).sum(dim=-1).mean()

    hard_label_alpha = max(0.0, min(1.0, float(hard_label_alpha)))
    lambda_utility = max(0.0, float(lambda_utility))
    combined = hard_label_alpha * ce
    combined = combined + lambda_cost * expected_cost - lambda_utility * expected_score

    one_hot = torch.nn.functional.one_hot(labels, num_classes=logits.size(-1)).to(probs.dtype)
    calibration = ((probs - one_hot) ** 2).sum(dim=-1).mean()
    return combined + gamma_calibration * calibration


def evaluate_epoch(
    model: LateFusionRouter,
    loader: DataLoader,
    device: torch.device,
    eval_score_weight: float,
    eval_token_penalty: float,
    eval_under_penalty: float,
    eval_regret_penalty: float,
) -> dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    predicted_budgets: list[int] = []
    oracle_budgets: list[int] = []
    selected_scores: list[float] = []

    with torch.no_grad():
        for batch in loader:
            prompt = batch["prompt_features"].to(device)
            image = batch["image_features"].to(device)
            labels = batch["label"].to(device)
            score_vectors = batch["score_vector"]
            logits = model(prompt, image)
            pred = logits.argmax(dim=-1)
            correct += int((pred == labels).sum().item())
            total += int(labels.numel())
            pred_indices = pred.cpu().tolist()
            for row_index, idx in enumerate(pred_indices):
                predicted_budgets.append(model.config.budgets[idx])
                selected_scores.append(float(score_vectors[row_index, idx].item()))
            oracle_budgets.extend(int(v) for v in batch["oracle_budget"].cpu().tolist())

    regret = mean(max(0, pred - oracle) for pred, oracle in zip(predicted_budgets, oracle_budgets))
    under_budget_rate = mean(float(pred < oracle) for pred, oracle in zip(predicted_budgets, oracle_budgets))
    avg_budget = mean(predicted_budgets) if predicted_budgets else 0.0
    avg_score = mean(selected_scores) if selected_scores else 0.0
    max_budget = float(max(model.config.budgets))
    objective = (
        eval_score_weight * avg_score
        - eval_token_penalty * (avg_budget / max_budget)
        - eval_under_penalty * under_budget_rate
        - eval_regret_penalty * (regret / max_budget)
    )
    return {
        "accuracy": correct / max(1, total),
        "avg_predicted_budget": avg_budget,
        "avg_selected_score": avg_score,
        "budget_regret": regret,
        "under_budget_rate": under_budget_rate,
        "objective": objective,
    }


def collect_val_logits(
    model: LateFusionRouter,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, list[int], list[int], torch.Tensor]:
    model.eval()
    all_logits: list[torch.Tensor] = []
    label_indices: list[int] = []
    oracle_budgets: list[int] = []
    score_vectors: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            prompt = batch["prompt_features"].to(device)
            image = batch["image_features"].to(device)
            logits = model(prompt, image).cpu()
            all_logits.append(logits)
            label_indices.extend(int(v) for v in batch["label"].cpu().tolist())
            oracle_budgets.extend(int(v) for v in batch["oracle_budget"].cpu().tolist())
            score_vectors.append(batch["score_vector"].cpu())

    if not all_logits:
        return torch.empty((0, len(model.config.budgets))), [], [], torch.empty((0, len(model.config.budgets)))
    return torch.cat(all_logits, dim=0), label_indices, oracle_budgets, torch.cat(score_vectors, dim=0)


def tune_fallback_threshold(
    logits: torch.Tensor,
    oracle_budgets: list[int],
    score_vectors: torch.Tensor,
    budgets: list[int],
    thresholds: list[float],
    cost_bias_grid: list[float],
    cost_power: float,
    score_weight: float,
    token_penalty: float,
    under_penalty: float,
    regret_penalty: float,
    target_avg_budget: float | None = None,
    target_budget_penalty: float = 0.0,
) -> tuple[float, dict[str, float]]:
    if logits.numel() == 0:
        return 0.0, {
            "cost_bias": 0.0,
            "accuracy": 0.0,
            "avg_selected_score": 0.0,
            "avg_predicted_budget": 0.0,
            "under_budget_rate": 0.0,
            "budget_regret": 0.0,
            "objective": 0.0,
        }

    max_budget = float(max(budgets))
    best_threshold = 0.0
    best_bias = 0.0
    best_metrics: dict[str, float] | None = None
    best_objective = float("-inf")

    for threshold in thresholds:
        for cost_bias in cost_bias_grid:
            chosen_budgets: list[int] = []
            chosen_indices: list[int] = []
            selected_scores: list[float] = []
            for row_index, row in enumerate(logits):
                budget, _ = predict_budget(
                    row,
                    budgets,
                    threshold,
                    cost_bias=float(cost_bias),
                    cost_power=cost_power,
                )
                chosen_budgets.append(int(budget))
                pred_index = budgets.index(int(budget))
                chosen_indices.append(pred_index)
                selected_scores.append(float(score_vectors[row_index, pred_index].item()))

            avg_score = mean(selected_scores)
            avg_budget = mean(chosen_budgets)
            under_rate = mean(float(pred < oracle) for pred, oracle in zip(chosen_budgets, oracle_budgets))
            regret = mean(float(max(0, pred - oracle)) for pred, oracle in zip(chosen_budgets, oracle_budgets))
            objective = (
                score_weight * avg_score
                - token_penalty * (avg_budget / max_budget)
                - under_penalty * under_rate
                - regret_penalty * (regret / max_budget)
            )
            if target_avg_budget is not None and target_budget_penalty > 0.0:
                objective -= target_budget_penalty * (abs(avg_budget - float(target_avg_budget)) / max_budget)

            if objective > best_objective:
                best_objective = objective
                best_threshold = float(threshold)
                best_bias = float(cost_bias)
                best_metrics = {
                    "cost_bias": float(cost_bias),
                    "accuracy": 0.0,
                    "avg_selected_score": float(avg_score),
                    "avg_predicted_budget": float(avg_budget),
                    "under_budget_rate": float(under_rate),
                    "budget_regret": float(regret),
                    "objective": float(objective),
                }

    assert best_metrics is not None
    best_metrics["cost_bias"] = float(best_bias)
    return best_threshold, best_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, help="Oracle-label JSONL from oracle_labeling.py")
    parser.add_argument("--config", default="configs/router.yaml")
    parser.add_argument(
        "--mode",
        choices=["prompt", "image", "multimodal", "cross_attention"],
        default="multimodal",
    )
    parser.add_argument("--out", default="checkpoints/router_multimodal.pt")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--include-datasets",
        nargs="+",
        default=None,
        help="Optional dataset allow-list (space/comma separated)",
    )
    parser.add_argument(
        "--exclude-datasets",
        nargs="+",
        default=None,
        help="Optional dataset deny-list (space/comma separated), e.g. textvqa",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    config, raw_config = load_router_config(args.config)
    include_datasets = parse_dataset_names(args.include_datasets)
    exclude_datasets = parse_dataset_names(args.exclude_datasets)
    rows = filter_rows_by_dataset(
        load_jsonl(args.labels),
        include_datasets=include_datasets,
        exclude_datasets=exclude_datasets,
    )
    print(f"rows_after_dataset_filter: {len(rows)}")
    if include_datasets is not None:
        print(f"include_datasets: {sorted(include_datasets)}")
    if exclude_datasets is not None:
        print(f"exclude_datasets: {sorted(exclude_datasets)}")
    config = infer_embedding_dims(rows, config)
    dataset = OracleBudgetDataset(
        rows,
        config,
        soft_target_temperature=float(raw_config.get("soft_target_temperature", 0.35)),
        soft_target_cost_weight=float(raw_config.get("soft_target_cost_weight", 0.20)),
    )

    if len(dataset) < 2:
        raise ValueError("Need at least two oracle-labeled examples to train/validate a router.")

    train_indices, val_indices = stratified_split_indices(
        dataset.label_indices,
        val_fraction=float(args.val_fraction),
        seed=int(args.seed),
    )
    train_data = Subset(dataset, train_indices)
    val_data = Subset(dataset, val_indices)

    train_labels = [dataset.label_indices[index] for index in train_indices]
    val_labels = [dataset.label_indices[index] for index in val_indices]
    counts = class_counts(train_labels, num_classes=len(config.budgets))
    print(f"class_counts_train: {counts}")
    print(f"class_counts_val: {class_counts(val_labels, num_classes=len(config.budgets))}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = build_class_weights(
        counts=counts,
        device=device,
        mode=str(raw_config.get("class_balance_mode", "inverse")),
        effective_beta=float(raw_config.get("class_balance_effective_beta", 0.999)),
    )
    if class_weights is not None:
        print(f"class_weights: {[round(float(v), 4) for v in class_weights.cpu().tolist()]}")

    weighted_sampler = build_weighted_sampler(
        labels=train_labels,
        mode=str(raw_config.get("sampler_mode", "inverse")),
    )

    loader_kwargs = {
        "batch_size": int(raw_config["batch_size"]),
        "num_workers": int(raw_config.get("num_workers", 0)),
    }
    if weighted_sampler is None:
        train_loader = DataLoader(train_data, shuffle=True, **loader_kwargs)
    else:
        train_loader = DataLoader(train_data, sampler=weighted_sampler, **loader_kwargs)
    val_loader = DataLoader(val_data, shuffle=False, **loader_kwargs)

    model = LateFusionRouter(config, mode=args.mode).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(raw_config["learning_rate"]),
        weight_decay=float(raw_config["weight_decay"]),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    label_smoothing = float(raw_config.get("label_smoothing", 0.05))
    focal_gamma = float(raw_config.get("focal_gamma", 1.0))
    max_patience = int(raw_config.get("early_stop_patience", 6))
    min_delta = float(raw_config.get("early_stop_min_delta", 1e-4))
    hard_label_alpha = float(raw_config.get("hard_label_alpha", 0.40))
    lambda_utility = float(raw_config.get("lambda_utility", 1.0))
    selection_metric = str(raw_config.get("selection_metric", "objective"))
    selection_metric = selection_metric if selection_metric in {"accuracy", "objective"} else "objective"
    eval_score_weight = float(raw_config.get("eval_score_weight", 1.0))
    eval_token_penalty = float(raw_config.get("eval_token_penalty", 0.04))
    eval_under_penalty = float(raw_config.get("eval_under_penalty", 0.10))
    eval_regret_penalty = float(raw_config.get("eval_regret_penalty", 0.05))
    cost_power = float(raw_config["cost_power"])

    best_val = float("-inf")
    best_epoch = 0
    best_state = None
    patience = 0

    for epoch in range(1, int(raw_config["epochs"]) + 1):
        model.train()
        losses = []
        for batch in train_loader:
            prompt = batch["prompt_features"].to(device)
            image = batch["image_features"].to(device)
            labels = batch["label"].to(device)
            soft_targets = batch["soft_target"].to(device)
            score_vectors = batch["score_vector"].to(device)
            logits = model(prompt, image)
            loss = cost_aware_loss(
                logits=logits,
                labels=labels,
                soft_targets=soft_targets,
                score_vectors=score_vectors,
                budgets=config.budgets,
                lambda_cost=float(raw_config["lambda_cost"]),
                cost_power=cost_power,
                class_weights=class_weights,
                label_smoothing=label_smoothing,
                focal_gamma=focal_gamma,
                soft_target_alpha=float(raw_config.get("soft_target_alpha", 0.35)),
                hard_label_alpha=hard_label_alpha,
                lambda_utility=lambda_utility,
                gamma_calibration=float(raw_config.get("gamma_calibration", 0.05)),
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(raw_config.get("grad_clip_norm", 1.0)))
            optimizer.step()
            losses.append(float(loss.item()))

        metrics = evaluate_epoch(
            model,
            val_loader,
            device,
            eval_score_weight=eval_score_weight,
            eval_token_penalty=eval_token_penalty,
            eval_under_penalty=eval_under_penalty,
            eval_regret_penalty=eval_regret_penalty,
        )
        print(
            f"epoch={epoch:02d} objective_loss={mean(losses):.4f} "
            f"val_acc={metrics['accuracy']:.4f} "
            f"val_score={metrics['avg_selected_score']:.4f} "
            f"avg_budget={metrics['avg_predicted_budget']:.2f} "
            f"regret={metrics['budget_regret']:.2f} "
            f"under={metrics['under_budget_rate']:.4f} "
            f"objective={metrics['objective']:.4f}"
        )

        current_metric = float(metrics[selection_metric])
        improved = current_metric > (best_val + min_delta)
        if improved:
            best_val = current_metric
            best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= max_patience:
                print(f"early_stop at epoch={epoch}")
                break

    if best_state is None:
        best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        best_epoch = 1

    model.load_state_dict(best_state)
    val_logits, _, val_oracle_budgets, val_score_vectors = collect_val_logits(model, val_loader, device)

    threshold_grid = [float(v) for v in raw_config.get("fallback_threshold_grid", [0.0, 0.45, 0.50, 0.55, 0.60])]
    cost_bias_grid = [float(v) for v in raw_config.get("cost_bias_grid", [0.0, 0.05, 0.10, 0.15, 0.20])]
    tuned_threshold, tuned_metrics = tune_fallback_threshold(
        logits=val_logits,
        oracle_budgets=val_oracle_budgets,
        score_vectors=val_score_vectors,
        budgets=config.budgets,
        thresholds=threshold_grid,
        cost_bias_grid=cost_bias_grid,
        cost_power=cost_power,
        score_weight=float(raw_config.get("threshold_score_weight", 1.0)),
        token_penalty=float(raw_config.get("threshold_token_penalty", 0.04)),
        under_penalty=float(raw_config.get("threshold_under_penalty", 0.10)),
        regret_penalty=float(raw_config.get("threshold_regret_penalty", 0.05)),
        target_avg_budget=(
            float(raw_config["target_avg_budget"])
            if raw_config.get("target_avg_budget") is not None
            else None
        ),
        target_budget_penalty=float(raw_config.get("target_budget_penalty", 0.0)),
    )
    print(
        "tuned_fallback:"
        f" threshold={tuned_threshold:.2f}"
        f" cost_bias={tuned_metrics['cost_bias']:.3f}"
        f" val_score={tuned_metrics['avg_selected_score']:.4f}"
        f" avg_budget={tuned_metrics['avg_predicted_budget']:.2f}"
        f" under={tuned_metrics['under_budget_rate']:.4f}"
        f" regret={tuned_metrics['budget_regret']:.2f}"
        f" objective={tuned_metrics['objective']:.4f}"
    )

    torch.save(
        {
            "model_state": best_state,
            "router_config": config.__dict__,
            "raw_config": raw_config,
            "mode": args.mode,
            "best_val_accuracy": float(best_val),
            "best_val_metric": float(best_val),
            "selection_metric": selection_metric,
            "best_epoch": int(best_epoch),
            "tuned_fallback_threshold": float(tuned_threshold),
            "tuned_cost_bias": float(tuned_metrics.get("cost_bias", 0.0)),
            "tuned_fallback_metrics": tuned_metrics,
            "train_class_counts": counts,
            "include_datasets": sorted(include_datasets) if include_datasets is not None else None,
            "exclude_datasets": sorted(exclude_datasets) if exclude_datasets is not None else None,
        },
        out_path,
    )
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
