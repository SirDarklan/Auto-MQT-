# Auto-MQT: Learned Token Budgeting for MQT-LLaVA

## Feasibility

Yes, this is doable as a course project if MQT-LLaVA remains frozen and the only trained component is a small routing network. The expensive part is generating oracle labels by running examples at multiple visual-token budgets. Router training itself should be lightweight.

The implementation should be staged:

- Run frozen MQT-LLaVA at a manageable budget set such as `{36, 64, 144, 256}`.
- Label each example with the smallest sufficient budget.
- Train a prompt-only, image-only, and multimodal router.
- Compare routers against fixed-budget baselines and the oracle selector.
- Report quality, average visual tokens, latency, budget regret, and under-budget rate.

## Research Question

Can a tiny router predict how many MQT visual tokens an image-question pair needs before answer generation, preserving most high-budget accuracy while reducing average visual-token usage?

## Method

### 1. Oracle Budget Labeling

For each example `(image, prompt, answer)`, run MQT-LLaVA at every candidate budget. The oracle label is the smallest budget that produces an acceptable answer.

Two definitions are supported:

- Correctness label: smallest budget with a correct answer under the dataset metric.
- Tolerance label: smallest budget within margin `delta` of the full 256-token score.

The exhaustive oracle is not deployable, but it gives a supervised target for the router and an upper bound for analysis.

### 2. Router

The router predicts a categorical distribution over token budgets. The implemented starter router is a late-fusion MLP with three modes:

- `prompt`: prompt features only.
- `image`: image features only.
- `multimodal`: concatenated prompt and image features.

The proposal also mentions a tiny cross-attention router. That is a natural next ablation after the MLP pipeline works end to end.

### 3. Cost-Aware Training

The router uses cross-entropy against the oracle budget plus a normalized expected-token cost:

`loss = CE(oracle_budget, predicted_budget) + lambda_cost * E[token_cost]`

This discourages always choosing the largest budget while still learning from oracle labels.

### 4. Inference

At test time:

1. Extract prompt/image features.
2. Router predicts a budget.
3. If confidence is low, optionally fall back to the next larger budget.
4. Run MQT-LLaVA once at the selected budget.

## Baselines

- Fixed-budget MQT: always use one budget from `{8, 16, 36, 64, 144, 256}`.
- Rule-based task router: keyword/task mapping from the earlier scaffold.
- Prompt-only learned router.
- Image-only learned router.
- Multimodal learned router.
- Oracle selector.

## Datasets

Proposal-aligned datasets:

- VQAv2 for broad visual question answering.
- GQA for spatial and compositional reasoning.
- TextVQA for OCR-heavy questions.
- ScienceQA-IMG for science/reasoning examples with images.

Use subsets first because oracle label generation requires multiple MQT-LLaVA runs per example.

The implemented workflow converts Hugging Face subsets into cached JSONL manifests first, so all later scripts are dataset-agnostic.

## Metrics

- Dataset task score: VQAv2, GQA, TextVQA, or ScienceQA accuracy.
- Average visual tokens.
- End-to-end latency.
- Accuracy at matched average token budgets.
- Budget regret: extra tokens used relative to oracle minimum.
- Under-budget rate: fraction of examples where the router chooses fewer tokens than the oracle label.

## Implemented Files

- `src/oracle_labeling.py`: creates resumable oracle labels by evaluating selected budgets.
- `src/prepare_datasets.py`: converts Hugging Face dataset subsets into local JSONL manifests and cached image files.
- `src/verify_manifest.py`: checks required manifest fields and confirms images are readable.
- `src/router_model.py`: late-fusion MLP router.
- `src/train_router.py`: trains prompt/image/multimodal routers with cost-aware loss.
- `src/evaluate_router.py`: evaluates a trained router.
- `src/evaluate_token_policy.py`: fixed-budget and rule-based baselines.
- `src/extract_router_features.py`: extracts frozen CLIP prompt/image embeddings into JSONL rows.
- `src/analyze_results.py`: aggregates result JSONL files into report-ready metrics.
- `configs/router.yaml`: router and training hyperparameters.
- `configs/datasets.yaml`: Hugging Face dataset registry, splits, subset sizes, and schema mappings.
- `notebooks/mqt.ipynb`: notebook workflow.

## Remaining Model-Specific Work

The adapter can call the cloned MQT-LLaVA repo through `MQT_LLAVA_REPO`. For large oracle-label generation, use the persistent backend (`MQT_LLAVA_BACKEND=persistent`) so the 7B model is loaded once per run. Router features should come from frozen embeddings written by `src/extract_router_features.py`, not hashed fallbacks.
