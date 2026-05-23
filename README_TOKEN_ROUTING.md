# Auto-MQT Token Routing Starter

This scaffold implements the proposal workflow for Auto-MQT: generate oracle token-budget labels, train a small router, and compare adaptive routing against fixed-budget MQT-LLaVA baselines.

## Data Format

The project uses JSONL manifests as the canonical cache, even when the raw data comes from Hugging Face. Each row contains normalized fields:

```json
{"dataset": "textvqa", "split": "train", "example_id": "textvqa_train_12", "image": "data/images/textvqa/train/textvqa_train_12.jpg", "prompt": "what word is written on the sign?", "answer": "stop", "answers": ["stop"], "task": "ocr"}
```

The `task` field is optional. If omitted, the rule-based baseline in `src/task_token_policy.py` uses keyword rules from `configs/task_token_policy.yaml`.

Router training expects oracle-labeled JSONL produced by `src/oracle_labeling.py`. For report-quality results, first add frozen prompt/image embeddings with `src/extract_router_features.py` (recommended). If embeddings are missing, training still runs with deterministic hashed fallback features so pipeline smoke tests can proceed.

## Prepare Hugging Face Datasets

Edit `configs/datasets.yaml` to choose dataset mirrors, splits, and subset sizes for VQAv2, GQA, TextVQA, and ScienceQA-IMG. Then create local cached manifests:

```bash
python3 src/prepare_datasets.py \
  --config configs/datasets.yaml \
  --train-limit 5 \
  --eval-limit 5
```

This writes:

```text
data/manifests/train.jsonl
data/manifests/eval.jsonl
data/images/
```

Verify the manifests before running MQT-LLaVA:

```bash
python3 src/verify_manifest.py --manifest data/manifests/train.jsonl
python3 src/verify_manifest.py --manifest data/manifests/eval.jsonl
```

You can prepare one dataset at a time:

```bash
python3 src/prepare_datasets.py --datasets textvqa --train-limit 5 --eval-limit 5
```

For proposal-aligned multi-dataset training, use:

```bash
python3 src/prepare_datasets.py --config configs/datasets_proposal_balanced.yaml --strict-datasets
python3 src/verify_manifest.py --manifest data/manifests/train_proposal_balanced.jsonl
python3 src/verify_manifest.py --manifest data/manifests/eval_proposal_balanced.jsonl
```

## Run Fixed Baselines

```bash
python3 src/evaluate_token_policy.py --data data/manifests/eval.jsonl --fixed-budget 8 --out results/fixed_8.jsonl
python3 src/evaluate_token_policy.py --data data/manifests/eval.jsonl --fixed-budget 36 --prompt-style short --out results/fixed_36.jsonl
python3 src/evaluate_token_policy.py --data data/manifests/eval.jsonl --fixed-budget 256 --prompt-style short --out results/fixed_256.jsonl
```

## Generate Oracle Labels

Run the frozen MQT-LLaVA model at each candidate budget and choose the smallest sufficient budget.

```bash
python3 src/oracle_labeling.py \
  --data data/manifests/train.jsonl \
  --out data/manifests/oracle_train_textvqa_small.jsonl \
  --budgets 36 64 144 256 \
  --score-key dataset_score \
  --prompt-style short \
  --limit 10
```

The oracle script appends one completed example at a time and resumes by default. If Colab disconnects, rerun the same command and it will skip existing `example_id`s already written to `--out`. Use `--no-resume` only when you intentionally want a fresh output file.

## Extract Frozen Router Features (Recommended)

```bash
python3 src/extract_router_features.py \
  --data data/manifests/oracle_train_textvqa_small.jsonl \
  --out data/manifests/oracle_train_textvqa_small_with_features.jsonl \
  --clip-model openai/clip-vit-large-patch14-336 \
  --batch-size 16 \
  --normalize
```

For router evaluation, also extract features for eval rows:

```bash
python3 src/extract_router_features.py \
  --data data/manifests/eval.jsonl \
  --out data/manifests/eval_with_features.jsonl \
  --clip-model openai/clip-vit-large-patch14-336 \
  --batch-size 16 \
  --normalize
```

`src/train_router.py` automatically infers embedding dimensions from the labeled JSONL when `prompt_embedding` / `image_embedding` are present, so you do not need to manually edit `prompt_dim` and `image_dim` in `configs/router.yaml` for CLIP variants.

## Train Routers

Train proposal baselines:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python3 src/train_router.py \
  --labels data/manifests/oracle_train_with_features.jsonl \
  --config configs/router_proposal_4budgets.yaml \
  --mode prompt \
  --out checkpoints/router_prompt.pt

KMP_DUPLICATE_LIB_OK=TRUE python3 src/train_router.py \
  --labels data/manifests/oracle_train_with_features.jsonl \
  --config configs/router_proposal_4budgets.yaml \
  --mode image \
  --out checkpoints/router_image.pt

KMP_DUPLICATE_LIB_OK=TRUE python3 src/train_router.py \
  --labels data/manifests/oracle_train_with_features.jsonl \
  --config configs/router_proposal_4budgets.yaml \
  --mode multimodal \
  --out checkpoints/router_multimodal.pt

KMP_DUPLICATE_LIB_OK=TRUE python3 src/train_router.py \
  --labels data/manifests/oracle_train_with_features.jsonl \
  --config configs/router_proposal_4budgets.yaml \
  --mode cross_attention \
  --out checkpoints/router_cross_attention.pt
```

`KMP_DUPLICATE_LIB_OK=TRUE` is included because this local macOS environment currently reports an OpenMP duplicate-runtime issue when importing PyTorch.

## Evaluate Learned Routing

```bash
KMP_DUPLICATE_LIB_OK=TRUE python3 src/evaluate_router.py \
  --data data/manifests/eval_with_features.jsonl \
  --checkpoint checkpoints/router_multimodal.pt \
  --out results/router_multimodal.jsonl

KMP_DUPLICATE_LIB_OK=TRUE python3 src/evaluate_router.py \
  --data data/manifests/eval_with_features.jsonl \
  --checkpoint checkpoints/router_cross_attention.pt \
  --out results/router_cross_attention.jsonl
```

## Run Rule-Based Task Routing

```bash
python3 src/evaluate_token_policy.py \
  --data data/manifests/eval.jsonl \
  --policy configs/task_token_policy.yaml \
  --out results/rule_task_aware.jsonl
```

## Model Adapter

Point this project at your cloned MQT-LLaVA repository:

```bash
export MQT_LLAVA_REPO=/absolute/path/to/MQT-LLaVA
export MQT_LLAVA_MODEL_PATH=gordonhu/MQT-LLaVA-7b
export MQT_LLAVA_BACKEND=persistent
export MQT_LLAVA_OFFLOAD_FOLDER=offload
```

Then run any evaluation script from this project. The adapter in `src/mqt_llava_adapter.py` imports:

```python
from llava.eval.run_llava import eval_model
```

and passes our selected budget through the repo's official argument:

```python
num_visual_tokens=<budget>
```

Example:

```bash
MQT_LLAVA_REPO=/absolute/path/to/MQT-LLaVA \
python3 src/evaluate_token_policy.py \
  --data data/manifests/eval.jsonl \
  --fixed-budget 36 \
  --out results/fixed_36.jsonl
```

If you installed MQT-LLaVA into the active Python environment and do not want to pass a repo path, use:

```bash
MQT_LLAVA_USE_INSTALLED=1 python3 src/evaluate_token_policy.py --data data/manifests/eval.jsonl --fixed-budget 36
```

Depending on the repo, that argument may be named something like:

- `m`
- `num_query_tokens`
- `token_budget`
- `visual_tokens`

For this repository's README, the relevant name is `num_visual_tokens`. Keep the outer evaluation code unchanged so all experiments are comparable.

The default adapter uses a persistent loaded model so the 7B checkpoint is not reloaded for every example or budget. If you need to debug against the original MQT-LLaVA helper, set `MQT_LLAVA_BACKEND=eval`, but that path is too slow and fragile for oracle labeling.

For local smoke tests before spending Colab GPU credits, you can use a lightweight backend:

```bash
export MQT_LLAVA_BACKEND=blip_vqa_smoke
export AUTO_MQT_SMOKE_MODEL=Salesforce/blip-vqa-base
export AUTO_MQT_SMOKE_DEVICE=auto
```

This backend is for pipeline validation only (it ignores `visual_tokens`), not final Auto-MQT accuracy/token trade-off reporting.

## Proposal Files

- `src/oracle_labeling.py`: Step 1, oracle budget labeling.
- `src/prepare_datasets.py`: converts Hugging Face subsets into local JSONL manifests.
- `src/verify_manifest.py`: validates manifest fields and saved images.
- `src/router_model.py`: late-fusion MLP router for prompt-only, image-only, and multimodal ablations.
- `src/train_router.py`: Step 2/3, router training with cost-aware loss.
- `src/evaluate_router.py`: Step 4, adaptive inference policy with optional confidence fallback.
- `src/evaluate_token_policy.py`: fixed-budget and rule-based baselines.
- `src/extract_router_features.py`: frozen CLIP feature extraction for prompt/image embeddings.
- `src/analyze_results.py`: report-friendly summary tables from result JSONL files.
- `configs/router_proposal_4budgets.yaml`: tuned training settings for the 4-budget proposal setup.
- `configs/datasets_proposal_balanced.yaml`: larger multi-dataset subset config for proposal-scale experiments.
