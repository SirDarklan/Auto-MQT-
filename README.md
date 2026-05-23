# Auto-MQT: Prompt-Adaptive Visual Token Budgeting

This repository implements a practical Auto-MQT pipeline on top of MQT-LLaVA:

1. Build dataset manifests (image + prompt + answer JSONL).
2. Run frozen MQT-LLaVA at multiple token budgets to create oracle labels.
3. Extract frozen prompt/image features for router training.
4. Train prompt-only, image-only, and multimodal routers.
5. Evaluate accuracy vs token usage against fixed-budget baselines.

## 0) Environment Split (Recommended)

- Local Windows (your RTX 3050 Ti laptop): coding, quick smoke tests, report analysis, plotting.
- Google Colab GPU: oracle labeling, feature extraction, and any larger-scale baseline/router evaluations.

This split is important because oracle labeling is the expensive stage.

## 1) Local Setup (Windows + conda)

```powershell
conda create -n auto-mqt python=3.12 -y
conda activate auto-mqt
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Clone MQT-LLaVA somewhere on your machine:

```powershell
git clone https://github.com/gordonhu608/MQT-LLaVA.git C:\GitHub_Repos\MQT-LLaVA
pip install -e C:\GitHub_Repos\MQT-LLaVA --no-deps
```

Set environment variables in the same shell before running scripts:

```powershell
$env:MQT_LLAVA_REPO="C:\GitHub_Repos\MQT-LLaVA"
$env:MQT_LLAVA_MODEL_PATH="gordonhu/MQT-LLaVA-7b"
$env:MQT_LLAVA_BACKEND="persistent"
$env:MQT_LLAVA_OFFLOAD_FOLDER="offload"
```

## 2) Colab from VS Code: File Access Reality

When your notebook is connected to a Colab kernel, code runs on the remote Colab VM (`/content`), not directly on your local disk.

If your synced folder is under Google Drive web `Computers > My Laptop > Auto-MQT-`, Colab usually will not see that path directly through `drive.mount`. Put a shortcut or a copy of `Auto-MQT-` under `My Drive` once, and use that path for Colab.

Use one of these:

- Right-click files/folders in VS Code -> `Upload to Colab` (uploads to the active server).
- `Colab: Mount Google Drive to Server...` for Drive-backed persistence.
- Git clone the repo inside Colab.

The Colab notebook [auto_mqt_colab_pipeline.ipynb](C:\GitHub_Repos\Auto-MQT-\notebooks\auto_mqt_colab_pipeline.ipynb) now auto-mounts Drive and auto-detects the repo from common `MyDrive` locations (or `AUTO_MQT_REPO` env var).
If oracle labeling hits CUDA OOM on L4, restart the runtime and set `MQT_LLAVA_LOAD_8BIT=1` in the notebook env cell before running oracle labeling.

## 2.1) Fast Local Smoke Test (Before Colab)

Use [auto_mqt_local_smoke_test.ipynb](C:\GitHub_Repos\Auto-MQT-\notebooks\auto_mqt_local_smoke_test.ipynb) to validate the full pipeline on your laptop GPU with a small backend (`Salesforce/blip-vqa-base`).

- It exercises manifests, baseline evaluation, oracle labeling, feature extraction, router training, and result summarization.
- It is for correctness checks only; final project numbers should come from MQT-LLaVA runs on Colab.

## 3) First Local Smoke Test (Tiny)

Use tiny limits first to verify everything wires correctly:

```powershell
python src/prepare_datasets.py --config configs/datasets.yaml --datasets textvqa --train-limit 5 --eval-limit 5
python src/verify_manifest.py --manifest data/manifests/train.jsonl
python src/verify_manifest.py --manifest data/manifests/eval.jsonl
```

If this passes, move heavy steps to Colab.

## 4) Heavy Compute on Colab (Main Pipeline)

For best router quality, do not stay at TextVQA-only tiny subsets. After smoke tests, move to the proposal-style multi-dataset config (`configs/datasets_proposal_balanced.yaml`).
Use `requirements_colab.txt` on Colab (not `requirements.txt`) to avoid overriding Colab-managed Jupyter packages.
The proposal config uses an image-backed VQAv2 mirror (`landersanmi/VQAv2`) for pipeline compatibility. Some other mirrors expose question/answer metadata without image payloads and will fail manifest building.

### 4.1 Oracle Labels (expensive)

```bash
python src/oracle_labeling.py \
  --data data/manifests/train.jsonl \
  --out data/manifests/oracle_train.jsonl \
  --budgets 36 64 144 256 \
  --score-key dataset_score \
  --zero-score-budget 64 \
  --prompt-style short
python src/oracle_diagnostics.py --data data/manifests/oracle_train.jsonl
```

This script is resumable by default. If Colab disconnects, rerun the exact same command.
For label stability, set `--zero-score-budget 64`: if all candidate budgets score 0 on an example, the oracle will assign 64 instead of over-penalizing with max-budget labels.

Recommended larger dataset preparation (proposal-aligned):

```bash
python src/prepare_datasets.py --config configs/datasets_proposal_balanced.yaml --prompt-style none --strict-datasets
python src/verify_manifest.py --manifest data/manifests/train_proposal_balanced.jsonl
python src/verify_manifest.py --manifest data/manifests/eval_proposal_balanced.jsonl
```

Then use those manifests for oracle/eval steps.

### 4.2 Frozen Feature Extraction (new, GPU-friendly)

This writes real CLIP features into each JSONL row (`prompt_embedding`, `image_embedding`):

```bash
python src/extract_router_features.py \
  --data data/manifests/oracle_train.jsonl \
  --out data/manifests/oracle_train_with_features.jsonl \
  --clip-model openai/clip-vit-large-patch14-336 \
  --batch-size 16 \
  --normalize
```

Also extract features for evaluation data:

```bash
python src/extract_router_features.py \
  --data data/manifests/eval.jsonl \
  --out data/manifests/eval_with_features.jsonl \
  --clip-model openai/clip-vit-large-patch14-336 \
  --batch-size 16 \
  --normalize
```

### 4.3 Router Training

```bash
python src/train_router.py --labels data/manifests/oracle_train_with_features.jsonl --config configs/router_proposal_4budgets.yaml --mode prompt --out checkpoints/router_prompt.pt
python src/train_router.py --labels data/manifests/oracle_train_with_features.jsonl --config configs/router_proposal_4budgets.yaml --mode image --out checkpoints/router_image.pt
python src/train_router.py --labels data/manifests/oracle_train_with_features.jsonl --config configs/router_proposal_4budgets.yaml --mode multimodal --out checkpoints/router_multimodal.pt
python src/train_router.py --labels data/manifests/oracle_train_with_features.jsonl --config configs/router_proposal_4budgets.yaml --mode cross_attention --out checkpoints/router_cross_attention.pt
```

### 4.4 Baseline + Router Evaluation

```bash
python src/evaluate_token_policy.py --data data/manifests/eval_with_features.jsonl --fixed-budget 36 --prompt-style short --out results/fixed_36.jsonl
python src/evaluate_token_policy.py --data data/manifests/eval_with_features.jsonl --fixed-budget 64 --prompt-style short --out results/fixed_64.jsonl
python src/evaluate_token_policy.py --data data/manifests/eval_with_features.jsonl --fixed-budget 144 --prompt-style short --out results/fixed_144.jsonl
python src/evaluate_token_policy.py --data data/manifests/eval_with_features.jsonl --fixed-budget 256 --prompt-style short --out results/fixed_256.jsonl

python src/evaluate_router.py --data data/manifests/eval_with_features.jsonl --checkpoint checkpoints/router_prompt.pt --out results/router_prompt.jsonl
python src/evaluate_router.py --data data/manifests/eval_with_features.jsonl --checkpoint checkpoints/router_image.pt --out results/router_image.jsonl
python src/evaluate_router.py --data data/manifests/eval_with_features.jsonl --checkpoint checkpoints/router_multimodal.pt --out results/router_multimodal.jsonl
python src/evaluate_router.py --data data/manifests/eval_with_features.jsonl --checkpoint checkpoints/router_cross_attention.pt --out results/router_cross_attention.jsonl
```

### 4.5 Optional: Oracle Labels for Eval (for regret/under-budget metrics)

If you want strict regret diagnostics on the same evaluation set:

```bash
python src/oracle_labeling.py \
  --data data/manifests/eval.jsonl \
  --out data/manifests/oracle_eval.jsonl \
  --budgets 36 64 144 256 \
  --score-key dataset_score \
  --prompt-style short
```

## 5) Local Analysis + Report Tables

Pull result JSONL files back to local and summarize:

```powershell
python src/analyze_results.py `
  --inputs `
    fixed_36=results/fixed_36.jsonl `
    fixed_64=results/fixed_64.jsonl `
    fixed_144=results/fixed_144.jsonl `
    fixed_256=results/fixed_256.jsonl `
    router_prompt=results/router_prompt.jsonl `
    router_image=results/router_image.jsonl `
    router_multimodal=results/router_multimodal.jsonl `
    router_cross_attention=results/router_cross_attention.jsonl `
  --oracle data/manifests/oracle_eval.jsonl `
  --train-manifest data/manifests/train.jsonl `
  --eval-manifest data/manifests/eval_with_features.jsonl `
  --checkpoints `
    router_prompt=checkpoints/router_prompt.pt `
    router_image=checkpoints/router_image.pt `
    router_multimodal=checkpoints/router_multimodal.pt `
    router_cross_attention=checkpoints/router_cross_attention.pt `
  --backend-label mqt-persistent `
  --out-csv results/summary.csv `
  --out-markdown results/final_summary.md
```

This gives:

- exact/relaxed accuracy
- dataset_score (dataset-specific proxy metric)
- average visual tokens
- average latency
- budget regret
- under-budget rate
- dataset/task composition summary
- method/settings summary (including checkpoint hyperparameters)
- proposal-coverage checklist (implemented vs pending items)

### 5.1 Paper-Style Per-Dataset Table + Graphs (No Re-Inference)

Use existing `results/*.jsonl` to generate:

- per-dataset comparison table (like the original paper style)
- overall score table
- per-dataset detail table
- graphs for run comparison and token/accuracy curves

```powershell
python src/report_dataset_breakdown.py `
  --inputs `
    fixed_36=results/fixed_36.jsonl `
    fixed_64=results/fixed_64.jsonl `
    fixed_144=results/fixed_144.jsonl `
    fixed_256=results/fixed_256.jsonl `
    router_prompt=results/router_prompt.jsonl `
    router_image=results/router_image.jsonl `
    router_multimodal=results/router_multimodal.jsonl `
    router_cross_attention=results/router_cross_attention.jsonl `
  --out-dir results `
  --out-prefix proposal_4ds
```

Generated files:

- `results/proposal_4ds.md`
- `results/proposal_4ds_paper_style.csv`
- `results/proposal_4ds_overall.csv`
- `results/proposal_4ds_by_dataset.csv`
- `results/proposal_4ds_overall_score.png`
- `results/proposal_4ds_per_dataset_by_run.png`
- `results/proposal_4ds_fixed_budget_curves.png`
- `results/proposal_4ds_tradeoff.png`

### 5.2 Toggle TextVQA On/Off (Parameter Only)

You can disable TextVQA with `--exclude-datasets textvqa` (or enable only selected sets with `--include-datasets ...`).

No re-inference needed (just re-score existing JSONL results):

```powershell
python src/report_dataset_breakdown.py `
  --inputs `
    fixed_36=results/fixed_36.jsonl `
    fixed_64=results/fixed_64.jsonl `
    fixed_144=results/fixed_144.jsonl `
    fixed_256=results/fixed_256.jsonl `
    router_prompt=results/router_prompt.jsonl `
    router_image=results/router_image.jsonl `
    router_multimodal=results/router_multimodal.jsonl `
    router_cross_attention=results/router_cross_attention.jsonl `
  --exclude-datasets textvqa `
  --out-dir results `
  --out-prefix proposal_no_textvqa
```

Rerun path (only if you want models trained/evaluated without TextVQA):

- `src/oracle_labeling.py`: supports `--exclude-datasets`
- `src/extract_router_features.py`: supports `--exclude-datasets`
- `src/train_router.py`: supports `--exclude-datasets`
- `src/evaluate_token_policy.py`: supports `--exclude-datasets`
- `src/evaluate_router.py`: supports `--exclude-datasets`
- `src/analyze_results.py`: supports `--exclude-datasets`

Example (same commands as Sections 4.x, plus one flag):

```bash
# Oracle labels
python src/oracle_labeling.py --data data/manifests/train_proposal_balanced.jsonl --out data/manifests/oracle_train_proposal.jsonl --budgets 36 64 144 256 --score-key dataset_score --prompt-style short --exclude-datasets textvqa

# Feature extraction
python src/extract_router_features.py --data data/manifests/oracle_train_proposal.jsonl --out data/manifests/oracle_train_proposal_with_features.jsonl --normalize --exclude-datasets textvqa
python src/extract_router_features.py --data data/manifests/eval_proposal_balanced.jsonl --out data/manifests/eval_proposal_with_features.jsonl --normalize --exclude-datasets textvqa

# Router training
python src/train_router.py --labels data/manifests/oracle_train_proposal_with_features.jsonl --config configs/router_proposal_4budgets.yaml --mode multimodal --out checkpoints/router_multimodal_no_textvqa.pt --exclude-datasets textvqa

# Evaluation
python src/evaluate_token_policy.py --data data/manifests/eval_proposal_with_features.jsonl --fixed-budget 64 --out results/fixed_64_no_textvqa.jsonl --exclude-datasets textvqa
python src/evaluate_router.py --data data/manifests/eval_proposal_with_features.jsonl --checkpoint checkpoints/router_multimodal_no_textvqa.pt --out results/router_multimodal_no_textvqa.jsonl --exclude-datasets textvqa
```

## 6) Suggested Milestone Order

1. End-to-end tiny run on one dataset (TextVQA) with 10-20 examples.
2. Scale oracle labeling on Colab.
3. Extract real features and train 3 router variants.
4. Run matched-budget comparisons.
5. Write category-level and efficiency analysis in final report.

## 7) Key Project Files

- `src/oracle_labeling.py`: oracle budget generation (resumable)
- `src/extract_router_features.py`: frozen CLIP feature extraction
- `src/train_router.py`: cost-aware router training
- `src/evaluate_router.py`: adaptive routing evaluation
- `src/evaluate_token_policy.py`: fixed-budget and rule-based baselines
- `src/analyze_results.py`: summary table generation
