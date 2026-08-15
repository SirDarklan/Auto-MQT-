# Token Routing Pipeline

This document describes the Auto-MQT routing workflow. The pipeline converts benchmark examples into a shared JSONL format, labels each example with an oracle visual-token budget, trains lightweight routers, and evaluates the accuracy-cost tradeoff against fixed-budget MQT-LLaVA baselines.

The final experiments used four candidate budgets:

```text
36, 64, 144, 256
```
## Format 

All datasets are converted into a common JSONL schema:

```json
{
  "dataset": "vqav2",
  "split": "validation",
  "example_id": "vqav2_val_000001",
  "image": "data/images/vqav2/validation/example.jpg",
  "prompt": "What color is the bus?",
  "answer": "yellow",
  "answers": ["yellow"],
  "task": "vqa"
}
```

## Dataset Preparation

Dataset definitions are controlled by YAML files in `configs/`. The balanced proposal setting uses:

```text
configs/datasets_proposal_balanced.yaml
```

Prepare local manifests and image caches:

```bash
python src/prepare_datasets.py --config configs/datasets_proposal_balanced.yaml
```

Check that each row has a readable image, prompt, and answer:

```bash
python src/verify_manifest.py --data data/manifests/eval_proposal.jsonl
```

## MQT-LLaVA Backend

The adapter in `src/mqt_llava_adapter.py` connects this project to a local MQT-LLaVA clone. The expected setup is:

```bash
git clone https://github.com/gordonhu608/MQT-LLaVA.git
```

Set the backbone path before running inference:

```bash
export MQT_LLAVA_REPO=/path/to/MQT-LLaVA
export MQT_LLAVA_MODEL_PATH=gordonhu/MQT-LLaVA-7b
```

For Colab or GPU runs, the persistent backend is recommended because it loads the model once and reuses it across examples:

```bash
export MQT_LLAVA_BACKEND=persistent
export MQT_LLAVA_OFFLOAD_FOLDER=/content/mqt_offload
```

## Fixed-Budget Evaluation

Fixed-budget runs provide the main baselines:

```bash
python src/evaluate_token_policy.py \
  --data data/manifests/eval_proposal_with_features.jsonl \
  --fixed-budget 64 \
  --out results/fixed_64.jsonl
```

Each output row includes the original fields plus the model prediction, selected token budget, latency, and score fields.

## Oracle Budget Labeling

Oracle labeling runs each example at every candidate budget and selects the smallest budget that achieves the best available score.

```bash
python src/oracle_labeling.py \
  --data data/manifests/train_proposal_balanced.jsonl \
  --out data/manifests/oracle_train_proposal.jsonl \
  --budgets 36 64 144 256 \
  --score-key dataset_score
```

The script is resumable. If an output file already contains labeled examples, existing `example_id` values are skipped.

## Router Training

Router settings are controlled by:

```text
configs/router_proposal_4budgets.yaml
```

To train a router:

```bash
python src/train_router.py \
  --config configs/router_proposal_4budgets.yaml \
  --mode image
```

Available modes:

- `prompt`: uses question features
- `image`: uses image features
- `multimodal`: combines prompt and image features
- `cross_attention`: uses a small cross-attention router

## Router Evaluation

Evaluate a trained router:

```bash
python src/evaluate_router.py \
  --data data/manifests/eval_proposal_with_features.jsonl \
  --checkpoint results/router_image.pt \
  --out results/router_image.jsonl
```

Router outputs can be compared against fixed-budget baselines using:

```bash
python src/report_dataset_breakdown.py
```

## Metrics

The project reports:

- `dataset_score`: dataset-aware accuracy score
- `avg_visual_tokens`: average selected visual-token budget
- `avg_latency_s`: average inference latency
- `budget_regret`: extra tokens used relative to the oracle budget
- `under_budget_rate`: fraction of examples where the router selected fewer tokens than the oracle budget

## Final Experimental Setting

The final reported experiment uses:

- training examples: 2,000 total, 500 per dataset
- evaluation examples: 600 total, 150 per dataset
- datasets: VQAv2, GQA, TextVQA, ScienceQA-IMG
- candidate budgets: `36`, `64`, `144`, `256`
- backbone: frozen MQT-LLaVA-7B

