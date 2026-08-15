# Auto-MQT

Auto-MQT studies adaptive visual-token budgeting for efficient large vision-language model inference. The project builds on MQT-LLaVA, which supports multiple visual-token budgets at inference time, and adds a lightweight router that predicts how many visual tokens an image-question pair should use.

The main idea is built upon the idea that simpler trask should not always pay for a large visual-token budget, while OCR-heavy or visually detailed examples may need more tokens. Auto-MQT treats token selection as a supervised routing problem over a small budget set.

MQT-LLaVA uses a flexible visual-token budget of `36`, `64`, `144`, and `256` as opposed to the manually selected tokens in MQT-LLaVA. Auto-MQT trains small routers to select a budget automatically before running the frozen MQT-LLaVA backbone.

The project evaluates whether learned routing can improve the tradeoff between task accuracy and visual-token cost.

## Results

Experiments were run on a 600-example evaluation set balanced across VQAv2, GQA, TextVQA, and ScienceQA-IMG.

| Method | Accuracy | Avg Tokens | Avg Latency (s) |
| --- | ---: | ---: | ---: |
| Fixed 36 | 0.5387 | 36.00 | 0.271 |
| Fixed 64 | 0.5475 | 64.00 | 0.285 |
| Fixed 144 | 0.5532 | 144.00 | 0.305 |
| Fixed 256 | 0.5428 | 256.00 | 0.354 |
| Router Prompt | 0.5485 | 87.21 | 0.293 |
| Router Image | **0.5607** | 104.89 | 0.298 |
| Router Multimodal | 0.5452 | 76.99 | 0.289 |
| Router Cross Attention | 0.5447 | 63.08 | 0.285 |

The image router achieved the highest overall accuracy while using fewer visual tokens than the fixed 144-token and 256-token baselines.

## Figures

The main result figures are available in `results/`. We ran experiments for all 4 datasets as well as an experiment excluding TextVQA because of it low performance initially. The main plots for inference that illustrated the difference between the routers and datasets were: 

- `results/proposal_4ds_tradeoff.png`
- `results/proposal_4ds_fixed_budget_curves.png`

## Repository Layout

```text
configs/      Experiment and dataset configuration files
data/         Local manifests and cached image data
notebooks/    Colab and local notebook workflows
Proposal/     ACL-style report source and compiled PDF
results/      Evaluation outputs, tables, and plots
src/          Dataset preparation, MQT-LLaVA adapter, labeling, routing, and evaluation code
tests/        Lightweight validation tests
```

## Setup

Clone the MQT-LLaVA backbone separately:

```bash
git clone https://github.com/gordonhu608/MQT-LLaVA.git
```

Install this project’s Python dependencies:

```bash
pip install -r requirements.txt
```

Auto-MQT requires a CUDA-compatible NVIDIA GPU for efficient model inference. Otherwise, Google Colab works for non-compatible devices but may be less efficient. A colab test is laid out in:

```text
notebooks/test_colab.ipynb
```

## Data Preparation

Dataset examples are normalized into JSONL manifests with a common schema:

```json
{
  "dataset": "textvqa",
  "split": "train",
  "example_id": "textvqa_train_0001",
  "image": "data/images/textvqa/train/example.jpg",
  "prompt": "What word is written on the sign?",
  "answer": "stop",
  "answers": ["stop"],
  "task": "ocr"
}
```

To prepare manifests from Hugging Face datasets run:

```bash
python src/prepare_datasets.py --config configs/datasets_proposal_balanced.yaml
```

To validate a manifest run:

```bash
python src/verify_manifest.py --data data/manifests/eval_proposal.jsonl
```

## Oracle Budget Labeling

Oracle labels are created by running MQT-LLaVA at each candidate budget and selecting the smallest budget that achieves the best score for the example.

```bash
python src/oracle_labeling.py \
  --data data/manifests/train_proposal_balanced.jsonl \
  --out data/manifests/oracle_train_proposal.jsonl \
  --budgets 36 64 144 256 \
  --score-key dataset_score
```

## Router Training

Train a router with:

```bash
python src/train_router.py \
  --config configs/router_proposal_4budgets.yaml \
  --mode image
```

Supported router modes include:

- `prompt`
- `image`
- `multimodal`
- `cross_attention`

## Evaluation

Run a fixed-budget baseline:

```bash
python src/evaluate_token_policy.py \
  --data data/manifests/eval_proposal_with_features.jsonl \
  --fixed-budget 144 \
  --out results/fixed_144.jsonl
```

Evaluate a trained router:

```bash
python src/evaluate_router.py \
  --data data/manifests/eval_proposal_with_features.jsonl \
  --checkpoint results/router_image.pt \
  --out results/router_image.jsonl
```

Generate report tables and plots:

```bash
python src/report_dataset_breakdown.py
```

