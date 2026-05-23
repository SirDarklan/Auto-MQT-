# Auto-MQT Final Summary

- Generated: 2026-05-23 19:36 UTC
- Backend: mqt-persistent

## Results

| run | examples | exact | relaxed | dataset_score | avg_tokens | avg_latency_s | regret | under_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_36 | 600 | 0.5467 | 0.5567 | 0.5387 | 36.00 | 0.271 | 0.00 | 0.4533 |
| fixed_64 | 600 | 0.5567 | 0.5700 | 0.5475 | 64.00 | 0.285 | 15.31 | 0.0283 |
| fixed_144 | 600 | 0.5633 | 0.5833 | 0.5532 | 144.00 | 0.305 | 93.04 | 0.0133 |
| fixed_256 | 600 | 0.5533 | 0.5717 | 0.5428 | 256.00 | 0.354 | 203.55 | 0.0000 |
| router_prompt | 600 | 0.5583 | 0.5717 | 0.5485 | 87.21 | 0.293 | 38.95 | 0.0633 |
| router_image | 600 | 0.5717 | 0.5867 | 0.5607 | 104.89 | 0.298 | 54.67 | 0.0167 |
| router_multimodal | 600 | 0.5550 | 0.5683 | 0.5452 | 76.99 | 0.289 | 34.25 | 0.2517 |
| router_cross_attention | 600 | 0.5533 | 0.5667 | 0.5447 | 63.08 | 0.285 | 15.00 | 0.0583 |

## Dataset Summary

- Train manifest: `data/manifests/train_proposal_balanced.jsonl`
- Train rows: 2000
- Train datasets: gqa:500, scienceqa_img:500, textvqa:500, vqav2:500
- Train tasks: ocr:500, reasoning:1000, vqa:500
- Train avg prompt words: 18.86
- Eval manifest: `data/manifests/eval_proposal_with_features.jsonl`
- Eval rows: 600
- Eval datasets: gqa:150, scienceqa_img:150, textvqa:150, vqav2:150
- Eval tasks: ocr:150, reasoning:300, vqa:150
- Eval avg prompt words: 17.82

## Oracle Summary

- Oracle file: `data/manifests/oracle_eval_proposal.jsonl`
- Oracle rows: 600
- Candidate budgets: [36, 64, 144, 256]
- Oracle budget distribution: 144:9, 256:8, 36:328, 64:255
- Avg oracle budget: 52.45
- Oracle score keys: dataset_score:600

## Method/Settings Summary

- Metrics in this report: exact match, relaxed match, dataset_score, avg visual tokens, avg latency, regret, under-budget rate.
- Regret is computed as `max(0, chosen_budget - oracle_budget)` averaged over overlapping example IDs.
- Under-budget rate is the fraction of examples where `chosen_budget < oracle_budget`.

### Router Checkpoints

- router_prompt: mode=prompt, best_val_metric=0.5038 (objective), budgets=[36, 64, 144, 256], hidden_dim=384, lr=0.0005, batch_size=32, lambda_cost=0.05, cost_power=1.0, bias=0.4, threshold=0.0
- router_image: mode=image, best_val_metric=0.5071 (objective), budgets=[36, 64, 144, 256], hidden_dim=384, lr=0.0005, batch_size=32, lambda_cost=0.05, cost_power=1.0, bias=0.4, threshold=0.0
- router_multimodal: mode=multimodal, best_val_metric=0.5041 (objective), budgets=[36, 64, 144, 256], hidden_dim=384, lr=0.0005, batch_size=32, lambda_cost=0.05, cost_power=1.0, bias=0.4, threshold=0.0
- router_cross_attention: mode=cross_attention, best_val_metric=0.5052 (objective), budgets=[36, 64, 144, 256], hidden_dim=384, lr=0.0005, batch_size=32, lambda_cost=0.05, cost_power=1.0, bias=0.0, threshold=0.0

## Proposal Coverage

| Item | Status | Evidence |
| --- | --- | --- |
| Oracle budget labeling | implemented | src/oracle_labeling.py |
| Tolerance-based labels (delta) | implemented | --tolerance in oracle_labeling.py |
| Prompt-only router | implemented | --mode prompt in train_router.py |
| Image-only router | implemented | --mode image in train_router.py |
| Multimodal router | implemented | --mode multimodal in train_router.py |
| Tiny cross-attention router variant | implemented | --mode cross_attention in train_router.py |
| Cost-aware objective | implemented | cost_aware_loss in train_router.py |
| Confidence fallback policy | implemented | predict_budget threshold in router_model.py |
| Calibration loss term | implemented | gamma_calibration in cost_aware_loss |
| Soft budget target distribution | implemented | soft_target_distribution from budget_results |
| Official dataset-specific metrics | partial | dataset_score implemented (VQA-soft + exact), not external leaderboard scripts |
