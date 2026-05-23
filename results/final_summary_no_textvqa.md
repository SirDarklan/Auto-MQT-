# Auto-MQT Final Summary

- Generated: 2026-05-23 21:57 UTC

## Results

| run | examples | exact | relaxed | dataset_score | avg_tokens | avg_latency_s | regret | under_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_36 | 450 | 0.6333 | 0.6378 | 0.6333 | 36.00 | 0.245 | nan | nan |
| fixed_64 | 450 | 0.6489 | 0.6533 | 0.6489 | 64.00 | 0.257 | nan | nan |
| fixed_144 | 450 | 0.6444 | 0.6511 | 0.6444 | 144.00 | 0.278 | nan | nan |
| fixed_256 | 450 | 0.6244 | 0.6311 | 0.6244 | 256.00 | 0.324 | nan | nan |
| router_prompt | 450 | 0.6422 | 0.6467 | 0.6422 | 79.87 | 0.263 | nan | nan |
| router_image | 450 | 0.6533 | 0.6578 | 0.6533 | 100.84 | 0.267 | nan | nan |
| router_multimodal | 450 | 0.6400 | 0.6444 | 0.6400 | 70.12 | 0.259 | nan | nan |
| router_cross_attention | 450 | 0.6444 | 0.6489 | 0.6444 | 61.77 | 0.256 | nan | nan |

## Dataset Summary


## Oracle Summary

- Oracle summary not provided.

## Method/Settings Summary

- Metrics in this report: exact match, relaxed match, dataset_score, avg visual tokens, avg latency, regret, under-budget rate.
- Regret is computed as `max(0, chosen_budget - oracle_budget)` averaged over overlapping example IDs.
- Under-budget rate is the fraction of examples where `chosen_budget < oracle_budget`.

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
