# Auto-MQT Per-Dataset Benchmark Breakdown

## Paper-Style Dataset Score Table (%)

| run | vqav2 | gqa | scienceqa_img | overall | avg_tokens |
| --- | --- | --- | --- | --- | --- |
| fixed_36 | 67.33 | 70.00 | 52.67 | 63.33 | 36.00 |
| fixed_64 | 68.67 | 72.67 | 53.33 | 64.89 | 64.00 |
| fixed_144 | 69.33 | 72.00 | 52.00 | 64.44 | 144.00 |
| fixed_256 | 68.67 | 70.67 | 48.00 | 62.44 | 256.00 |
| router_prompt | 68.00 | 72.00 | 52.67 | 64.22 | 79.87 |
| router_image | 68.67 | 73.33 | 54.00 | 65.33 | 100.84 |
| router_multimodal | 68.00 | 71.33 | 52.67 | 64.00 | 70.12 |
| router_cross_attention | 68.67 | 71.33 | 53.33 | 64.44 | 61.77 |

## Overall Metrics

| run | examples | exact | relaxed | dataset_score | avg_tokens | avg_latency_s |
| --- | --- | --- | --- | --- | --- | --- |
| fixed_36 | 450 | 63.33 | 63.78 | 63.33 | 36.00 | 0.245 |
| fixed_64 | 450 | 64.89 | 65.33 | 64.89 | 64.00 | 0.257 |
| fixed_144 | 450 | 64.44 | 65.11 | 64.44 | 144.00 | 0.278 |
| fixed_256 | 450 | 62.44 | 63.11 | 62.44 | 256.00 | 0.324 |
| router_prompt | 450 | 64.22 | 64.67 | 64.22 | 79.87 | 0.263 |
| router_image | 450 | 65.33 | 65.78 | 65.33 | 100.84 | 0.267 |
| router_multimodal | 450 | 64.00 | 64.44 | 64.00 | 70.12 | 0.259 |
| router_cross_attention | 450 | 64.44 | 64.89 | 64.44 | 61.77 | 0.256 |

## Per-Dataset Detail

| run | dataset | examples | exact | relaxed | dataset_score | avg_tokens | avg_latency_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_36 | gqa | 150 | 70.00 | 70.00 | 70.00 | 36.00 | 0.237 |
| fixed_36 | scienceqa_img | 150 | 52.67 | 52.67 | 52.67 | 36.00 | 0.236 |
| fixed_36 | vqav2 | 150 | 67.33 | 68.67 | 67.33 | 36.00 | 0.260 |
| fixed_64 | gqa | 150 | 72.67 | 72.67 | 72.67 | 64.00 | 0.248 |
| fixed_64 | scienceqa_img | 150 | 53.33 | 53.33 | 53.33 | 64.00 | 0.248 |
| fixed_64 | vqav2 | 150 | 68.67 | 70.00 | 68.67 | 64.00 | 0.274 |
| fixed_144 | gqa | 150 | 72.00 | 72.67 | 72.00 | 144.00 | 0.269 |
| fixed_144 | scienceqa_img | 150 | 52.00 | 52.00 | 52.00 | 144.00 | 0.278 |
| fixed_144 | vqav2 | 150 | 69.33 | 70.67 | 69.33 | 144.00 | 0.288 |
| fixed_256 | gqa | 150 | 70.67 | 71.33 | 70.67 | 256.00 | 0.321 |
| fixed_256 | scienceqa_img | 150 | 48.00 | 48.00 | 48.00 | 256.00 | 0.311 |
| fixed_256 | vqav2 | 150 | 68.67 | 70.00 | 68.67 | 256.00 | 0.339 |
| router_prompt | gqa | 150 | 72.00 | 72.00 | 72.00 | 92.59 | 0.259 |
| router_prompt | scienceqa_img | 150 | 52.67 | 52.67 | 52.67 | 83.87 | 0.256 |
| router_prompt | vqav2 | 150 | 68.00 | 69.33 | 68.00 | 63.15 | 0.273 |
| router_image | gqa | 150 | 73.33 | 73.33 | 73.33 | 120.64 | 0.261 |
| router_image | scienceqa_img | 150 | 54.00 | 54.00 | 54.00 | 84.27 | 0.256 |
| router_image | vqav2 | 150 | 68.67 | 70.00 | 68.67 | 97.63 | 0.284 |
| router_multimodal | gqa | 150 | 71.33 | 71.33 | 71.33 | 95.92 | 0.258 |
| router_multimodal | scienceqa_img | 150 | 52.67 | 52.67 | 52.67 | 63.68 | 0.249 |
| router_multimodal | vqav2 | 150 | 68.00 | 69.33 | 68.00 | 50.75 | 0.270 |
| router_cross_attention | gqa | 150 | 71.33 | 71.33 | 71.33 | 60.91 | 0.246 |
| router_cross_attention | scienceqa_img | 150 | 53.33 | 53.33 | 53.33 | 63.25 | 0.247 |
| router_cross_attention | vqav2 | 150 | 68.67 | 70.00 | 68.67 | 61.15 | 0.274 |