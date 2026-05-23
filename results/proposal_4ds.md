# Auto-MQT Per-Dataset Benchmark Breakdown

## Paper-Style Dataset Score Table (%)

| run | vqav2 | gqa | textvqa | scienceqa_img | overall | avg_tokens |
| --- | --- | --- | --- | --- | --- | --- |
| fixed_36 | 67.33 | 70.00 | 25.47 | 52.67 | 53.87 | 36.00 |
| fixed_64 | 68.67 | 72.67 | 24.33 | 53.33 | 54.75 | 64.00 |
| fixed_144 | 69.33 | 72.00 | 27.93 | 52.00 | 55.32 | 144.00 |
| fixed_256 | 68.67 | 70.67 | 29.80 | 48.00 | 54.28 | 256.00 |
| router_prompt | 68.00 | 72.00 | 26.73 | 52.67 | 54.85 | 87.21 |
| router_image | 68.67 | 73.33 | 28.27 | 54.00 | 56.07 | 104.89 |
| router_multimodal | 68.00 | 71.33 | 26.07 | 52.67 | 54.52 | 76.99 |
| router_cross_attention | 68.67 | 71.33 | 24.53 | 53.33 | 54.47 | 63.08 |

## Overall Metrics

| run | examples | exact | relaxed | dataset_score | avg_tokens | avg_latency_s |
| --- | --- | --- | --- | --- | --- | --- |
| fixed_36 | 600 | 54.67 | 55.67 | 53.87 | 36.00 | 0.271 |
| fixed_64 | 600 | 55.67 | 57.00 | 54.75 | 64.00 | 0.285 |
| fixed_144 | 600 | 56.33 | 58.33 | 55.32 | 144.00 | 0.305 |
| fixed_256 | 600 | 55.33 | 57.17 | 54.28 | 256.00 | 0.354 |
| router_prompt | 600 | 55.83 | 57.17 | 54.85 | 87.21 | 0.293 |
| router_image | 600 | 57.17 | 58.67 | 56.07 | 104.89 | 0.298 |
| router_multimodal | 600 | 55.50 | 56.83 | 54.52 | 76.99 | 0.289 |
| router_cross_attention | 600 | 55.33 | 56.67 | 54.47 | 63.08 | 0.285 |

## Per-Dataset Detail

| run | dataset | examples | exact | relaxed | dataset_score | avg_tokens | avg_latency_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_36 | gqa | 150 | 70.00 | 70.00 | 70.00 | 36.00 | 0.237 |
| fixed_36 | scienceqa_img | 150 | 52.67 | 52.67 | 52.67 | 36.00 | 0.236 |
| fixed_36 | textvqa | 150 | 28.67 | 31.33 | 25.47 | 36.00 | 0.349 |
| fixed_36 | vqav2 | 150 | 67.33 | 68.67 | 67.33 | 36.00 | 0.260 |
| fixed_64 | gqa | 150 | 72.67 | 72.67 | 72.67 | 64.00 | 0.248 |
| fixed_64 | scienceqa_img | 150 | 53.33 | 53.33 | 53.33 | 64.00 | 0.248 |
| fixed_64 | textvqa | 150 | 28.00 | 32.00 | 24.33 | 64.00 | 0.370 |
| fixed_64 | vqav2 | 150 | 68.67 | 70.00 | 68.67 | 64.00 | 0.274 |
| fixed_144 | gqa | 150 | 72.00 | 72.67 | 72.00 | 144.00 | 0.269 |
| fixed_144 | scienceqa_img | 150 | 52.00 | 52.00 | 52.00 | 144.00 | 0.278 |
| fixed_144 | textvqa | 150 | 32.00 | 38.00 | 27.93 | 144.00 | 0.386 |
| fixed_144 | vqav2 | 150 | 69.33 | 70.67 | 69.33 | 144.00 | 0.288 |
| fixed_256 | gqa | 150 | 70.67 | 71.33 | 70.67 | 256.00 | 0.321 |
| fixed_256 | scienceqa_img | 150 | 48.00 | 48.00 | 48.00 | 256.00 | 0.311 |
| fixed_256 | textvqa | 150 | 34.00 | 39.33 | 29.80 | 256.00 | 0.444 |
| fixed_256 | vqav2 | 150 | 68.67 | 70.00 | 68.67 | 256.00 | 0.339 |
| router_prompt | gqa | 150 | 72.00 | 72.00 | 72.00 | 92.59 | 0.259 |
| router_prompt | scienceqa_img | 150 | 52.67 | 52.67 | 52.67 | 83.87 | 0.256 |
| router_prompt | textvqa | 150 | 30.67 | 34.67 | 26.73 | 109.23 | 0.385 |
| router_prompt | vqav2 | 150 | 68.00 | 69.33 | 68.00 | 63.15 | 0.273 |
| router_image | gqa | 150 | 73.33 | 73.33 | 73.33 | 120.64 | 0.261 |
| router_image | scienceqa_img | 150 | 54.00 | 54.00 | 54.00 | 84.27 | 0.256 |
| router_image | textvqa | 150 | 32.67 | 37.33 | 28.27 | 117.01 | 0.390 |
| router_image | vqav2 | 150 | 68.67 | 70.00 | 68.67 | 97.63 | 0.284 |
| router_multimodal | gqa | 150 | 71.33 | 71.33 | 71.33 | 95.92 | 0.258 |
| router_multimodal | scienceqa_img | 150 | 52.67 | 52.67 | 52.67 | 63.68 | 0.249 |
| router_multimodal | textvqa | 150 | 30.00 | 34.00 | 26.07 | 97.60 | 0.379 |
| router_multimodal | vqav2 | 150 | 68.00 | 69.33 | 68.00 | 50.75 | 0.270 |
| router_cross_attention | gqa | 150 | 71.33 | 71.33 | 71.33 | 60.91 | 0.246 |
| router_cross_attention | scienceqa_img | 150 | 53.33 | 53.33 | 53.33 | 63.25 | 0.247 |
| router_cross_attention | textvqa | 150 | 28.00 | 32.00 | 24.53 | 67.01 | 0.372 |
| router_cross_attention | vqav2 | 150 | 68.67 | 70.00 | 68.67 | 61.15 | 0.274 |