# Project Outline
This document provides an outline of Auto MQT. More details are outlined in Final Report.pdf

## Motivation

Large vision-language models convert images into visual tokens before passing them to a language model. A fixed visual-token budget is simple, but not always efficient. Some image-question pairs only require coarse visual information, while others need detailed evidence such as text, small objects, counting, or spatial relationships.

MQT-LLaVA provides a useful starting point because it supports multiple visual-token budgets at inference time. However, the budget is normally selected manually. Auto-MQT studies whether this budget can be selected automatically for each example.

## Research Question

Can a lightweight router predict the visual-token budget needed for an image-question pair while preserving answer accuracy and reducing average token usage? The project focuses on the tradeoffs between accuracy and efficiency. 

- smaller budgets reduce visual-token cost and latency
- larger budgets may improve detailed visual reasoning
- adaptive routing may outperform a single fixed budget

## Method Overview

Auto-MQT keeps the pretrained MQT-LLaVA backbone frozen and trains only a small budget router. The router predicts one of four candidate visual-token budgets:

```text
36, 64, 144, 256
```

The selected budget is then passed to MQT-LLaVA for answer generation.

## Oracle Budget Labeling

Router supervision comes from oracle budget labels. For each training example, MQT-LLaVA is evaluated at every candidate budget. The oracle label is the smallest budget that achieves the best available score for that example.

This produces a target budget without modifying the VLM backbone.

Oracle labeling is expensive because each example requires multiple full inference passes. For this reason, the project uses a limited candidate budget set and capped dataset subsets.

## Router Variants

The project evaluates four router designs:

- `prompt`: predicts the budget from question features
- `image`: predicts the budget from image features
- `multimodal`: combines prompt and image features
- `cross_attention`: uses a small cross-attention module to combine prompt and image information

All routers are lightweight compared with the frozen MQT-LLaVA-7B model.

## Datasets

Experiments use four vision-language benchmarks:

- VQAv2 for general visual question answering
- GQA for visual and spatial reasoning
- TextVQA for OCR-focused visual question answering
- ScienceQA-IMG for multimodal science reasoning

The final evaluation uses 600 examples total, balanced across the four datasets with 150 examples each.

## Evaluation

Auto-MQT is compared against fixed-budget MQT-LLaVA baselines:

```text
Fixed 36, Fixed 64, Fixed 144, Fixed 256
```

The main metrics are:

- accuracy
- average visual tokens
- average latency
- budget regret
- under-budget rate

Budget regret measures extra tokens used relative to the oracle budget. Under-budget rate measures how often the router selects fewer tokens than the oracle budget.

## Results Summary

The image router achieved the best overall result:

- accuracy: `0.5607`
- average visual tokens: `104.89`
- average latency: `0.298s`

This outperformed all fixed-budget baselines in accuracy while using fewer average tokens than the fixed 144-token and 256-token settings.

The cross-attention router performed similarly to the fixed 64-token baseline, suggesting that more complex fusion does not automatically improve budget selection in this setting.

## Key Findings

Learned routing can improve the accuracy-cost tradeoff for MQT-LLaVA.

Image features were especially useful for budget prediction, which suggests that visual complexity is a strong signal for token allocation.

Different datasets respond differently to token budgets. TextVQA benefits more from larger budgets, while VQAv2 and GQA are more stable across budget levels.

Router calibration matters. Selecting too few tokens can reduce answer quality, even when the average token budget is low.

## Limitations

Oracle labeling is computationally expensive because every labeled example requires inference at multiple budgets.

The evaluation uses a capped 600-example subset rather than full benchmark splits.

TextVQA remains challenging because OCR-heavy examples often require detailed visual information and exact text recognition.

The current router predicts among four discrete budgets. Although MQT-LLaVA supports arbitrary integer budgets from 1 to 256, this project uses a smaller budget set for controlled comparison.

## Future Work

Future work could explore denser or continuous budget spaces, larger training and evaluation sets, and additional benchmarks such as POPE and MME.

Another direction is reinforcement learning for budget selection. Instead of training the router only to imitate oracle labels, the router could be treated as a policy that receives a reward for correct answers and a penalty for using more visual tokens:

```text
reward = task_score - token_cost_penalty
```

This would optimize the router directly for the accuracy-efficiency objective. It could also reduce reliance on exhaustive oracle labeling, although it would introduce challenges such as sparse rewards, unstable training, and expensive VLM queries during policy learning.

Auto-MQT could also be extended beyond MQT-LLaVA to test whether lightweight token-budget routing generalizes to other vision-language model architectures.
