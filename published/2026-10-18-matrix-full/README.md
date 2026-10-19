# TADBench full matrix — 2026-10-18

Full-mode run of `matrix_config.json` (11 detector variants × embedders: small, medium, large)
on `data/processed/benchmark_description_combined.jsonl` (3,451 samples: 820 benign,
2,631 malicious).

## Setup

| | |
|---|---|
| Code | `20b8c84e6117419da96ac504f19914d964250e01` |
| Host | AWS `c7i.4xlarge` (Intel Xeon Platinum 8488C, 8 cores / 16 threads, 32 GB), eu-north-1, Ubuntu 24.04 |
| Python | 3.11, PyTorch 2.14.1 (default CUDA wheel; the host has no GPU and the runner uses `device="cpu"`); exact packages in `requirements-frozen.txt` |
| Data | Fresh `tadbench download` + `tadbench prepare` on the host |
| Mode | `--mode full`: latency profiled on all 3,451 samples × 3 runs, throughput on all samples |
| Wall time | small 1.2 h, medium 2.5 h, large 6.9 h |

Each embedder ran as a separate `tadbench matrix` invocation with the same config,
narrowed to one embedder. `matrix_results.csv` concatenates them (594 rows).

Latency is per-sample `score_single()` time with the embedding cache bypassed, so it
includes model inference. Throughput is batch scoring of the whole dataset.

## Results

Aggregate metrics (`attack_type == "all"`). TPR is interpolated from the ROC curve at
the given false-positive rate. ECE is expected calibration error over 10 bins.

| Detector | Embedder | AUROC | AUPRC | TPR@0.1%FPR | TPR@1%FPR | ECE | p50 ms | p95 ms | Throughput (rps) |
|---|---|---|---|---|---|---|---|---|---|
| `cascade_s1_0.8_s2_0.7` | small | 0.542 | 0.826 | 0.071 | 0.098 | 0.278 | 35.4 | 52.2 | 56 |
| `cascade_s1_0.8_s2_0.7` | medium | 0.834 | 0.952 | 0.414 | 0.605 | 0.250 | 78.1 | 144.8 | 20 |
| `cascade_s1_0.8_s2_0.7` | large | 0.789 | 0.932 | 0.219 | 0.353 | 0.144 | 109.0 | 291.0 | 13 |
| `intent_content_bi_encoder_full` | small | 0.246 | 0.657 | 0.004 | 0.008 | 0.322 | 17.9 | 27.8 | 57 |
| `intent_content_bi_encoder_full` | medium | 0.173 | 0.608 | 0.000 | 0.000 | 0.381 | 77.2 | 142.4 | 13 |
| `intent_content_bi_encoder_full` | large | 0.193 | 0.617 | 0.000 | 0.000 | 0.368 | 262.0 | 465.9 | 4 |
| `intent_content_bi_encoder_summary_first_512` | small | 0.246 | 0.657 | 0.004 | 0.008 | 0.322 | 18.0 | 28.0 | 57 |
| `intent_content_bi_encoder_summary_first_512` | medium | 0.173 | 0.608 | 0.000 | 0.000 | 0.381 | 77.2 | 142.5 | 13 |
| `intent_content_bi_encoder_summary_first_512` | large | 0.193 | 0.617 | 0.000 | 0.000 | 0.368 | 262.2 | 466.0 | 4 |
| `intent_content_cross_encoder_full` | small | 0.797 | 0.931 | 0.259 | 0.309 | 0.522 | 104.5 | 342.2 | 8 |
| `intent_content_cross_encoder_full` | medium | 0.797 | 0.931 | 0.259 | 0.309 | 0.522 | 104.5 | 339.7 | 8 |
| `intent_content_cross_encoder_full` | large | 0.797 | 0.931 | 0.259 | 0.309 | 0.522 | 103.9 | 220.0 | 9 |
| `intra_prompt_outlier_ecod_sentence_max` | small | 0.692 | 0.854 | 0.385 | 0.391 | 0.559 | 3.7 | 33.1 | 102 |
| `intra_prompt_outlier_ecod_sentence_max` | medium | 0.692 | 0.854 | 0.385 | 0.391 | 0.559 | 3.9 | 138.4 | 33 |
| `intra_prompt_outlier_ecod_sentence_max` | large | 0.692 | 0.854 | 0.385 | 0.391 | 0.559 | 4.1 | 453.0 | 11 |
| `intra_prompt_outlier_iforest_sentence_max` | small | 0.692 | 0.854 | 0.385 | 0.391 | 0.501 | 3.8 | 76.2 | 44 |
| `intra_prompt_outlier_iforest_sentence_max` | medium | 0.692 | 0.854 | 0.385 | 0.391 | 0.499 | 3.8 | 178.9 | 23 |
| `intra_prompt_outlier_iforest_sentence_max` | large | 0.692 | 0.854 | 0.385 | 0.391 | 0.498 | 4.1 | 501.6 | 9 |
| `intra_prompt_outlier_lof_sentence_max` | small | 0.692 | 0.854 | 0.385 | 0.391 | 0.567 | 3.6 | 34.1 | 98 |
| `intra_prompt_outlier_lof_sentence_max` | medium | 0.692 | 0.854 | 0.385 | 0.391 | 0.573 | 3.9 | 141.2 | 32 |
| `intra_prompt_outlier_lof_sentence_max` | large | 0.692 | 0.854 | 0.385 | 0.391 | 0.533 | 3.9 | 447.4 | 10 |
| `intra_prompt_outlier_lof_sentence_zscore_max` | small | 0.692 | 0.854 | 0.385 | 0.391 | 0.603 | 3.7 | 34.6 | 100 |
| `intra_prompt_outlier_lof_sentence_zscore_max` | medium | 0.692 | 0.854 | 0.385 | 0.391 | 0.589 | 3.9 | 139.1 | 32 |
| `intra_prompt_outlier_lof_sentence_zscore_max` | large | 0.692 | 0.854 | 0.385 | 0.391 | 0.601 | 3.8 | 444.4 | 11 |
| `threat_prototype_flat_top1_max` | small | 0.738 | 0.913 | 0.236 | 0.295 | 0.238 | 16.0 | 25.0 | 315 |
| `threat_prototype_flat_top1_max` | medium | 0.834 | 0.950 | 0.349 | 0.540 | 0.176 | 35.0 | 99.0 | 43 |
| `threat_prototype_flat_top1_max` | large | 0.789 | 0.932 | 0.219 | 0.353 | 0.179 | 110.0 | 296.0 | 13 |
| `threat_prototype_flat_top5_mean` | small | 0.760 | 0.923 | 0.222 | 0.330 | 0.259 | 16.0 | 26.0 | 313 |
| `threat_prototype_flat_top5_mean` | medium | 0.849 | 0.956 | 0.477 | 0.594 | 0.204 | 35.0 | 103.0 | 43 |
| `threat_prototype_flat_top5_mean` | large | 0.803 | 0.938 | 0.280 | 0.384 | 0.205 | 109.0 | 295.0 | 13 |
| `threat_prototype_hnsw_top5_mean` | small | 0.757 | 0.922 | 0.222 | 0.329 | 0.160 | 6.5 | 14.9 | 324 |
| `threat_prototype_hnsw_top5_mean` | medium | 0.844 | 0.954 | 0.475 | 0.592 | 0.206 | 24.5 | 80.6 | 43 |
| `threat_prototype_hnsw_top5_mean` | large | 0.798 | 0.936 | 0.280 | 0.383 | 0.208 | 81.5 | 262.0 | 13 |

### Per-attack-type AUROC

Each attack type is scored against all 820 benign samples.

| Detector | Embedder | direct_injection | iccd_evasion | ipsod_evasion | tps_evasion |
|---|---|---|---|---|---|
| `cascade_s1_0.8_s2_0.7` | small | 0.482 | 0.471 | 0.993 | 0.764 |
| `cascade_s1_0.8_s2_0.7` | medium | 0.895 | 0.273 | 0.993 | 0.622 |
| `cascade_s1_0.8_s2_0.7` | large | 0.913 | 0.228 | 0.441 | 0.436 |
| `intent_content_bi_encoder_full` | small | 0.165 | 0.471 | 0.324 | 0.764 |
| `intent_content_bi_encoder_full` | medium | 0.118 | 0.289 | 0.127 | 0.662 |
| `intent_content_bi_encoder_full` | large | 0.144 | 0.264 | 0.126 | 0.685 |
| `intent_content_bi_encoder_summary_first_512` | small | 0.165 | 0.471 | 0.324 | 0.764 |
| `intent_content_bi_encoder_summary_first_512` | medium | 0.118 | 0.289 | 0.127 | 0.662 |
| `intent_content_bi_encoder_summary_first_512` | large | 0.144 | 0.264 | 0.126 | 0.685 |
| `intent_content_cross_encoder_full` | small | 0.776 | 0.911 | 0.708 | 0.986 |
| `intent_content_cross_encoder_full` | medium | 0.776 | 0.911 | 0.708 | 0.986 |
| `intent_content_cross_encoder_full` | large | 0.776 | 0.911 | 0.708 | 0.986 |
| `intra_prompt_outlier_ecod_sentence_max` | small | 0.700 | 0.500 | 1.000 | 0.500 |
| `intra_prompt_outlier_ecod_sentence_max` | medium | 0.700 | 0.500 | 1.000 | 0.500 |
| `intra_prompt_outlier_ecod_sentence_max` | large | 0.700 | 0.500 | 1.000 | 0.500 |
| `intra_prompt_outlier_iforest_sentence_max` | small | 0.700 | 0.500 | 1.000 | 0.500 |
| `intra_prompt_outlier_iforest_sentence_max` | medium | 0.700 | 0.500 | 1.000 | 0.500 |
| `intra_prompt_outlier_iforest_sentence_max` | large | 0.700 | 0.500 | 1.000 | 0.500 |
| `intra_prompt_outlier_lof_sentence_max` | small | 0.700 | 0.500 | 1.000 | 0.500 |
| `intra_prompt_outlier_lof_sentence_max` | medium | 0.700 | 0.500 | 1.000 | 0.500 |
| `intra_prompt_outlier_lof_sentence_max` | large | 0.700 | 0.500 | 1.000 | 0.500 |
| `intra_prompt_outlier_lof_sentence_zscore_max` | small | 0.700 | 0.500 | 1.000 | 0.500 |
| `intra_prompt_outlier_lof_sentence_zscore_max` | medium | 0.700 | 0.500 | 1.000 | 0.500 |
| `intra_prompt_outlier_lof_sentence_zscore_max` | large | 0.700 | 0.500 | 1.000 | 0.500 |
| `threat_prototype_flat_top1_max` | small | 0.861 | 0.230 | 0.334 | 0.397 |
| `threat_prototype_flat_top1_max` | medium | 0.935 | 0.252 | 0.749 | 0.477 |
| `threat_prototype_flat_top1_max` | large | 0.913 | 0.228 | 0.441 | 0.436 |
| `threat_prototype_flat_top5_mean` | small | 0.884 | 0.241 | 0.365 | 0.423 |
| `threat_prototype_flat_top5_mean` | medium | 0.944 | 0.265 | 0.823 | 0.500 |
| `threat_prototype_flat_top5_mean` | large | 0.924 | 0.237 | 0.502 | 0.438 |
| `threat_prototype_hnsw_top5_mean` | small | 0.882 | 0.243 | 0.362 | 0.405 |
| `threat_prototype_hnsw_top5_mean` | medium | 0.943 | 0.239 | 0.817 | 0.479 |
| `threat_prototype_hnsw_top5_mean` | large | 0.922 | 0.234 | 0.487 | 0.408 |

## Caveats

- **IPSOD measures prompt length, not outlier structure.** Prompts that cannot be split
  into at least 5 chunks score exactly 0.0. No benign prompt in this dataset reaches
  5 chunks, while 1,012 of 2,631 malicious prompts do. The classifier "has ≥ 5 chunks"
  alone scores AUROC 0.692, which is what every IPSOD variant scores on every embedder.
  The per-attack table shows the same effect: 1.000 on the long `ipsod_evasion` set and
  0.500 (all ties) on the short evasion sets.
- **The ICCD bi-encoder ranks below chance** (AUROC < 0.5). The score is `1 − cosine`
  between description and body, so higher means more mismatched, which is the intended
  direction. In this dataset, malicious bodies are on average closer to their
  descriptions than benign bodies are. The cause has not been established.
- **`summary_first_512` is identical to `full`.** It keeps the first 512 whitespace
  tokens, but the embedders already truncate at 256 or 512 subword tokens, so both
  strategies embed the same text.
- **Cascade thresholds are fixed, not calibrated per embedder.** The stage-1 threshold
  of 0.8 applies to raw cosine similarity, whose range differs by model. The share of
  prompts exiting at stage 1 is 21% (small), 52% (medium), and 99% (large). On the
  large embedder the cascade therefore reduces to `threat_prototype_flat_top1_max`,
  and its metrics match that row.
- **The cross-encoder ignores the embedder.** `intent_content_cross_encoder_full` uses
  `cross-encoder/nli-deberta-v3-base` directly, so its rows repeat across embedders.
- **The benign `none` group has no per-attack AUROC**, since it contains no positives.

## Reproduce

```bash
pip install -e .
tadbench download && tadbench prepare
tadbench matrix --config matrix_config.json --mode full --embedder small --output results/matrix_small.csv
```

Repeat for `medium` and `large`.
