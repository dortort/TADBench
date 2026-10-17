# TADBench: A Unified Benchmark for Embedding-Based Threat Detection in LLM Prompts

TADBench is a reproducible benchmark for evaluating embedding-based techniques that detect
malicious prompts before they reach an LLM. It provides standardized datasets, detector
implementations, an evaluation harness, and a CLI that runs the full pipeline from raw
data to a results matrix.

---

## Overview

Three complementary detection techniques are benchmarked, plus a cascade that combines all three:

### Threat-Prototype Similarity (TPS)

Embeds each prompt and performs a FAISS nearest-neighbor search against a bank of known
malicious prompts collected from the training splits. The maximum (or mean) cosine similarity
to the top-k neighbors is the threat score. Supports flat and HNSW FAISS index types.

### Intra-Prompt Semantic Outlier Detection (IPSOD)

Splits each prompt into sentence-level chunks, embeds every chunk, then applies an
outlier detector (LOF, Isolation Forest, or ECOD) to the chunk embedding matrix.
The aggregate outlier score (max or zscore-max) flags prompts that contain at least one
semantically anomalous segment — the hallmark of a hidden injection buried in otherwise
benign content.

### Intent-Content Consistency Detection (ICCD)

Measures the semantic mismatch between the advertised task description and the actual
prompt body. A bi-encoder computes cosine
similarity; a cross-encoder (DeBERTa NLI) provides a second scorer. Low consistency
signals that the prompt body is doing something different from what was described — a
strong indicator of prompt injection or role-override attacks.

### Cascade Pipeline

Chains TPS -> IPSOD -> ICCD with early exit. A sample that exceeds the stage-1 threshold
is immediately classified malicious without running IPSOD or ICCD. A sample that passes
stage 1 but exceeds the stage-2 threshold skips ICCD. Only prompts that pass both gates
reach the most expensive stage. This mirrors production deployment where cheap filters guard
expensive ones. The benchmark records the average stage reached and the fraction of samples
filtered at each gate.

---

## Datasets

| Dataset | Source (HuggingFace) | Approx. size | Notes |
|---|---|---|---|
| PromptShield | `hendzh/PromptShield` | ~43,000 | Pre-split train/val/test |
| deepset | `deepset/prompt-injections` | 662 | Val created from 20% of train |
| SPML | `reshabhs/SPML_Chatbot_Prompt_Injection` | ~16,000 | System prompt field used as `advertised_description`; severity metadata included |
| Synthetic mismatch | Generated | ~3,000 | Benign pairs, malicious injections, hard-negative cross-category pairs |
| Adversarial evasion | Generated | ~600 | 200 TPS-evasion, 200 IPSOD-evasion, 200 ICCD-evasion samples |

All datasets are normalized to the unified `BenchmarkSample` schema (see below).

---

## Embedder Tiers

Three sentence-embedding models are benchmarked to assess the accuracy-latency-memory
tradeoff at different capacity points:

| Tier | Model | Dimension | Max tokens |
|---|---|---|---|
| small | `sentence-transformers/all-MiniLM-L6-v2` | 384 | 256 |
| medium | `BAAI/bge-base-en-v1.5` | 768 | 512 |
| large | `intfloat/e5-large-v2` | 1024 | 512 |

The `large` model uses instruction prefixes (`query: ` / `passage: `) as required by E5.

---

## Installation

Python 3.10-3.12 is required. Python 3.11 is the recommended local default.

```bash
git clone https://github.com/dortort/TADBench
cd TADBench
python3.11 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
```

Core dependencies: `sentence-transformers`, `faiss-cpu`, `scikit-learn`,
`datasets`, `transformers`, `nltk`, `psutil`, `click`, `tqdm`, `numpy`, `pandas`, `h5py`.

---

## Quick Start

```bash
# 1. Download raw datasets from HuggingFace
tadbench download

# 2. Transform to unified JSONL + generate synthetic and adversarial data
tadbench prepare

# 3. Build FAISS index for TPS (required before running TPS or cascade)
tadbench build-index --embedder small

# 4. Calibrate detection thresholds on the validation set
tadbench calibrate --embedder small --dataset data/processed/benchmark_description_val_combined.jsonl

# 5. Run a single technique
tadbench run \
    --technique threat_prototype \
    --dataset data/processed/benchmark_combined.jsonl \
    --embedder small

# 6. Run all techniques and produce a combined CSV
tadbench run \
    --technique all \
    --dataset data/processed/benchmark_description_combined.jsonl \
    --embedder small \
    --output results/all_small.csv

# 7. Run the full benchmark matrix (all embedder x technique x variant combinations)
tadbench matrix --config matrix_config.json --output results/matrix.csv

# 8. Run the same matrix in fast mode when you need a cheaper sweep
tadbench matrix --config matrix_config.json --mode fast --output results/matrix_fast.csv
```

---

## Benchmark Modes

Benchmark execution now supports two explicit modes:

- `full` — preserves the existing benchmark behavior. Use this when you want the most complete efficiency measurements and are willing to pay the runtime cost.
- `fast` — keeps the same detector/dataset/variant combinations, but samples fewer requests for latency and throughput profiling so the benchmark finishes much sooner.

Mode selection is available on both `tadbench run` and `tadbench matrix` via `--mode`.

Guidance:

- Use `tadbench run --mode full` when you want the most faithful per-technique measurement for a focused experiment.
- Use `tadbench matrix --mode fast` for large sweeps, CI smoke runs, or remote preflight comparisons before spending time on a full matrix.
- The output records `benchmark_mode` and profiling sample counts so fast-mode efficiency numbers are identifiable as sampled estimates.

Examples:

```bash
# Single-technique run with lighter profiling
tadbench run \
    --technique threat_prototype \
    --dataset data/processed/benchmark_description_combined.jsonl \
    --embedder small \
    --mode fast

# Full matrix with exhaustive profiling
tadbench matrix \
    --config matrix_config.json \
    --mode full \
    --output results/matrix.csv
```

---

## CLI Commands

All commands are available via the `tadbench` entry point. Run `tadbench --help` or
`tadbench <command> --help` for full option details.

### `tadbench download`

Downloads the three public HuggingFace datasets (PromptShield, deepset, SPML) and saves
each split as Parquet under `data/raw/<dataset>/`.

```
--output-dir PATH   Override default data/raw destination.
```

### `tadbench prepare`

Transforms raw Parquet files to unified JSONL, generates the synthetic mismatch dataset,
generates the adversarial evasion sets, and builds all train/val/test splits, the threat
bank, length-binned subsets, drift holdout files, the general benchmark sets, and
description-compatible benchmark sets used by ICCD, cascade, and `--technique all`.

### `tadbench build-index`

Embeds the threat bank and builds a FAISS index for TPS.

```
--embedder [small|medium|large]   Embedder tier (required).
--index-type [flat|hnsw]          FAISS index type (default: flat).
--threat-bank PATH                Path to threat_bank.jsonl.
```

### `tadbench calibrate`

Searches for the optimal detection threshold on a held-out validation set and writes a
threshold JSON file under `thresholds/`.

```
--embedder [small|medium|large]   Embedder tier (required).
--dataset PATH                    Validation JSONL file (required).
--target-fpr FLOAT                FPR target to calibrate at (default: 0.01).
```

### `tadbench run`

Runs one or all detection techniques against a dataset and reports metrics.

```
--technique TEXT                  threat_prototype | intra_prompt_outlier |
                                  intent_content_consistency | cascade | all
--dataset PATH                    Test JSONL file (required).
--embedder [small|medium|large]   Embedder tier (required).
--detector-config TEXT            Inline JSON or path to JSON file with detector
                                  hyperparameters (overrides defaults).
--threshold-config PATH           Calibrated threshold JSON from tadbench calibrate.
--cache / --no-cache              Cache embeddings across techniques (default: on).
--threat-bank PATH                Override default threat bank path.
--output PATH                     Output path (.json for single, .csv for all).
--mode [full|fast]                Benchmark mode (default: full).
```

For `intent_content_consistency`, `cascade`, and `--technique all`, use a dataset where
every sample has a non-empty `advertised_description`, such as
`data/processed/benchmark_description_combined.jsonl`.

`full` preserves the existing CLI profiling behavior. `fast` reduces the latency and
throughput profiling sample sizes so single runs finish sooner while still reporting
quality metrics on the full dataset.

### `tadbench matrix`

Runs the full benchmark matrix defined in `matrix_config.json`. Each
(technique x variant x embedder x dataset) combination is executed and results are
aggregated into a single CSV.

```
--config PATH                     Matrix config JSON (default: matrix_config.json).
--output PATH                     Output CSV (default: results/matrix_results.csv).
--embedder [small|medium|large]   Override embedder for all runs.
--dry-run                         Print planned runs without executing.
--mode [full|fast]                Benchmark mode (default: full).
--threat-bank PATH                Override default threat bank path.
```

`full` preserves the existing exhaustive matrix profiling behavior, including repeated
per-sample latency measurements across each full dataset slice. `fast` keeps the same
matrix grid but profiles only sampled subsets for latency and throughput, which makes
large sweeps substantially cheaper to run.

### `tadbench drift-eval`

Evaluates detection performance across prompt-length bins and per-attack-type holdout
files to characterize distribution drift.

### `tadbench scaling`

Runs throughput and memory scaling experiments across varying dataset sizes and reports
scaling curves.

---

## Evaluation Metrics

Every `run` and `matrix` invocation reports the following metrics:

**Detection quality**
- `auroc` — Area Under the ROC Curve.
- `auprc` — Area Under the Precision-Recall Curve (average precision).
- `tpr_at_fpr` — True Positive Rate at FPR targets of 0.1%, 0.5%, and 1.0%.
- `calibration_error` — Expected Calibration Error (ECE) across 10 equal-width bins.
- `per_attack_recall` — Per-attack-type recall at the calibrated threshold
  (direct_injection, hidden_instruction, role_override, exfiltration, jailbreak,
  indirect_injection, tool_misuse, tps_evasion, ipsod_evasion, iccd_evasion).

**Efficiency**
- `p50_ms`, `p95_ms`, `p99_ms` — Latency percentiles measured per sample.
- `throughput_rps` — Samples scored per second (batch).
- `rss_mb` — Resident set size in MB after scoring.
- `embedding_calls` — Embedding calls per sample.
- `benchmark_mode` — Whether the run used `full` or `fast` profiling.
- `latency_profile_samples`, `throughput_profile_samples` — How many samples were used
  for the efficiency profiling pass.

**Cascade-specific**
- `avg_stage_reached` — Mean pipeline stage reached (1, 2, or 3).
- `pct_filtered_stage1` — Fraction of traffic exiting at stage 1.
- `pct_filtered_stage2` — Fraction of traffic exiting at stage 1 or 2.

---

## Project Structure

```
TADBench/
├── tadbench/                   # Library source
│   ├── cli/                    # Click commands (download, prepare, run, matrix, ...)
│   ├── data/                   # Schema, download, transform, generate, splits
│   ├── detectors/              # TPS, IPSOD, ICCD, Cascade + base class
│   ├── embedders/              # SentenceTransformer wrapper, cache, protocol
│   ├── evaluation/             # Metrics, calibration, drift, efficiency
│   ├── utils/                  # Chunker, I/O helpers
│   └── config.py               # Paths, registries, valid values, defaults
├── tests/                      # pytest test suite
├── data/
│   ├── raw/                    # Downloaded Parquet files (not committed)
│   └── processed/              # Unified JSONL files (not committed)
├── indexes/                    # FAISS index files (not committed)
├── results/                    # Output CSVs and JSONs (not committed)
├── thresholds/                 # Calibrated threshold JSON files
├── matrix_config.json          # Benchmark matrix specification
├── bench.py                    # Convenience wrapper around the CLI
└── pyproject.toml
```

---

## Configuration

### matrix_config.json

Controls which embedders, technique variants, and datasets are included in a matrix run.
Key fields:

```json
{
    "embedders": ["small", "medium", "large"],
    "techniques": [
        {
            "name": "threat_prototype",
            "variants": [
                {"top_k": 1, "aggregation": "max", "index_type": "flat"},
                {"top_k": 5, "aggregation": "mean", "index_type": "hnsw"}
            ]
        }
    ],
    "datasets": ["data/processed/benchmark_description_combined.jsonl"],
    "target_fprs": [0.001, 0.005, 0.01],
    "length_bins": true,
    "attack_type_breakdown": true,
    "n_efficiency_runs": 3
}
```

### Environment Variables

All paths can be overridden without modifying source code:

| Variable | Default | Purpose |
|---|---|---|
| `TADBENCH_ROOT` | Project root | Base directory |
| `TADBENCH_DATA_DIR` | `<root>/data` | Parent of raw and processed |
| `TADBENCH_RAW_DIR` | `<root>/data/raw` | Downloaded Parquet files |
| `TADBENCH_PROCESSED_DIR` | `<root>/data/processed` | Unified JSONL files |
| `TADBENCH_INDEX_DIR` | `<root>/indexes` | FAISS index files |
| `TADBENCH_RESULTS_DIR` | `<root>/results` | Output CSVs and JSONs |
| `TADBENCH_THRESHOLDS_DIR` | `<root>/thresholds` | Calibrated thresholds |
| `TADBENCH_CACHE_DIR` | `<root>/indexes/cache` | Embedding cache |

---

## License

MIT. See [LICENSE](LICENSE) for the full text.
