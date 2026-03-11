# TADBench Data

This directory contains the raw downloads and processed benchmark files used by TADBench.
Raw data is not committed to version control. Run `tadbench download` followed by
`tadbench prepare` to populate both subdirectories.

---

## Data Pipeline

```
tadbench download
    |
    v
data/raw/<dataset>/<split>.parquet   (HuggingFace datasets, pinned revisions)
    |
    v  tadbench prepare (step 1: transform)
    |
data/processed/<dataset>_<split>.jsonl   (unified BenchmarkSample JSONL)
    |
    v  tadbench prepare (step 2: generate synthetic)
    |
data/processed/synthetic_mismatch.jsonl
    |
    v  tadbench prepare (step 3: generate adversarial)
    |
data/processed/adversarial_tps_evasion.jsonl
data/processed/adversarial_ipsod_evasion.jsonl
data/processed/adversarial_iccd_evasion.jsonl
    |
    v  tadbench prepare (step 4: split)
    |
data/processed/threat_bank.jsonl
data/processed/<dataset>_{train,val,test}.jsonl
data/processed/benchmark_test_{short,medium,long,very_long}.jsonl
data/processed/drift_holdout_<attack_type>.jsonl
data/processed/benchmark_combined.jsonl
data/processed/benchmark_val_combined.jsonl
data/processed/benchmark_description_combined.jsonl
data/processed/benchmark_description_val_combined.jsonl
```

---

## Source Datasets

### PromptShield

- **HuggingFace path**: `hendzh/PromptShield` (revision: `main`)
- **Approximate size**: ~43,000 samples across three splits
- **HuggingFace splits**: train, validation, test (used as-is; validation renamed to `val`)
- **Raw columns used**: `prompt` (mapped to `prompt_body`), `label` (0/1 integer)
- **Notes**: No system-prompt field is available. `advertised_description` is always empty.
  Attack type is set to `direct_injection` for malicious samples and `none` for benign.

### deepset

- **HuggingFace path**: `deepset/prompt-injections` (revision: `main`)
- **Approximate size**: 662 samples across train and test
- **HuggingFace splits**: train, test
- **Raw columns used**: `text` (mapped to `prompt_body`), `label` (0/1 integer)
- **Split handling**: A stratified 80/20 split of the HuggingFace train set produces the
  TADBench `train` and `val` files. The HuggingFace `test` split is used directly.
- **Notes**: `advertised_description` is always empty.

### SPML (SPML_Chatbot_Prompt_Injection)

- **HuggingFace path**: `reshabhs/SPML_Chatbot_Prompt_Injection` (revision: `main`)
- **Approximate size**: ~16,000 samples
- **HuggingFace splits**: train (single large split)
- **Raw columns used**:
  - `User Prompt` -> `prompt_body`
  - `System Prompt` -> `advertised_description`
  - `Prompt injection` (0/1) -> `label`
  - `Degree` -> `metadata.severity` (integer, when present)
- **Split handling**: Stratified 70/15/15 split into train/val/test.
- **Notes**: SPML is the only source dataset that provides a system prompt, making it
  compatible with ICCD and the cascade pipeline without synthetic augmentation.

---

## Unified Schema

Every processed JSONL file contains one JSON object per line conforming to `BenchmarkSample`:

```json
{
    "id":                     "<uuid-string>",
    "advertised_description": "<string>",
    "prompt_body":            "<string>",
    "label":                  "benign" | "malicious",
    "attack_type":            "<string>",
    "source":                 "<string>",
    "metadata": {
        "tokens":    <int>,
        "language":  "en",
        "severity":  <int | null>
    }
}
```

### Field Details

| Field | Type | Description |
|---|---|---|
| `id` | string | UUID, auto-generated if not present in source. |
| `advertised_description` | string | The operator-supplied system prompt or task description. Empty string when unavailable. Required to be non-empty for ICCD and the cascade stage 3. |
| `prompt_body` | string | The full prompt text submitted by the user. |
| `label` | string | `"benign"` or `"malicious"`. |
| `attack_type` | string | Category of threat. See valid values below. |
| `source` | string | Dataset origin. See valid values below. |
| `metadata.tokens` | int | Approximate token count (whitespace-split word count). |
| `metadata.language` | string | Always `"en"` for all current datasets. |
| `metadata.severity` | int or null | Injection severity level from SPML `Degree` field; absent for other sources. |

### Valid `attack_type` Values

| Value | Meaning |
|---|---|
| `none` | Benign sample; no attack. |
| `direct_injection` | Direct prompt injection. |
| `hidden_instruction` | Instruction hidden within otherwise benign content. |
| `role_override` | Attempt to redefine the model's role or persona. |
| `exfiltration` | Attempt to leak data or system context. |
| `jailbreak` | Attempt to bypass safety guidelines. |
| `indirect_injection` | Injection delivered through retrieved or external content. |
| `tool_misuse` | Misuse of tool-call or function-calling capabilities. |
| `tps_evasion` | Adversarially paraphrased injection designed to evade TPS. |
| `ipsod_evasion` | Injection distributed across sentences to evade IPSOD. |
| `iccd_evasion` | Injection paired with a semantically aligned description to evade ICCD. |

### Valid `source` Values

`promptshield`, `deepset`, `spml`, `synthetic_mismatch`, `adversarial`

---

## Generated Datasets

### Synthetic Mismatch (`synthetic_mismatch.jsonl`)

Generated by `tadbench/data/generate_synthetic.py` with `seed=42`.
Default composition (3,000 samples total):

| Category | Count | Label | Description |
|---|---|---|---|
| Benign matching pairs | 1,000 | `benign` | Description and body drawn from the same task category (summarize, translate, write, analyze, explain, compare, list, plan, review, format). |
| Malicious mismatch | 1,000 | `malicious` | Benign-looking description paired with a malicious injection body. `attack_type = direct_injection`. |
| Hard-negative cross-category | 1,000 | `benign` | Description and body drawn from different task categories. Tests whether ICCD can distinguish genuine mismatch from legitimate task variation. |

All samples have non-empty `advertised_description`, making the full set compatible with
ICCD and the cascade.

### Adversarial Evasion

Generated by `tadbench/data/generate_adversarial.py` with `seed=42`.
All adversarial samples are labeled `malicious`.

| File | Count | `attack_type` | Evasion strategy |
|---|---|---|---|
| `adversarial_tps_evasion.jsonl` | 200 | `tps_evasion` | Synonym substitution, sentence restructuring, and filler-phrase injection to paraphrase known injections away from the threat bank. |
| `adversarial_ipsod_evasion.jsonl` | 200 | `ipsod_evasion` | Malicious intent distributed across 4-6 individually innocuous sentences, padded with benign fragments to dilute the outlier signal. |
| `adversarial_iccd_evasion.jsonl` | 200 | `iccd_evasion` | Descriptions phrased as legitimate security-research or penetration-testing requests, semantically aligned enough to suppress the ICCD mismatch score. |

---

## Splits

### Per-Dataset Splits

| File | Content |
|---|---|
| `promptshield_train.jsonl` | PromptShield train split (HuggingFace) |
| `promptshield_val.jsonl` | PromptShield validation split (renamed from HuggingFace `validation`) |
| `promptshield_test.jsonl` | PromptShield test split (HuggingFace) |
| `deepset_train.jsonl` | 80% stratified sample of HuggingFace deepset train |
| `deepset_val.jsonl` | 20% stratified holdout from HuggingFace deepset train |
| `deepset_test.jsonl` | deepset test split (HuggingFace) |
| `spml_train.jsonl` | 70% stratified sample of SPML |
| `spml_val.jsonl` | 15% stratified sample of SPML |
| `spml_test.jsonl` | 15% stratified sample of SPML |
| `synthetic_mismatch_train.jsonl` | 70% stratified sample of synthetic mismatch |
| `synthetic_mismatch_val.jsonl` | 15% stratified sample of synthetic mismatch |
| `synthetic_mismatch_test.jsonl` | 15% stratified sample of synthetic mismatch |

All stratified splits preserve the benign/malicious class ratio within each partition
(`seed=42`).

### Threat Bank

`threat_bank.jsonl` — Malicious samples extracted from the train splits of PromptShield,
deepset, and SPML. Used as the reference prototype set for TPS and as the FAISS index
source. Not used as evaluation data.

### Combined Sets

| File | Content |
|---|---|
| `benchmark_combined.jsonl` | Union of all test splits plus all adversarial evasion files. Primary evaluation file for TPS and IPSOD. |
| `benchmark_val_combined.jsonl` | Union of all val splits (PromptShield, deepset, SPML, synthetic mismatch). General validation file for TPS and IPSOD. |
| `benchmark_description_combined.jsonl` | Description-compatible subset of the combined benchmark. Use for ICCD, cascade, and `tadbench run --technique all`. |
| `benchmark_description_val_combined.jsonl` | Description-compatible validation subset. Use for ICCD, cascade, and full-technique calibration. |

### Length-Binned Subsets

Built from the union of the four per-dataset test splits. Binned by approximate token
count (whitespace-split word count):

| File | Token range |
|---|---|
| `benchmark_test_short.jsonl` | 0 – 128 tokens |
| `benchmark_test_medium.jsonl` | 129 – 512 tokens |
| `benchmark_test_long.jsonl` | 513 – 2,048 tokens |
| `benchmark_test_very_long.jsonl` | 2,049+ tokens |

Used by `tadbench drift-eval` to measure whether detection quality degrades on longer
prompts.

### Drift Holdout Files

One file per non-`none` attack type present in the processed data, containing all samples
of that type regardless of split origin. Files with fewer than 10 samples are skipped.

Pattern: `drift_holdout_<attack_type>.jsonl`

Examples: `drift_holdout_direct_injection.jsonl`, `drift_holdout_tps_evasion.jsonl`,
`drift_holdout_iccd_evasion.jsonl`.

---

## Directory Structure

After a full `tadbench download` and `tadbench prepare` run the data directory will
contain:

```
data/
├── raw/
│   ├── promptshield/
│   │   ├── train.parquet
│   │   ├── validation.parquet
│   │   └── test.parquet
│   ├── deepset/
│   │   ├── train.parquet
│   │   └── test.parquet
│   └── spml/
│       └── train.parquet
└── processed/
    ├── promptshield_train.jsonl
    ├── promptshield_val.jsonl
    ├── promptshield_test.jsonl
    ├── deepset_train.jsonl
    ├── deepset_val.jsonl
    ├── deepset_test.jsonl
    ├── spml_train.jsonl
    ├── spml_val.jsonl
    ├── spml_test.jsonl
    ├── synthetic_mismatch.jsonl
    ├── synthetic_mismatch_train.jsonl
    ├── synthetic_mismatch_val.jsonl
    ├── synthetic_mismatch_test.jsonl
    ├── adversarial_tps_evasion.jsonl
    ├── adversarial_ipsod_evasion.jsonl
    ├── adversarial_iccd_evasion.jsonl
    ├── threat_bank.jsonl
    ├── benchmark_combined.jsonl
    ├── benchmark_val_combined.jsonl
    ├── benchmark_test_short.jsonl
    ├── benchmark_test_medium.jsonl
    ├── benchmark_test_long.jsonl
    ├── benchmark_test_very_long.jsonl
    ├── drift_holdout_direct_injection.jsonl
    ├── drift_holdout_tps_evasion.jsonl
    ├── drift_holdout_ipsod_evasion.jsonl
    └── drift_holdout_iccd_evasion.jsonl
```
