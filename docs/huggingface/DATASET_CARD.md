# TADBench

Versioned evaluation inputs for **TADBench: A Unified Benchmark for Embedding-Based
Threat Detection in LLM Prompts**.

The implementation, issue tracker, and development documentation live at
[github.com/dortort/TADBench](https://github.com/dortort/TADBench).
This dataset repository distributes prepared inputs; it does not contain trained
model weights or establish a leaderboard.

## Loading

```python
from datasets import load_dataset

data = load_dataset("YOUR_NAMESPACE/TADBench", "description", revision="v0.1.0")
validation, test = data["validation"], data["test"]
bank = load_dataset("YOUR_NAMESPACE/TADBench", "threat_bank",
                    revision="v0.1.0", split="train")
```

For reproducible experiments, record the resolved dataset commit SHA as well as
the release tag. Use the exact JSONL files under `data/` with the TADBench CLI;
`manifest.json` contains their SHA-256 hashes and row counts. Re-serializing a
loaded dataset can change bytes and hashes.

## Configurations

| Configuration | Splits | Intended use |
|---|---|---|
| `general` (default) | validation, test | TPS and IPSOD across all sources |
| `description` | validation, test | All techniques, including ICCD and cascade |
| `promptshield` | train, validation, test | Normalized PromptShield |
| `deepset` | train, validation, test | Normalized deepset prompt injections |
| `spml` | train, validation, test | Normalized SPML system/user pairs |
| `synthetic_mismatch` | train, validation, test | TADBench template-generated pairs |
| `adversarial` | test | TADBench TPS/IPSOD/ICCD evasion templates |
| `threat_bank` | train | Malicious source training rows filtered against held-out bodies |

The combined configurations deliberately expose validation and test only. Source
training splits remain available separately; the provided threat bank is the
fitting input for TPS and cascade. Configurations overlap by design: do not
concatenate combined datasets with their component source datasets.

## Schema

| Field | Meaning |
|---|---|
| `id` | UUID assigned during preparation; stable within this release |
| `advertised_description` | Intended task/system prompt; empty when unavailable |
| `prompt_body` | Text to classify |
| `label` | `benign` or `malicious` |
| `attack_type` | Normalized source label or synthetic attack family |
| `source` | `promptshield`, `deepset`, `spml`, `synthetic_mismatch`, or `adversarial` |
| `metadata` | Whitespace word count (`tokens`), language, and optional severity |

`metadata.tokens` is a word-count proxy, not a model tokenizer count. Language is
assigned `en` during transformation rather than independently detected.

## Construction and provenance

See `manifest.json` for exact upstream commits, the code revision, package
versions, preparation parameters, raw Parquet fingerprints, source file hashes,
prepared-file hashes, and counts by label/source. Original upstream dataset cards
are preserved under `provenance/`; attribution and terms are in [NOTICE.md](NOTICE.md).
The preparation code is archived under `provenance/source/` as well.

- PromptShield retains upstream train/validation/test assignments.
- Deepset retains its test split; 20% of training input groups are reserved for validation.
- SPML and synthetic mismatch use approximately 70/15/15 train/validation/test splits.
- Locally created splits are label-stratified with seed 42; identical
  description/body inputs travel together. Grouping can change exact fractions.
- Synthetic mismatch starts with 1,000 benign pairs, 1,000 malicious mismatches,
  and 1,000 cross-category benign hard negatives. Adversarial generation creates
  200 examples per technique, all labeled malicious and used for testing only.
- The threat bank uses malicious PromptShield/deepset/SPML training rows and
  excludes bodies found in held-out source or adversarial inputs.

The export fails if combined validation and test sets share sample IDs or
normalized description/body pairs, if any threat-bank body or ID overlaps them,
or if a threat-bank row is outside malicious source training inputs. This is
exact-match checking; semantic near-duplicates can remain. Source training files
are preserved rather than globally decontaminated; use the filtered threat bank
for benchmark fitting.

Preparation uses UUID4 IDs. Rebuilding with the same seed preserves template and
split choices but does not reproduce IDs or file bytes. Download the archived
JSONL to reproduce this release exactly.

## Intended use and limitations

This release supports research into embedding-based detection of prompt attacks.
Labels describe the source datasets and template generators; they are not proof
of exploit success against a particular deployed model. SPML labels depend on
the accompanying system prompt. The source authors caution against assuming
that training on SPML generalizes to prompt injection generally.

Synthetic examples repeat templates and can expose shortcuts based on length,
wording, or chunk eligibility. Attack-family labels for binary public sources
are coarse (`direct_injection`) rather than independently annotated taxonomies.
Combined sets mix distinct distributions and can be imbalanced; report source
breakdowns and empirical false-positive counts alongside aggregate metrics.
Description-compatible data excludes sources without task descriptions.

Calibrate thresholds on validation only. Very low false-positive-rate targets
may be below the resolution supported by the available benign examples. This
benchmark does not establish production safety or performance on unseen attacks.
Some examples contain malicious instructions or offensive text as research data.

This is a fresh data snapshot. It does not reproduce or replace the historical
October 2026 result tables, which document implementation caveats and require
new runs of the corrected evaluation code.

## Attribution

Please cite the upstream works listed in their preserved cards and identify
TADBench's code revision and this dataset's commit SHA in experimental reports.
