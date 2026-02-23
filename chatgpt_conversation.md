# ChatGPT Conversation: Embedding-Based Threat Detection Techniques

> Source: [ChatGPT shared conversation](https://chatgpt.com/share/699c0c5e-ec2c-8006-a29b-4f77440acfd0)
> Model: GPT-5-2 (thinking)

---

## User

for the following three threat detection methods within an LLM prompt using embeddings, what is each one of these techniques called, what libraries or frameworks employ these techniques, and what academic research references these techniques? The first one being the proximity of the embeddings of a prompt to known threats. The second being the divergence within a prompt where the majority of it is in one space, while a few statements are drastically different. And the third approach being the divergence of embeddings of the prompt body compared to its advertised description.

---

## Assistant

Below is a practical mapping of the **three embedding-based threat detection ideas** to (1) the commonly used technique name, (2) libraries/frameworks that implement or support it, and (3) research literature that uses or formalizes the idea. 📌

---

## 1) Proximity of a prompt embedding to known threat embeddings

### Common name
- **Embedding similarity / nearest-neighbor threat matching**
- Also commonly described as:
  - **Vector similarity detection**
  - **Prototype-based detection**
  - **kNN / cosine similarity against known bad prompts**
  - **Vector DB “bad actor” matching**

### Libraries / frameworks that employ it
- **WhyLabs Guardrails / LangKit**  
  WhyLabs documents a guardrail metric that explicitly computes **cosine similarity to known injections stored in a vector DB** (`prompt.similarity.injection`). citeturn5view0turn8view0
- **FAISS (Meta)**  
  Very common for implementing the retrieval/indexing layer for this pattern (ANN / similarity search over embeddings). FAISS explicitly supports L2, dot product, and cosine (via normalized vectors). citeturn9view0
- **Custom pipelines with SBERT / sentence-transformers + FAISS**  
  This is the standard open implementation pattern (embed → index/search → threshold).

### Academic references
- **Ayub & Majumdar (2024), “Embedding-based classifiers can detect prompt injection attacks”**  
  Directly uses prompt embeddings and classical ML classifiers for prompt injection detection. citeturn5view4turn0search9
- **Zizzo et al. (OpenReview benchmark paper)**  
  Explicitly describes **LangKit Injection Detection** as: prompt → embedding → compare to embeddings of known jailbreaks via cosine similarity. citeturn5view2
- **Rashid et al. (2025, MDPI / MIT thesis mirror)**  
  Uses **FAISS** for nearest-neighbor retrieval over message embeddings (IndexFlatIP / cosine-style retrieval). This is the exact mechanics used in proximity matching. citeturn5view3

---

## 2) Intra-prompt divergence (most of prompt is “normal”, a few statements are semantically far away)

### Common name
There is not one universally fixed term in LLM security, but the closest established names are:

- **Embedding-based anomaly detection (text anomaly detection)**
- **Semantic outlier detection**
- **Local outlier detection in embedding space**
- **Segment/chunk-level anomaly detection** (when you split a prompt into sentences/clauses and score each)

Conceptually, this is:
1. Split prompt into sentences/chunks  
2. Embed each chunk  
3. Detect chunks that are outliers relative to the local cluster (the rest of the prompt)

### Libraries / frameworks that employ it
- **PyOD** (Python Outlier Detection)  
  Broad anomaly-detection library (LOF, ECOD, iForest, etc.) used on embedding vectors. citeturn5view6turn10view0
- **scikit-learn**  
  Especially:
  - **LocalOutlierFactor (LOF)** for local density deviation
  - **IsolationForest** for isolation-based anomalies  
  LOF is particularly aligned with your “few statements drastically different from the majority” description. citeturn5view7turn2search16
- **Alibi Detect**  
  Supports outlier/adversarial/drift detection and includes text support; useful for productionizing embedding-based detectors. citeturn5view8

### Academic references
- **Szoplák et al. (2023), “Anomaly detection in texts using sentence embeddings”**  
  Very close to your formulation: they explicitly aim to use sentence embeddings to determine **boundaries of anomalous parts of text**. citeturn5view5
- **TAD-Bench (2025), “A Comprehensive Benchmark for Embedding-Based Text Anomaly Detection”**  
  Systematically studies embedding-based anomaly detection pipelines for text and explicitly describes the two-stage process: text embeddings + anomaly detector. It also notes use of PyOD algorithms (LOF, iForest, etc.). citeturn10view0turn5view6
- **Securing AI Agents Against Prompt Injection Attacks (2025)**  
  Describes a defense layer using **embedding-based anomaly detection** for retrieved content, which is the same pattern applied to prompt/context chunks. citeturn6view1

---

## 3) Divergence between prompt body and its advertised description (semantic mismatch)

### Common name
This is best described as one of the following:

- **Semantic consistency checking**
- **Intent–content alignment / intent–context consistency**
- **Semantic mismatch detection**
- **Entailment/contradiction-based consistency checking** (if using NLI models)
- In prompt-injection literature (especially indirect injection): **semantic context analysis** / **context consistency detection**

This is a **pairwise** technique: compare two texts (e.g., “advertised intent/description” vs actual prompt body or retrieved content) and score whether they are aligned, unrelated, or contradictory.

### Libraries / frameworks that employ it
- **sentence-transformers (SBERT)**  
  Standard choice for bi-encoder semantic similarity (cosine on embeddings). Sentence-Transformers docs explicitly support STS and cosine similarity. citeturn5view10turn5view11
- **Sentence-Transformers CrossEncoder**  
  Useful when you want a higher-accuracy pairwise scorer for “description vs body” (similarity/relevance score for a text pair). citeturn5view13
- **Hugging Face Transformers (NLI-based zero-shot pipeline)**  
  The HF pipeline code explicitly notes the **zero-shot classification pipeline is NLI-based**; this is often used to detect whether one text entails/contradicts another (semantic mismatch framing). citeturn4search7turn5view14
- **WhyLabs Guardrails (analogous pattern)**  
  WhyLabs exposes pairwise similarity metrics such as **response↔prompt** and **response↔context** using embeddings, which is the same architectural idea as “body vs advertised description” consistency scoring. citeturn8view2

### Academic references
- **Alamsabi et al. (Algorithms, 2026), “Embedding-Based Detection of Indirect Prompt Injection Attacks … Using Semantic Context Analysis”**  
  This is the closest direct reference to your third idea. It explicitly:
  - models the **semantic relationship between user intent and external content**
  - formalizes IPIA detection as a **semantic context consistency** problem
  - combines/jointly embeds user intent + external content and classifies benign vs malicious. citeturn6view0
- **SNLI (Bowman et al., 2015)**  
  Foundational NLI formulation (entailment / contradiction / neutral), which underpins “advertised description vs body” contradiction checks. citeturn5view14
- **SBERT (Reimers & Gurevych, 2019)**  
  Foundational sentence embedding work showing semantically similar sentences are close in vector space and enabling scalable semantic similarity comparisons. citeturn5view10turn5view11

---

## Practical taxonomy summary (recommended labels)

If you want a concise naming scheme for documentation/specs, I’d use:

1. **Threat-Prototype Similarity Detection**  
   (embedding proximity to known bad prompts)

2. **Intra-Prompt Semantic Outlier Detection**  
   (chunk/sentence anomaly detection within a single prompt)

3. **Intent–Content Consistency Detection**  
   (semantic mismatch between stated purpose and actual body)

These names are clear, technically accurate, and align with how the literature and tooling are evolving. ✅

---

## URLs
- https://docs.whylabs.ai/docs/secure/guardrail-metrics/
- https://github.com/facebookresearch/faiss
- https://openreview.net/pdf?id=a44MiSFw6G
- https://arxiv.org/html/2410.22284v1
- https://www.mdpi.com/2504-4990/7/3/77
- https://ceur-ws.org/Vol-3498/paper4.pdf
- https://arxiv.org/html/2501.11960v1
- https://pyod.readthedocs.io/en/latest/
- https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html
- https://docs.seldon.ai/alibi-detect
- https://www.researchgate.net/publication/400096377_Embedding-Based_Detection_of_Indirect_Prompt_Injection_Attacks_in_Large_Language_Models_Using_Semantic_Context_Analysis
- https://arxiv.org/pdf/1908.10084
- https://sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html
- https://sbert.net/docs/package_reference/cross_encoder/cross_encoder.html
- https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/zero_shot_classification.py
- https://aclanthology.org/D15-1075.pdf

---

## User

Are there benchmarks that compare the efficiency of each of these three techniques?

---

## Assistant

Yes—but **not a single benchmark that cleanly compares all three techniques head-to-head** (same datasets, same metrics, same latency budget). The current landscape is fragmented. ⚠️

## What exists today

### 1) **Technique 2 (intra-prompt semantic outlier / anomaly detection)** — **best benchmark coverage**
There **are** benchmarks for this family:
- **TAD-Bench** is a dedicated benchmark for **embedding-based text anomaly detection** and explicitly evaluates combinations of embeddings + anomaly detectors across datasets. It also discusses **robustness, efficiency, and generalizability** as benchmark goals. citeturn1view2turn0search3turn4view1
- **Text-ADBench** (2025) is another broad benchmark for embedding-based text anomaly detection and compares shallow vs deep anomaly detectors using multiple embedding backbones. It also highlights an **efficient model/embedding evaluation strategy**. citeturn4view1

### 2) **Technique 1 (proximity to known threats / vector similarity to a threat bank)** — **partial efficiency data, not benchmark-standardized**
There are papers and implementation guidance, but not a canonical benchmark focused on this technique alone:
- Some prompt-injection detection work and industry guidance describe **vector similarity search** over malicious prompt embeddings and provide **latency examples** (e.g., vector DB dataset size vs p95 latency). This is useful for operational efficiency estimates, but it is **not** a standardized benchmark comparing against the other two techniques. citeturn1view4
- Prompt-injection benchmarks like **PromptShield** and **WAInjectBench** evaluate detector performance broadly, but they do **not** isolate “nearest-neighbor similarity to known threats” as a distinct benchmark track with uniform efficiency comparisons. citeturn3view0turn3view1

### 3) **Technique 3 (description↔body semantic mismatch / consistency checking)** — **very limited benchmarking**
This one is the least benchmarked as a standalone category:
- There is emerging work framing indirect prompt injection detection as **semantic consistency/context analysis**, but I did not find a widely adopted benchmark that isolates this technique and reports standardized **efficiency** metrics (latency/throughput/memory) alongside techniques 1 and 2. citeturn0search11turn0search4
- There are lightweight pairwise “semantic drift” methods (embedding-distance between clean and injected variants) that emphasize being computationally lighter, but again this is not part of a unified 3-way benchmark. citeturn3view2

---

## Practical answer to your question

If your question is specifically:

> “Is there a benchmark that compares the efficiency of all three techniques directly?”

**Answer: not really (at least not a standard one).** ✅

What we have instead is:
- **Strong benchmark coverage for anomaly/outlier methods** (Technique 2)
- **Operational latency data and ad hoc evaluations** for vector similarity / nearest-neighbor methods (Technique 1)
- **Task-specific semantic consistency papers** for mismatch detection (Technique 3), but **no common efficiency benchmark** yet

---

## Closest benchmarks you can use (today)

- **For broad prompt-injection detector comparisons** (accuracy-centric, some deployability emphasis):
  - **PromptShield** (general prompt injection detectors, low-FPR evaluation emphasis) citeturn3view0
  - **WAInjectBench** (web-agent prompt injection detectors; text + image attack settings) citeturn3view1turn1view0

- **For efficiency-oriented embedding anomaly pipelines** (closest to Technique 2):
  - **TAD-Bench** / **Text-ADBench** (embedding + anomaly detector combinations) citeturn1view2turn4view1

---

## If you want a rigorous apples-to-apples comparison

The clean way to do it would be to build a small benchmark harness with:
- Same embedder
- Same prompt corpus
- Same hardware
- Same threshold calibration target (e.g., 0.1% FPR)
- Metrics: AUROC/AUPRC + **p50/p95 latency**, **throughput**, **RAM**, **index build time**, **incremental update cost**

That would let you compare:
1. **kNN threat-bank similarity**
2. **Chunk-level outlier detection**
3. **Description↔body consistency scoring (bi-encoder or cross-encoder)**

If you want, I can sketch a benchmark design (datasets + metrics + implementation stack) that covers all three fairly. 📐

## URLs
- https://arxiv.org/abs/2501.11960
- https://huggingface.co/papers/2507.12295
- https://arxiv.org/abs/2501.15145
- https://arxiv.org/html/2501.15145v2
- https://arxiv.org/abs/2510.01354
- https://openreview.net/forum?id=bTYxaNh8R7
- https://media.sciltp.com/articles/2506000841/2506000841.pdf
- https://openreview.net/pdf?id=fMG1EpPkEX
- https://www.mdpi.com/1999-4893/19/1/92
- https://www.researchgate.net/publication/400096377_Embedding-Based_Detection_of_Indirect_Prompt_Injection_Attacks_in_Large_Language_Models_Using_Semantic_Context_Analysis

---

## User

Yes

---

## Assistant

Excellent—here is a **clean benchmark design** you can use to compare the three embedding-based techniques **fairly and efficiently**. 📐

---

## Goal

Compare these three detectors under the **same conditions**:

1. **Threat-Prototype Similarity**  
   (prompt embedding vs known threat embeddings)

2. **Intra-Prompt Semantic Outlier Detection**  
   (sentence/chunk anomalies within a prompt)

3. **Intent–Content Consistency Detection**  
   (advertised description vs prompt body mismatch)

This fills a real gap: current benchmarks are strong for anomaly detection (e.g., TAD-Bench) and broad prompt-injection detection (PromptShield, WAInjectBench), but they do not provide a standardized **3-way efficiency comparison** for your exact techniques. citeturn0search0turn0search1turn0search2

---

## Benchmark design (recommended)

## 1) Standardize the input format

Use a unified schema for every sample:

```json
{
  \{
  "id": "sample_001",
  "advertised_description": "Summarize the attached meeting notes",
  "prompt_body": "Please summarize... [body text here]",
  "label": "benign | malicious",
  "attack_type": "none | direct_injection | hidden_instruction | role_override | exfiltration | tool_misuse",
  "source": "synthetic | PromptShield-like | WAInjectBench-like | internal",
  "metadata": {
    "tokens": 312,
    "language": "en"
  }
}
```
  }
}
```

### Why this matters
- Technique 1 needs `prompt_body`
- Technique 2 needs `prompt_body` chunked into segments
- Technique 3 needs both `advertised_description` and `prompt_body`

---

## 2) Use the same embedding backbone(s)

To avoid unfairness, benchmark with **fixed embedders** across all three methods.

### Suggested embedding tiers
- **Tier A (fast)**: `all-MiniLM-L6-v2`
- **Tier B (balanced)**: `bge-base-en-v1.5` or equivalent
- **Tier C (stronger/slower)**: `e5-large` / `bge-large`

This gives you **accuracy-vs-latency curves** rather than a single point.

---

## 3) Define the three detector implementations precisely

## A) Threat-Prototype Similarity (Technique 1)
### Pipeline
1. Embed `prompt_body`
2. Query ANN index of known malicious prompt embeddings
3. Score = max cosine similarity (or top-k aggregate)
4. Threshold for classification

### Variants to test
- `top-1 cosine`
- `top-k mean`
- `top-k weighted`
- Class-conditional prototype banks (e.g., exfiltration / jailbreak / role override)

### Efficiency knobs
- Index type: Flat vs HNSW vs IVF
- Threat bank size: 1k / 10k / 100k / 1M
- Update frequency (incremental inserts)

---

## B) Intra-Prompt Outlier Detection (Technique 2)
### Pipeline
1. Split `prompt_body` into sentences/clauses/chunks
2. Embed each chunk
3. Run outlier detector over chunk embeddings
4. Aggregate chunk anomaly scores into prompt score

### Outlier detectors to include
- **LOF**
- **Isolation Forest**
- **ECOD** (very fast baseline)
- Optional: one deep detector for completeness

This aligns with embedding-based anomaly pipelines studied in TAD-Bench-style evaluations. citeturn0search0

### Aggregation strategies
- `max(chunk_score)`
- `top-2 mean`
- `% chunks above threshold`
- Robust z-score of chunk distances to centroid

### Efficiency knobs
- Chunking granularity (sentence vs fixed token chunks)
- Avg chunks/prompt (important for latency scaling)

---

## C) Intent–Content Consistency (Technique 3)
### Pipeline options (benchmark both)
#### Option 1: **Bi-encoder similarity** (fast)
1. Embed `advertised_description`
2. Embed `prompt_body` (or summary of body)
3. Score = cosine similarity / distance
4. Threshold or calibrated classifier

#### Option 2: **Cross-encoder / NLI-style consistency** (slower, stronger)
1. Pair (`advertised_description`, `prompt_body`)
2. Score semantic alignment / contradiction
3. Threshold

### Important fairness detail
For long prompt bodies, compare:
- `description` ↔ full body embedding
- `description` ↔ body summary embedding
- `description` ↔ mean of chunk embeddings
- `description` ↔ top-k suspicious chunks

This prevents unfair penalization of long benign prompts.

---

## 4) Datasets (build a composite benchmark)

Since no single benchmark covers all 3 methods, use a **composite suite**:

### A. General prompt injection corpus
Use PromptShield-style data (or equivalent public attack/benign sets) for broad detector evaluation. PromptShield is explicitly benchmark-oriented and deployment-focused. citeturn0search1turn0search10

### B. Web-agent prompt injection corpus
Use WAInjectBench-style text attacks for indirect/web contexts (and optionally ignore images if you want text-only first). WAInjectBench is specifically a benchmark for web-agent prompt injection detection. citeturn0search2turn0search5

### C. Synthetic mismatch set (for Technique 3)
You likely need to generate this:
- Benign: description matches body
- Malicious mismatch: benign-looking description, malicious body
- Hard negatives: loosely related but benign (avoid trivial keyword mismatch)

### D. Mixed-length stress set
Create bins:
- 0–128 tokens
- 129–512
- 513–2k
- 2k+

This is critical for efficiency comparisons.

---

## 5) Metrics to report (accuracy + efficiency)

## Detection quality
Report per technique and overall:
- **AUROC**
- **AUPRC**
- **TPR @ fixed FPR** (e.g., 1%, 0.1%)
- **Calibration error** (optional, but useful)
- **Per-attack-family recall**

PromptShield emphasizes deployment realism and low-FPR behavior, so this metric is especially important. citeturn0search1turn0search10

## Efficiency (the key part)
Measure on the same hardware:
- **p50 / p95 / p99 latency** (ms)
- **Throughput** (req/s)
- **Peak RSS memory**
- **CPU utilization** (or GPU if used)
- **Index build time** (Technique 1)
- **Index update cost** (Technique 1)
- **Latency vs prompt length**
- **Latency vs threat-bank size** (Technique 1)
- **Latency vs chunks/prompt** (Technique 2)
- **Latency vs pair length** (Technique 3)

## Cost proxy (very useful)
- **Embedding calls per prompt**
- **Total tokens embedded**
- **Cross-encoder calls per prompt** (if used)

This makes results portable across infrastructure.

---

## 6) Threshold calibration protocol (avoid hidden bias)

This is where many comparisons become invalid.

### Required protocol
- Split data into:
  - Train (if needed)
  - Validation (threshold calibration)
  - Test (final reporting)

### Calibrate all methods to the same operating point
Example:
- Choose threshold so each detector achieves **0.5% FPR on validation**
- Compare **TPR, p95 latency, memory** on test

This gives an apples-to-apples deployment comparison.

---

## 7) Recommended benchmark matrix

Run this matrix:

### Dimensions
- **Technique**: 1 / 2 / 3
- **Embedder**: small / medium / large
- **Prompt length bin**: short / medium / long / very long
- **Threat type**: direct / indirect / hidden / obfuscated
- **Language**: English (then multilingual later)

### Output format (per row)
- `detector_name`
- `embedder`
- `dataset_slice`
- `FPR_target`
- `TPR`
- `AUROC`
- `p50_ms`
- `p95_ms`
- `rss_mb`
- `notes`

---

## 8) Practical implementation stack (reproducible)

### Core libraries
- **Embeddings**: `sentence-transformers`
- **ANN index (Technique 1)**: `faiss` or `hnswlib`
- **Anomaly detectors (Technique 2)**: `pyod`, `scikit-learn`
- **Consistency detector (Technique 3)**:
  - fast: `sentence-transformers` bi-encoder
  - strong: cross-encoder / NLI model via `transformers`

### Example CLI skeleton
```bash
# 1) Build threat bank index
python bench.py build-index \
  --embedder all-MiniLM-L6-v2 \
  --threat-bank data/threat_bank.jsonl \
  --index faiss_hnsw

# 2) Run technique 1
python bench.py run \
  --technique threat_prototype \
  --dataset data/benchmark_test.jsonl \
  --embedder all-MiniLM-L6-v2 \
  --threshold-config thresholds/fpr_0p5.json

# 3) Run technique 2
python bench.py run \
  --technique intra_prompt_outlier \
  --outlier-detector lof \
  --chunker sentence \
  --dataset data/benchmark_test.jsonl \
  --embedder all-MiniLM-L6-v2

# 4) Run technique 3
python bench.py run \
  --technique intent_content_consistency \
  --scorer bi_encoder \
  --dataset data/benchmark_test.jsonl \
  --embedder all-MiniLM-L6-v2
```

---

## 9) Expected outcomes (what usually happens)

A reasonable hypothesis before running:

### Technique 1 (prototype similarity)
- ✅ **Fastest online latency** (single embed + ANN lookup)
- ✅ Scales well with good indexing
- ❌ Weaker on novel/obfuscated attacks not close to known threats

### Technique 2 (intra-prompt outlier)
- ✅ Good at catching “one malicious clause hidden in benign text”
- ✅ Better novelty detection than pure prototype matching
- ❌ Latency grows with number of chunks
- ❌ Can over-flag heterogeneous but benign prompts

### Technique 3 (intent–content consistency)
- ✅ Best for “advertised benign / actual malicious” mismatch
- ✅ Especially strong on indirect injection scenarios
- ❌ Requires reliable description/intent field
- ❌ Cross-encoder variant can be the slowest

---

## 10) Best-practice additions (highly recommended)

### A) Add a **cascade benchmark**
This is often what wins in production:
1. Technique 1 first (cheap prefilter)
2. Technique 2 only if uncertain
3. Technique 3 for high-risk tools/actions

Measure:
- Quality vs cost reduction
- Average latency under realistic traffic mix

### B) Evaluate **adaptive attacks**
For each technique, include attacks designed to evade it:
- Technique 1 evasion: paraphrases far from threat bank
- Technique 2 evasion: distribute malicious intent across many chunks
- Technique 3 evasion: description subtly aligned but still malicious

### C) Include **drift tests**
Re-test after adding new attack families and measure:
- Retraining cost
- Threshold drift
- Recall degradation

---

## 11) Minimal benchmark spec (if you want to publish it)

A publishable benchmark can be very compact:

- **Datasets**: 3 sources (general PI, web-agent PI, synthetic mismatch)
- **Methods**: 3 techniques × 3 embedders
- **Metrics**: AUROC, TPR@0.5%FPR, p95 latency, memory
- **Hardware**: fixed CPU/GPU profile
- **Artifacts**: code, configs, seeds, raw logs

That would be genuinely useful to the community.

---

## I can also give you a starter repo layout
If you want, I can draft a **benchmark harness skeleton** (Python package layout + configs + evaluation scripts + JSONL schemas) so you can run this immediately.

---

## Sources
- TAD-Bench benchmark (embedding-based text anomaly detection): citeturn0search0turn0search3
- PromptShield benchmark (deployable prompt injection detectors): citeturn0search1turn0search10
- WAInjectBench benchmark (web-agent prompt injection detection): citeturn0search2turn0search5

## URLs
- https://arxiv.org/abs/2501.11960
- https://arxiv.org/abs/2501.15145
- https://arxiv.org/abs/2510.01354
