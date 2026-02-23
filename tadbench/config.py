"""Global configuration, paths, defaults, and registries."""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths (all relative to project root, overridable via env vars)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(os.environ.get("TADBENCH_ROOT", Path(__file__).resolve().parent.parent))
DATA_DIR = Path(os.environ.get("TADBENCH_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = Path(os.environ.get("TADBENCH_RAW_DIR", DATA_DIR / "raw"))
PROCESSED_DIR = Path(os.environ.get("TADBENCH_PROCESSED_DIR", DATA_DIR / "processed"))
INDEX_DIR = Path(os.environ.get("TADBENCH_INDEX_DIR", PROJECT_ROOT / "indexes"))
RESULTS_DIR = Path(os.environ.get("TADBENCH_RESULTS_DIR", PROJECT_ROOT / "results"))
THRESHOLDS_DIR = Path(os.environ.get("TADBENCH_THRESHOLDS_DIR", PROJECT_ROOT / "thresholds"))
CACHE_DIR = Path(os.environ.get("TADBENCH_CACHE_DIR", INDEX_DIR / "cache"))

# ---------------------------------------------------------------------------
# Embedding model registry
# ---------------------------------------------------------------------------
EMBEDDER_REGISTRY: dict[str, dict] = {
    "small": {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "dim": 384,
        "max_tokens": 256,
        "prefix": None,
    },
    "medium": {
        "model": "BAAI/bge-base-en-v1.5",
        "dim": 768,
        "max_tokens": 512,
        "prefix": None,
    },
    "large": {
        "model": "intfloat/e5-large-v2",
        "dim": 1024,
        "max_tokens": 512,
        "prefix": {"query": "query: ", "passage": "passage: "},
    },
}

# ---------------------------------------------------------------------------
# Dataset-technique compatibility matrix
# ---------------------------------------------------------------------------
TECHNIQUE_DATASET_COMPAT: dict[str, list[str]] = {
    "threat_prototype": [
        "promptshield", "deepset", "spml", "synthetic_mismatch", "adversarial",
        "benchmark_combined",
    ],
    "intra_prompt_outlier": [
        "promptshield", "deepset", "spml", "synthetic_mismatch", "adversarial",
        "benchmark_combined",
    ],
    "intent_content_consistency": [
        "spml", "synthetic_mismatch",
    ],
    "cascade": [
        "spml", "synthetic_mismatch",
    ],
}

# ---------------------------------------------------------------------------
# Valid schema values
# ---------------------------------------------------------------------------
VALID_LABELS = {"benign", "malicious"}
VALID_ATTACK_TYPES = {
    "none", "direct_injection", "hidden_instruction", "role_override",
    "exfiltration", "jailbreak", "indirect_injection", "tool_misuse",
    "tps_evasion", "ipsod_evasion", "iccd_evasion",
}
VALID_SOURCES = {
    "promptshield", "deepset", "spml", "synthetic_mismatch", "adversarial",
}

# ---------------------------------------------------------------------------
# Default detector hyperparameters
# ---------------------------------------------------------------------------
DEFAULT_DETECTOR_CONFIG: dict[str, dict] = {
    "threat_prototype": {
        "index_type": "flat",
        "top_k": 1,
        "aggregation": "max",
    },
    "intra_prompt_outlier": {
        "outlier_detector": "lof",
        "chunker": "sentence",
        "aggregation": "max",
        "min_chunks": 5,
    },
    "intent_content_consistency": {
        "scorer": "bi_encoder",
        "body_strategy": "full",
        "cross_encoder_model": "cross-encoder/nli-deberta-v3-base",
    },
    "cascade": {
        "stage1_threshold": 0.8,
        "stage2_threshold": 0.7,
    },
}

# ---------------------------------------------------------------------------
# HuggingFace dataset identifiers (pin revisions for reproducibility)
# ---------------------------------------------------------------------------
HF_DATASETS: dict[str, dict[str, str]] = {
    "promptshield": {
        "path": "hendzh/PromptShield",
        "revision": "main",
    },
    "deepset": {
        "path": "deepset/prompt-injections",
        "revision": "main",
    },
    "spml": {
        "path": "reshabhs/SPML_Chatbot_Prompt_Injection",
        "revision": "main",
    },
}
