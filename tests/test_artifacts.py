import hashlib
import json
from pathlib import Path

import numpy as np

from tadbench.detectors.intra_prompt_outlier import IntraPromptOutlierDetector
from tadbench.evaluation.artifacts import write_run_artifacts


def test_artifacts_capture_inputs_scores_and_length_baseline(tmp_path, mock_embedder, sample_data):
    data = tmp_path / "data.jsonl"
    data.write_text("original input\n")
    detector = IntraPromptOutlierDetector(mock_embedder)
    path = Path(write_run_artifacts(tmp_path, sample_data, np.zeros(len(sample_data)),
                                   {"test": data}, {"technique": "ipsod"}, detector))
    manifest = json.loads((path / "manifest.json").read_text())
    predictions = [json.loads(line) for line in (path / "predictions.jsonl").read_text().splitlines()]
    assert manifest["inputs"]["test"]["sha256"] == hashlib.sha256(data.read_bytes()).hexdigest()
    assert len(predictions) == len(sample_data)
    assert manifest["diagnostics"]["chunk_eligible_by_label"] == {"benign": 0, "malicious": 0}
    assert manifest["diagnostics"]["chunk_eligibility_baseline"]["auroc"] == .5
    assert manifest["diagnostics"]["chunk_eligibility_baseline"]["calibration_error"] is None
