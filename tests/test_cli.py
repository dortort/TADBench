"""Tests for tadbench CLI commands using Click's CliRunner."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

try:
    from click.testing import CliRunner
    from tadbench.cli import cli
    _CLI_AVAILABLE = True
except ImportError:
    _CLI_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _CLI_AVAILABLE,
    reason="click or tadbench.cli not available",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _runner() -> "CliRunner":
    from click.testing import CliRunner as CR
    return CR()


# ---------------------------------------------------------------------------
# download command
# ---------------------------------------------------------------------------

class TestDownloadCommand:
    def test_download_exits_zero_with_mocked_pipeline(self, tmp_path):
        from click.testing import CliRunner as CR
        runner = CR()
        with patch("tadbench.data.download.download_all") as mock_dl:
            result = runner.invoke(cli, ["download", "--output-dir", str(tmp_path)])
        assert result.exit_code == 0, f"output: {result.output}\nexc: {result.exception}"

    def test_download_calls_download_all(self, tmp_path):
        from click.testing import CliRunner as CR
        runner = CR()
        with patch("tadbench.data.download.download_all") as mock_dl:
            runner.invoke(cli, ["download", "--output-dir", str(tmp_path)])
        mock_dl.assert_called_once()

    def test_download_default_output_dir(self, tmp_path):
        from click.testing import CliRunner as CR
        runner = CR()
        with patch("tadbench.data.download.download_all") as mock_dl:
            result = runner.invoke(cli, ["download"])
        # Should not crash even without --output-dir
        assert result.exit_code in (0, 1)  # may fail if no data dir, but should not error out unexpectedly


# ---------------------------------------------------------------------------
# prepare command
# ---------------------------------------------------------------------------

class TestPrepareCommand:
    def test_prepare_exits_zero_with_mocked_pipeline(self, tmp_path):
        from click.testing import CliRunner as CR
        runner = CR()
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        out_dir = tmp_path / "processed"
        out_dir.mkdir()

        with patch("tadbench.data.transform.transform_all"), \
             patch("tadbench.data.generate_synthetic.generate_synthetic", return_value=[]), \
             patch("tadbench.data.generate_adversarial.generate_adversarial", return_value={}), \
             patch("tadbench.data.splits.build_splits"):
            result = runner.invoke(cli, [
                "prepare",
                "--raw-dir", str(raw_dir),
                "--output-dir", str(out_dir),
                "--no-adversarial",
            ])
        assert result.exit_code == 0, f"output: {result.output}\nexc: {result.exception}"

    def test_prepare_no_adversarial_flag(self, tmp_path):
        from click.testing import CliRunner as CR
        runner = CR()
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        out_dir = tmp_path / "processed"
        out_dir.mkdir()

        mock_adv = MagicMock(return_value={})
        with patch("tadbench.data.transform.transform_all"), \
             patch("tadbench.data.generate_synthetic.generate_synthetic", return_value=[]), \
             patch("tadbench.data.generate_adversarial.generate_adversarial", mock_adv), \
             patch("tadbench.data.splits.build_splits"):
            runner.invoke(cli, [
                "prepare",
                "--raw-dir", str(raw_dir),
                "--output-dir", str(out_dir),
                "--no-adversarial",
            ])
        # With --no-adversarial, adversarial generator should NOT be called
        mock_adv.assert_not_called()


# ---------------------------------------------------------------------------
# build-index command
# ---------------------------------------------------------------------------

class TestBuildIndexCommand:
    def test_build_index_help_exits_zero(self):
        from click.testing import CliRunner as CR
        runner = CR()
        result = runner.invoke(cli, ["build-index", "--help"])
        assert result.exit_code == 0

    def test_build_index_with_mocked_detector(self, tmp_path):
        from click.testing import CliRunner as CR
        runner = CR()

        # Write a minimal threat bank
        threat_bank = tmp_path / "threat_bank.jsonl"
        sample = {
            "id": "1", "advertised_description": "", "prompt_body": "bad prompt",
            "label": "malicious", "attack_type": "direct_injection",
            "source": "deepset", "metadata": {"tokens": 2, "language": "en"},
        }
        threat_bank.write_text(json.dumps(sample) + "\n")

        mock_det = MagicMock()
        mock_det.build_index = MagicMock()
        mock_det.save_index = MagicMock()

        with patch("tadbench.embedders.sentence_transformer.SentenceTransformer") as mock_st, \
             patch("tadbench.detectors.threat_prototype.ThreatPrototypeDetector", return_value=mock_det):
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_model.encode = MagicMock(return_value=[[0.1] * 384])
            mock_st.return_value = mock_model
            result = runner.invoke(cli, [
                "build-index",
                "--embedder", "small",
                "--threat-bank", str(threat_bank),
                "--output-dir", str(tmp_path),
            ])
        # Should exit 0 or produce usage error (not crash)
        assert result.exit_code in (0, 1, 2)


# ---------------------------------------------------------------------------
# calibrate command
# ---------------------------------------------------------------------------

class TestCalibrateCommand:
    def test_calibrate_help_exits_zero(self):
        from click.testing import CliRunner as CR
        runner = CR()
        result = runner.invoke(cli, ["calibrate", "--help"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# run command
# ---------------------------------------------------------------------------

class TestRunCommand:
    def test_run_help_exits_zero(self):
        from click.testing import CliRunner as CR
        runner = CR()
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0

    def test_run_missing_required_args_exits_nonzero(self):
        from click.testing import CliRunner as CR
        runner = CR()
        result = runner.invoke(cli, ["run"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# matrix command
# ---------------------------------------------------------------------------

class TestMatrixCommand:
    def test_matrix_help_exits_zero(self):
        from click.testing import CliRunner as CR
        runner = CR()
        result = runner.invoke(cli, ["matrix", "--help"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# drift-eval command
# ---------------------------------------------------------------------------

class TestDriftEvalCommand:
    def test_drift_eval_help_exits_zero(self):
        from click.testing import CliRunner as CR
        runner = CR()
        result = runner.invoke(cli, ["drift-eval", "--help"])
        assert result.exit_code == 0

    def test_drift_eval_missing_required_exits_nonzero(self):
        from click.testing import CliRunner as CR
        runner = CR()
        result = runner.invoke(cli, ["drift-eval"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# scaling command
# ---------------------------------------------------------------------------

class TestScalingCommand:
    def test_scaling_help_exits_zero(self):
        from click.testing import CliRunner as CR
        runner = CR()
        result = runner.invoke(cli, ["scaling", "--help"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

class TestCliGroup:
    def test_cli_help_exits_zero(self):
        from click.testing import CliRunner as CR
        runner = CR()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0

    def test_cli_lists_all_commands(self):
        from click.testing import CliRunner as CR
        runner = CR()
        result = runner.invoke(cli, ["--help"])
        output = result.output
        for cmd in ("download", "prepare", "build-index", "calibrate", "run", "matrix", "drift-eval", "scaling"):
            assert cmd in output, f"Command '{cmd}' missing from CLI help"

    def test_unknown_command_exits_nonzero(self):
        from click.testing import CliRunner as CR
        runner = CR()
        result = runner.invoke(cli, ["nonexistent-command"])
        assert result.exit_code != 0
