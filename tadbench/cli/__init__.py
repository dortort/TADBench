"""CLI commands for TADBench using Click multi-command group."""

from __future__ import annotations

import click


@click.group()
@click.version_option(package_name="tadbench")
def cli() -> None:
    """TADBench: Embedding-Based Threat Detection Benchmark."""


# Import and register all sub-commands
from tadbench.cli.download_cmd import download  # noqa: E402
from tadbench.cli.prepare_cmd import prepare  # noqa: E402
from tadbench.cli.build_index_cmd import build_index  # noqa: E402
from tadbench.cli.calibrate_cmd import calibrate  # noqa: E402
from tadbench.cli.run_cmd import run  # noqa: E402
from tadbench.cli.matrix_cmd import matrix  # noqa: E402
from tadbench.cli.drift_cmd import drift_eval  # noqa: E402
from tadbench.cli.scaling_cmd import scaling  # noqa: E402

cli.add_command(download)
cli.add_command(prepare)
cli.add_command(build_index)
cli.add_command(calibrate)
cli.add_command(run)
cli.add_command(matrix)
cli.add_command(drift_eval)
cli.add_command(scaling)
