"""Private subprocess execution shared by tennis-cut modules."""

from __future__ import annotations

import logging
import subprocess
from typing import Sequence


_LOG = logging.getLogger(__name__)


def run_command(command: Sequence[str]) -> None:
    """Run a command and raise with its captured diagnostics on failure."""

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        _LOG.error("Command failed (%s): %s", result.returncode, " ".join(command))
        if result.stdout:
            _LOG.error(result.stdout.strip())
        if result.stderr:
            _LOG.error(result.stderr.strip())
        raise subprocess.CalledProcessError(
            result.returncode, command, result.stdout, result.stderr
        )
