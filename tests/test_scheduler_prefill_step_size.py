# SPDX-License-Identifier: Apache-2.0
"""Reject prefill chunks that cannot make forward progress."""

import os
import subprocess
import sys

import pytest

from omlx.scheduler import SchedulerConfig


@pytest.mark.parametrize("value", [0, -1, 1.5, 2.0, "4", None, True, False])
def test_invalid_explicit_prefill_step_size(value):
    with pytest.raises(ValueError, match="prefill_step_size.*positive integer"):
        SchedulerConfig(prefill_step_size=value)


@pytest.mark.parametrize("value", [1, 4, 2048])
def test_positive_explicit_prefill_step_size(value):
    assert SchedulerConfig(prefill_step_size=value).prefill_step_size == value


@pytest.mark.parametrize("value", ["0", "-4", "1.5", "invalid", "", None, "1", "4096"])
def test_prefill_step_size_environment_at_import(value):
    # A fresh interpreter tests the startup failure without reloading scheduler
    # classes that the rest of the test session already holds references to.
    env = dict(os.environ)
    env.pop("OMLX_PREFILL_STEP_SIZE", None)
    if value is not None:
        env["OMLX_PREFILL_STEP_SIZE"] = value
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from omlx.scheduler import SchedulerConfig; "
            "print(SchedulerConfig().prefill_step_size)",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    invalid = value in {"0", "-4", "1.5", "invalid", ""}
    assert result.stdout.strip() == ("2048" if invalid or value is None else value)
    if invalid:
        assert "OMLX_PREFILL_STEP_SIZE" in result.stderr
        assert "positive integer" in result.stderr

