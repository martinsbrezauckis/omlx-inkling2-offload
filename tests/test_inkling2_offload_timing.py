# SPDX-License-Identifier: MIT
"""OMLX_INKLING2_TIMING counters: off by default, flushes one INFO line and resets."""
import logging
import os

import pytest

from omlx.patches.inkling2 import moe_offload as mo


@pytest.mark.skipif(os.environ.get("OMLX_INKLING2_TIMING") == "1", reason="enabled by env")
def test_timing_off_by_default():
    assert mo.TIMING is False


def test_flush_logs_and_resets(monkeypatch, caplog):
    mo._T.clear()
    mo._tcount("em_calls", 3)
    mo._tcount("em_bytes", 2e9)
    mo._T["em_wall"] = 1.0
    monkeypatch.setattr(mo, "TIMING_INTERVAL_S", 0.0)
    with caplog.at_level(logging.INFO, logger=mo.logger.name):
        mo._tflush()
    assert "Inkling2 offload timing" in caplog.text and "calls=3" in caplog.text
    assert "2.0GB/s" in caplog.text
    assert mo._T == {}


def test_flush_waits_for_interval(monkeypatch, caplog):
    mo._T.clear()
    mo._tcount("dec_calls")
    monkeypatch.setattr(mo, "TIMING_INTERVAL_S", 1e9)
    with caplog.at_level(logging.INFO, logger=mo.logger.name):
        mo._tflush()
    assert "Inkling2 offload timing" not in caplog.text
    assert mo._T == {"dec_calls": 1}
    mo._T.clear()
