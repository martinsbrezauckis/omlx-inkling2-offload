import importlib

import omlx.patches.moe_expert_offload as m


def test_default_is_twelve(monkeypatch):
    monkeypatch.delenv("OMLX_MOE_IO_WORKERS", raising=False)
    assert m._env_io_workers() == 12


def test_env_override(monkeypatch):
    monkeypatch.setenv("OMLX_MOE_IO_WORKERS", "24")
    assert m._env_io_workers() == 24


def test_invalid_and_negative(monkeypatch):
    monkeypatch.setenv("OMLX_MOE_IO_WORKERS", "lots")
    assert m._env_io_workers() == 12
    monkeypatch.setenv("OMLX_MOE_IO_WORKERS", "-3")
    assert m._env_io_workers() == 0
