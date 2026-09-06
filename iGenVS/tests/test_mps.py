from __future__ import annotations

import subprocess

from igenvs.mps import start_cuda_mps


def test_automatic_mps_falls_back_when_control_is_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("CUDA_MPS_PIPE_DIRECTORY", raising=False)
    monkeypatch.setattr("igenvs.mps.shutil.which", lambda _: None)
    state = start_cuda_mps(6, scratch_root=tmp_path, automatic=True)
    assert state.workers == 1
    assert state.source == "fallback-no-mps-control"


def test_managed_mps_sets_and_restores_environment(tmp_path, monkeypatch) -> None:
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.delenv("CUDA_MPS_PIPE_DIRECTORY", raising=False)
    monkeypatch.delenv("CUDA_MPS_LOG_DIRECTORY", raising=False)
    monkeypatch.setattr("igenvs.mps.shutil.which", lambda _: "/bin/mps")
    monkeypatch.setattr("igenvs.mps.subprocess.run", fake_run)
    state = start_cuda_mps(4, scratch_root=tmp_path, automatic=True)
    assert state.workers == 4
    assert state.source == "managed-run-private"
    assert state.root is not None and state.root.is_dir()
    state.close()
    assert len(calls) == 2
