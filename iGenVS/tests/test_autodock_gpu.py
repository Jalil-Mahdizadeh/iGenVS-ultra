from __future__ import annotations

from pathlib import Path

import pytest

import igenvs.autodock_gpu as backend
from igenvs.autodock_gpu import (
    autodock_gpu_pose_path,
    autodock_gpu_xml_path,
    build_autodock_gpu_command,
    dock_batch_resilient_autodock_gpu,
    parse_autodock_gpu_xml,
    partition_autodock_gpu_ligands,
    resolved_autodock_gpu_runs,
)
from igenvs.docking import DockInvocation, DockingConfig
from igenvs.records import PreparedLigand, ValidatedRecord


def _config(tmp_path: Path, **overrides: object) -> DockingConfig:
    values: dict[str, object] = {
        "receptor": tmp_path / "receptor.pdbqt",
        "center": (1.0, 2.0, 3.0),
        "size": (20.0, 21.0, 22.0),
        "engine": "autodock-gpu",
        "scoring": "ad4",
        "autodock_gpu_fld": tmp_path / "receptor.maps.fld",
    }
    values.update(overrides)
    return DockingConfig(**values)  # type: ignore[arg-type]


def _ligand(tmp_path: Path, stem: str = "lig_0001") -> PreparedLigand:
    path = tmp_path / f"{stem}.pdbqt"
    path.write_text("ROOT\nENDROOT\nTORSDOF 0\n", encoding="utf-8")
    record = ValidatedRecord("mol-1", "CCO", "CCO", 1, 3, 0)
    return PreparedLigand(record, path, 3, 0, 0.1)


def _flexible_ligand(
    tmp_path: Path,
    index: int,
    torsions: int,
) -> PreparedLigand:
    ligand = _ligand(tmp_path, f"lig_{index:04d}")
    return PreparedLigand(
        ValidatedRecord(
            f"mol-{index}",
            "CCO",
            "CCO",
            index + 1,
            3,
            torsions,
        ),
        ligand.path,
        3,
        torsions,
        0.1,
    )


def test_autodock_gpu_command_uses_v16_batch_contract(tmp_path: Path) -> None:
    config = _config(tmp_path, device_id=2, scores_only_output=True)
    command = build_autodock_gpu_command(
        config=config,
        filelist_path=tmp_path / "batch.txt",
    )

    assert command[command.index("--filelist") + 1] == str(tmp_path / "batch.txt")
    assert command[command.index("--devnum") + 1] == "3"
    assert command[command.index("--nrun") + 1] == "20"
    assert command[command.index("--heuristics") + 1] == "1"
    assert command[command.index("--autostop") + 1] == "1"
    assert command[command.index("--lsmet") + 1] == "ad"
    assert command[command.index("--dlgoutput") + 1] == "0"
    assert command[command.index("--clustering") + 1] == "0"
    assert command[command.index("--gbest") + 1] == "0"


@pytest.mark.parametrize(
    ("mode", "runs"),
    [("fast", 10), ("balance", 20), ("detail", 50)],
)
def test_search_modes_map_to_explicit_lga_runs(
    tmp_path: Path,
    mode: str,
    runs: int,
) -> None:
    assert resolved_autodock_gpu_runs(_config(tmp_path, search_mode=mode)) == runs
    assert (
        resolved_autodock_gpu_runs(
            _config(tmp_path, search_mode=mode, autodock_gpu_runs=7)
        )
        == 7
    )


def test_xml_parser_returns_best_finite_ad4_score(tmp_path: Path) -> None:
    output = tmp_path / "ligand.xml"
    output.write_text(
        """<?xml version="1.0"?>
<autodock_gpu>
  <runs>
    <run id="1"><free_NRG_binding>-7.25</free_NRG_binding></run>
    <run id="2"><free_NRG_binding>-8.50</free_NRG_binding></run>
  </runs>
</autodock_gpu>
""",
        encoding="utf-8",
    )
    assert parse_autodock_gpu_xml(output) == (-8.5,)

    output.write_text(
        "<autodock_gpu><runs><run><free_NRG_binding>nan"
        "</free_NRG_binding></run></runs></autodock_gpu>",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-finite"):
        parse_autodock_gpu_xml(output)


def test_best_pose_and_xml_paths_are_unambiguous(tmp_path: Path) -> None:
    ligand = _ligand(tmp_path)
    output_dir = tmp_path / "outputs"
    assert autodock_gpu_xml_path(ligand, output_dir) == output_dir / "lig_0001.xml"
    assert (
        autodock_gpu_pose_path(ligand, output_dir)
        == output_dir / "lig_0001-best.pdbqt"
    )


def test_backend_rejects_non_ad4_scoring(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="AD4"):
        build_autodock_gpu_command(
            config=_config(tmp_path, scoring="vina"),
            filelist_path=tmp_path / "batch.txt",
        )


def test_multiple_workers_split_filelists_and_collect_all_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ligands = [_ligand(tmp_path, f"lig_{index:04d}") for index in range(4)]
    calls: list[tuple[str, ...]] = []

    def fake_invoke(
        group,
        *,
        config: DockingConfig,
        work_dir: Path,
        label: str,
    ) -> DockInvocation:
        calls.append(tuple(ligand.path.stem for ligand in group))
        output_dir = work_dir / f"{label}_outputs"
        output_dir.mkdir(parents=True)
        for ligand in group:
            autodock_gpu_xml_path(ligand, output_dir).write_text(
                "<autodock_gpu><runs><run><free_NRG_binding>-7.5"
                "</free_NRG_binding></run></runs></autodock_gpu>",
                encoding="utf-8",
            )
        stdout_path = work_dir / f"{label}.stdout.log"
        stderr_path = work_dir / f"{label}.stderr.log"
        stdout_path.write_text("ok\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return DockInvocation(
            0,
            0.25,
            stdout_path,
            stderr_path,
            None,
            ("autodock_gpu",),
        )

    monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", str(tmp_path / "mps"))
    monkeypatch.setattr(backend, "invoke_autodock_gpu", fake_invoke)
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    results, invocations = dock_batch_resilient_autodock_gpu(
        ligands,
        config=_config(
            tmp_path,
            autodock_gpu_workers=2,
            scores_only_output=True,
        ),
        work_dir=work_dir,
        label="batch",
        retry_missing=False,
    )

    assert sorted(calls) == [
        ("lig_0000", "lig_0002"),
        ("lig_0001", "lig_0003"),
    ]
    assert len(invocations) == 2
    assert len(results) == 4
    assert all(result.status == "success" for result in results)
    assert [result.ligand.path.stem for result in results] == [
        "lig_0000",
        "lig_0001",
        "lig_0002",
        "lig_0003",
    ]


def test_worker_partition_balances_flexible_ligands_deterministically(
    tmp_path: Path,
) -> None:
    ligands = [
        _flexible_ligand(tmp_path, index, torsions)
        for index, torsions in enumerate([12, 11, 10, 2, 1, 0])
    ]
    groups = partition_autodock_gpu_ligands(ligands, 3)
    loads = [sum(ligand.torsion_count + 1 for ligand in group) for group in groups]
    assert max(loads) - min(loads) <= 1
    assert sorted(ligand.record.source_row for group in groups for ligand in group) == list(
        range(1, 7)
    )
    assert all(
        [ligand.record.source_row for ligand in group]
        == sorted(ligand.record.source_row for ligand in group)
        for group in groups
    )


def test_multiple_workers_require_explicit_mps_pipe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_MPS_PIPE_DIRECTORY", raising=False)
    with pytest.raises(ValueError, match="CUDA MPS"):
        dock_batch_resilient_autodock_gpu(
            [_ligand(tmp_path)],
            config=_config(tmp_path, autodock_gpu_workers=2),
            work_dir=tmp_path,
            label="batch",
        )
