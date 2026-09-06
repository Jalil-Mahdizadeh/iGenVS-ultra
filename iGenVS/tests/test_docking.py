from __future__ import annotations

from pathlib import Path

import pytest

from igenvs.docking import (
    DockingConfig,
    build_unidock_command,
    expected_output_path,
    parse_pdbqt_scores,
    parse_score_table,
)
from igenvs.records import PreparedLigand, ValidatedRecord


def _ligand(tmp_path: Path) -> PreparedLigand:
    record = ValidatedRecord("mol-1", "CCO", "CCO", 1, 3, 0)
    path = tmp_path / "ligand.pdbqt"
    path.write_text("MODEL 1\nENDMDL\n", encoding="utf-8")
    return PreparedLigand(record, path, 3, 0, 0.1)


def test_score_parser_requires_finite_scores(tmp_path: Path) -> None:
    output = tmp_path / "out.pdbqt"
    output.write_text(
        "REMARK VINA RESULT: -7.500 0.0 0.0\n"
        "REMARK VINA RESULT: -6.250 1.0 2.0\n",
        encoding="utf-8",
    )
    assert parse_pdbqt_scores(output) == (-7.5, -6.25)

    output.write_text("REMARK VINA RESULT: 1e999 0.0 0.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        parse_pdbqt_scores(output)


def test_command_contains_batch_and_memory_controls(tmp_path: Path) -> None:
    ligand = _ligand(tmp_path)
    config = DockingConfig(
        receptor=tmp_path / "receptor.pdbqt",
        center=(1.0, 2.0, 3.0),
        size=(20.0, 21.0, 22.0),
        max_gpu_memory=4096,
    )
    command = build_unidock_command(
        [ligand],
        config=config,
        index_path=tmp_path / "index.txt",
        output_dir=tmp_path / "out",
    )
    assert command[command.index("--search_mode") + 1] == "balance"
    assert command[command.index("--max_gpu_memory") + 1] == "4096"
    assert command[command.index("--refine_step") + 1] == "3"
    assert command[command.index("--verbosity") + 1] == "0"
    assert expected_output_path(ligand, tmp_path) == tmp_path / "ligand_out.pdbqt"

def test_scores_only_command_and_parser(tmp_path: Path) -> None:
    ligand = _ligand(tmp_path)
    score_path = tmp_path / "scores.tsv"
    config = DockingConfig(
        receptor=tmp_path / "receptor.pdbqt",
        center=(1.0, 2.0, 3.0),
        size=(20.0, 21.0, 22.0),
        scores_only_output=True,
    )
    command = build_unidock_command(
        [ligand], config=config, score_path=score_path,
        index_path=tmp_path / "index.txt", output_dir=tmp_path / "out",
    )
    assert "--scores_only_output" in command
    assert command[command.index("--score_file") + 1] == str(score_path)

    score_path.write_text(
        f"{ligand.path.resolve()}\t-7.125\n/another/ligand.pdbqt\tnan\n",
        encoding="utf-8",
    )
    scores = parse_score_table(score_path)
    assert scores[str(ligand.path.resolve())] == -7.125
    assert scores["/another/ligand.pdbqt"] != scores["/another/ligand.pdbqt"]

    with pytest.raises(ValueError, match="score_path"):
        build_unidock_command(
            [ligand], config=config, index_path=tmp_path / "index.txt",
            output_dir=tmp_path / "out",
        )
