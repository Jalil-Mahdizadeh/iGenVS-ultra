from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from igenvs.cli import _screen_config, build_parser
from igenvs.errors import InputError


MODELS = (
    "base-isomeric",
    "base-nonisomeric",
    "rl-isomeric",
    "rl-nonisomeric",
)


def _screen_args(model: str, batch_size: str = "auto") -> list[str]:
    return [
        "screen",
        "--generate-count",
        "1",
        "--model",
        model,
        "--generator-batch-size",
        batch_size,
        "--receptor",
        "receptor.pdbqt",
        "--center",
        "0",
        "0",
        "0",
        "--output-dir",
        "output",
    ]


@pytest.mark.parametrize("model", MODELS)
def test_all_four_models_are_selectable(model: str) -> None:
    args = build_parser().parse_args(_screen_args(model))
    assert args.model == model
    assert args.search_mode == "balance"
    assert args.prep_mode == "standard"
    assert args.pose_output == "merged"
    assert args.unidock_verbosity == 0


def test_generator_batch_size_is_parsed() -> None:
    assert build_parser().parse_args(_screen_args(MODELS[0], "512")).generator_batch_size == 512
    with pytest.raises(SystemExit):
        build_parser().parse_args(_screen_args(MODELS[0], "invalid"))


def test_speed_controls_are_selectable() -> None:
    args = build_parser().parse_args(
        _screen_args(MODELS[0])
        + ["--prep-mode", "fast", "--pose-output", "none", "--refine-step", "1"]
    )
    assert (args.prep_mode, args.pose_output, args.refine_step) == ("fast", "none", 1)


def _write_target_bundle(path: Path) -> None:
    path.mkdir()
    outputs = {
        "receptor_pdb": "receptor.pdb",
        "receptor_pdbqt": "receptor.pdbqt",
        "reference_ligand": "reference_ligand.sdf",
        "pocket": "pocket.json",
    }
    contents = {
        "receptor_pdb": "ATOM receptor\n",
        "receptor_pdbqt": "REMARK receptor\n",
        "reference_ligand": "reference ligand\n",
        "pocket": '{"schema_version": 1, "center": [1, 2, 3], "size": [20, 21, 22]}\n',
    }
    for label, filename in outputs.items():
        (path / filename).write_text(contents[label], encoding="utf-8")
    checksums = {
        label: hashlib.sha256((path / filename).read_bytes()).hexdigest()
        for label, filename in outputs.items()
    }
    (path / "manifest.json").write_text(
        json.dumps(
            {"schema_version": 1, "outputs": outputs, "sha256": checksums},
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_prepared_target_resolves_receptor_and_box(tmp_path: Path) -> None:
    target = tmp_path / "target"
    _write_target_bundle(target)
    args = build_parser().parse_args(
        [
            "screen",
            "--generate-count",
            "1",
            "--target",
            str(target),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )
    config = _screen_config(args)
    assert config.target == target.resolve()
    assert config.receptor == (target / "receptor.pdbqt").resolve()
    assert config.center == (1.0, 2.0, 3.0)
    assert config.size == (20.0, 21.0, 22.0)


def test_target_rejects_manual_box_override(tmp_path: Path) -> None:
    target = tmp_path / "target"
    _write_target_bundle(target)
    args = build_parser().parse_args(
        [
            "screen",
            "--generate-count",
            "1",
            "--target",
            str(target),
            "--size",
            "20",
            "20",
            "20",
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )
    with pytest.raises(InputError, match="cannot be combined"):
        _screen_config(args)


def test_manual_pocket_keeps_default_box_size() -> None:
    config = _screen_config(build_parser().parse_args(_screen_args(MODELS[0])))
    assert config.size == (22.5, 22.5, 22.5)


def test_prepare_target_complex_arguments_are_selectable() -> None:
    args = build_parser().parse_args(
        [
            "prepare-target",
            "--complex",
            "complex.pdb",
            "--ligand-id",
            "A:LIG:501",
            "--output-dir",
            "target",
        ]
    )
    assert args.complex == Path("complex.pdb")
    assert args.ligand_id == "A:LIG:501"


def test_autodock_gpu_expert_mode_resolves_ad4_backend() -> None:
    args = build_parser().parse_args(
        [
            "screen",
            "--generate-count",
            "1",
            "--engine",
            "autodock-gpu",
            "--receptor",
            "receptor.pdbqt",
            "--adgpu-grid",
            "receptor.maps.fld",
            "--center",
            "0",
            "0",
            "0",
            "--adgpu-workers",
            "4",
            "--output-dir",
            "output",
        ]
    )
    config = _screen_config(args)
    assert config.engine == "autodock-gpu"
    assert config.scoring == "ad4"
    assert config.autodock_gpu_fld == Path("receptor.maps.fld")
    assert config.autodock_gpu_runs is None
    assert config.autodock_gpu_autostop is True
    assert config.autodock_gpu_workers == 4


def test_engine_specific_scoring_is_rejected() -> None:
    args = build_parser().parse_args(
        _screen_args(MODELS[0])
        + ["--engine", "unidock", "--scoring", "ad4"]
    )
    with pytest.raises(InputError, match="Uni-Dock"):
        _screen_config(args)


def test_autodock_gpu_requires_grid_assets() -> None:
    args = build_parser().parse_args(
        _screen_args(MODELS[0]) + ["--engine", "autodock-gpu"]
    )
    with pytest.raises(InputError, match="AD4 maps"):
        _screen_config(args)


def test_autodock_gpu_workers_are_engine_specific() -> None:
    args = build_parser().parse_args(
        _screen_args(MODELS[0]) + ["--adgpu-workers", "2"]
    )
    with pytest.raises(InputError, match="requires --engine autodock-gpu"):
        _screen_config(args)


def test_unidock_rejects_autodock_gpu_protocol_controls() -> None:
    args = build_parser().parse_args(
        _screen_args(MODELS[0]) + ["--adgpu-runs", "5"]
    )
    with pytest.raises(InputError, match="protocol controls"):
        _screen_config(args)


def test_autodock_gpu_rejects_unidock_protocol_controls() -> None:
    args = build_parser().parse_args(
        _screen_args(MODELS[0])
        + [
            "--engine",
            "autodock-gpu",
            "--adgpu-grid",
            "receptor.maps.fld",
            "--refine-step",
            "2",
        ]
    )
    with pytest.raises(InputError, match="Uni-Dock-only"):
        _screen_config(args)
