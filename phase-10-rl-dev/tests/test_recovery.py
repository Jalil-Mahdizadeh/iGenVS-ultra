from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import torch

from igen3.model import GPTLikeModel

from igenvs_rl import trainer
from igenvs_rl.oracle import FakeOracle
from igen3.registry import resolve_model
from igenvs_rl.state import JobConfig


def _write_samples(path, smiles: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["canonical_smiles", "docking_status"],
        )
        writer.writeheader()
        writer.writerow(
            {"canonical_smiles": smiles, "docking_status": "success"}
        )


def test_checkpoint_authority_repairs_history_seen_and_evaluations(tmp_path) -> None:
    update_one = tmp_path / "updates/update-0001"
    _write_samples(update_one / "samples.csv", "CC")
    (update_one / "completion.json").write_text(
        json.dumps({"history": {"update": 1, "loss": 1.25}}),
        encoding="utf-8",
    )
    update_two = tmp_path / "updates/update-0002"
    _write_samples(update_two / "samples.csv", "CCC")
    with (tmp_path / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["update", "loss"])
        writer.writeheader()
        writer.writerow({"update": 1, "loss": 1.25})
        writer.writerow({"update": 2, "loss": 9.0})
    (tmp_path / "seen.smi").write_text("CC\nCCC\n", encoding="utf-8")
    with (tmp_path / "evaluations.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "reward_mean"])
        writer.writeheader()
        writer.writerow({"label": "update-0001", "reward_mean": 0.1})
        writer.writerow({"label": "update-0002", "reward_mean": 0.2})

    trainer._reconcile_completed_state(tmp_path, checkpoint_update=1)

    with (tmp_path / "history.csv").open(newline="") as handle:
        assert [int(row["update"]) for row in csv.DictReader(handle)] == [1]
    assert (tmp_path / "seen.smi").read_text(encoding="utf-8") == "CC\n"
    with (tmp_path / "evaluations.csv").open(newline="") as handle:
        assert [row["label"] for row in csv.DictReader(handle)] == ["update-0001"]


def test_absolute_target_update_is_idempotent_end_to_end(tmp_path, monkeypatch) -> None:
    model_root = tmp_path / "base"
    model_root.mkdir()
    (model_root / "vocab.pkl").write_bytes(b"fixture-vocabulary")
    torch.manual_seed(71)
    base = GPTLikeModel(13, 16, 4, 2, 0.0, 8).eval().state_dict()

    def load_bundle(*, policy_state=None, **_kwargs):
        policy = GPTLikeModel(13, 16, 4, 2, 0.0, 8).eval()
        prior = GPTLikeModel(13, 16, 4, 2, 0.0, 8).eval()
        policy.load_state_dict(policy_state or base)
        prior.load_state_dict(base)
        prior.requires_grad_(False)
        return SimpleNamespace(
            policy=policy,
            prior=prior,
            vocab=SimpleNamespace(eos_idx=12),
            device=torch.device("cpu"),
            spec=SimpleNamespace(
                model_id="base-isomeric",
                weights_name="weights.pth",
                vocab_name="vocab.pkl",
                vocab_path=lambda root: root / "vocab.pkl",
            ),
        )

    calls = []

    def sample_and_score(**kwargs):
        calls.append(kwargs["output_dir"])
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True)
        with (output_dir / "samples.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["canonical_smiles", "docking_status"]
            )
            writer.writeheader()
            writer.writerow(
                {"canonical_smiles": "CC", "docking_status": "success"}
            )
        rows = [
            {"canonical_smiles": "CC", "docking_status": "success"}
            for _ in range(4)
        ]
        metrics = {
            "docked": 4.0,
            "reward_mean": 0.5,
            "score_median": -1.0,
            "elite_fraction": 0.25,
            "elite_unique_count": 1.0,
            "qualified_elite_fraction": 0.25,
            "qualified_elite_unique_count": 1.0,
            "valid_fraction": 1.0,
            "unique_fraction": 0.25,
            "top_molecule_fraction": 1.0,
        }
        return (
            torch.randint(0, 12, (4, 8)),
            torch.tensor([0.0, 1.0, 0.5, 0.25]),
            torch.ones(4, dtype=torch.bool),
            rows,
            metrics,
        )

    monkeypatch.setattr(trainer, "load_policy_bundle", load_bundle)
    monkeypatch.setattr(trainer, "_oracle", lambda _config: object())
    monkeypatch.setattr(trainer, "_sample_and_score", sample_and_score)
    monkeypatch.setattr(trainer, "synchronize_sampler", lambda _bundle: None)
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "scores.json").write_text(
        json.dumps({"scores": [-2.0, -1.0, 0.0]}), encoding="utf-8"
    )
    config = JobConfig(
        schema_version=1,
        target=str(tmp_path / "target"),
        image="fixture.sif",
        igenvs_project=str(tmp_path),
        model_root=str(model_root),
        oracle="fake",
        batch_size=4,
        evaluation_every=0,
    )

    trainer.train_job(tmp_path, config, target_update=1)
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["completed_updates"] == 1
    assert progress["model_latest_exported"] is True
    assert len(calls) == 1

    best_export = tmp_path / "model/base_isomeric/weights.pth"
    assert best_export.is_file()
    best_export.unlink()
    trainer.train_job(tmp_path, config, target_update=1)
    assert len(calls) == 1
    assert best_export.is_file()
    with (tmp_path / "history.csv").open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 1


def test_reference_retry_archives_partial_docking_and_publishes_atomically(tmp_path, monkeypatch):
    partial = tmp_path / "reference/docking"
    partial.mkdir(parents=True)
    (partial / "interrupted.log").write_text("retained diagnostic")
    (tmp_path / "reference/scores.json").write_text('{"scores":')
    def generate(*args, output_path, **kwargs):
        output_path.write_text("CCO\n")
        return SimpleNamespace(generated=1, candidates_generated=1)
    monkeypatch.setattr(trainer, "write_de_novo_file", generate)
    config = SimpleNamespace(reference_count=1, batch_size=1, temperature=1.0, top_k=64, model_id="base-isomeric")
    scores = trainer._ensure_reference(tmp_path, config, SimpleNamespace(sampler=object()), FakeOracle())
    assert len(scores) == 1
    assert len(list(tmp_path.glob("reference.incomplete-*/docking/interrupted.log"))) == 1
    assert trainer._read_reference(tmp_path) == scores
    assert not (tmp_path / "reference/scores.json.tmp").exists()
    assert trainer._ensure_reference(tmp_path, config, None, None) == scores


def test_recover_job_repairs_exports_without_training_or_rng_use(tmp_path, monkeypatch):
    spec = resolve_model("base-isomeric")
    root = tmp_path / "base"
    spec.vocab_path(root).parent.mkdir(parents=True)
    spec.vocab_path(root).write_bytes(b"vocabulary")
    config = SimpleNamespace(model_id=spec.model_id, model_root=str(root))
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    torch.save({"update": 1, "policy_state": {"weight": torch.tensor([7.0])}}, checkpoints / "latest.pt")
    torch.save({"update": 0, "policy_state": {"weight": torch.tensor([1.0])}}, checkpoints / "best.pt")
    update = tmp_path / "updates/update-0001"
    _write_samples(update / "samples.csv", "CC")
    (update / "completion.json").write_text(json.dumps({"history": {"update": 1, "loss": 0.5}, "promote_best": True}))
    (tmp_path / "history.csv").write_text("update,loss\n1,0.5\n2,0.1\n")
    def forbidden(*args, **kwargs):
        raise AssertionError("recovery must not load a model or sample")
    monkeypatch.setattr(trainer, "load_policy_bundle", forbidden)
    monkeypatch.setattr(trainer, "_sample_and_score", forbidden)
    rng = torch.get_rng_state()
    assert trainer.recover_job(tmp_path, config) == 1
    assert torch.equal(torch.get_rng_state(), rng)
    for directory in ("model", "model-latest"):
        exported = torch.load(spec.weights_path(tmp_path / directory), weights_only=True)
        assert exported["weight"].item() == 7.0
    assert json.loads((tmp_path / "progress.json").read_text())["model_latest_exported"] is True
    with (tmp_path / "history.csv").open(newline="") as handle:
        assert [int(row["update"]) for row in csv.DictReader(handle)] == [1]


def test_reconcile_tolerates_a_torn_legacy_csv_tail(tmp_path):
    update = tmp_path / "updates/update-0001"
    _write_samples(update / "samples.csv", "CC")
    (update / "completion.json").write_text(json.dumps({"history": {"update": 1, "loss": 0.5}}))
    (tmp_path / "history.csv").write_text("update,loss\n1,0.5\npartial")
    (tmp_path / "evaluations.csv").write_text("label,reward_mean\nupdate-0001,0.8\nupdate-0002")
    trainer._reconcile_completed_state(tmp_path, 1)
    with (tmp_path / "history.csv").open(newline="") as handle:
        assert list(csv.DictReader(handle)) == [{"update": "1", "loss": "0.5"}]
    with (tmp_path / "evaluations.csv").open(newline="") as handle:
        assert list(csv.DictReader(handle)) == [{"label": "update-0001", "reward_mean": "0.8"}]
