from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tomllib

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from inference._decoder import decode_tokens, proportional_merge  # noqa: E402
from inference.generate_embeddings import PendingMolecule  # noqa: E402
from inference.gmolai import (  # noqa: E402
    EMBEDDING_DIMENSIONS,
    GLOBAL_SEED,
    SAMPLE_POOL_NAME,
    SAMPLING_PHASE,
    Proposal,
    _DiskBackedEmbeddingStore,
    _close_memmap_arrays,
    build_parser,
    filter_candidates,
    safe_seed_filename,
    sample_seed,
)


def test_byte_smiles_decode_is_lossless() -> None:
    smiles = "N[C@@H](C)C(=O)O"
    tokens = [ord(value) + 3 for value in smiles]
    decoded, error = decode_tokens([*tokens, 2, 0, 0])
    assert decoded == smiles
    assert error == ""


def test_frozen_balanced_merge_order() -> None:
    result = proportional_merge(
        [("beam", 0), ("beam", 1)],
        [("sample", 0), ("sample", 1)],
    )
    assert result == [
        ("beam", 0),
        ("sample", 0),
        ("beam", 1),
        ("sample", 1),
    ]


def test_sampling_seed_matches_step2d_definition() -> None:
    molecule_hash = hashlib.sha256(b"CCO").hexdigest()
    digest = hashlib.sha256(
        "\x1f".join(
            (
                str(GLOBAL_SEED),
                SAMPLING_PHASE,
                SAMPLE_POOL_NAME,
                molecule_hash,
            )
        ).encode("utf-8")
    ).hexdigest()
    expected = int(digest[:16], 16) % (2**63 - 1)
    assert sample_seed(molecule_hash) == expected


def test_candidate_filter_is_valid_unique_and_excludes_seed() -> None:
    config = json.loads(
        (REPOSITORY_ROOT / "inference" / "models" / "resolved_config.json")
        .read_text(encoding="utf-8")
    )
    seed_smiles = "CCO"
    seed_hash = hashlib.sha256(seed_smiles.encode("utf-8")).hexdigest()
    raw = (
        ("CCN", ""),
        ("CCN", ""),
        (seed_smiles, ""),
        ("C1", ""),
        ("c1ccccc1", ""),
        ("", "missing_eos"),
    )
    proposals = [
        Proposal(
            proposal_rank=index,
            source_kind="sample",
            source_rank=index,
            raw_smiles=smiles,
            token_error=error,
            decoder_log_probability=-float(index),
            generated_length=len(smiles) + 1,
        )
        for index, (smiles, error) in enumerate(raw, start=1)
    ]
    rows, stats = filter_candidates(
        proposals,
        seed_input_row=1,
        seed_id="ethanol",
        seed_canonical_smiles=seed_smiles,
        seed_hash=seed_hash,
        resolved_config=config,
        include_seed=False,
    )
    assert [row["canonical_smiles"] for row in rows] == ["CCN", "c1ccccc1"]
    assert len({row["canonical_smiles"] for row in rows}) == len(rows)
    assert all(0.0 <= row["morgan_tanimoto"] <= 1.0 for row in rows)
    assert stats["raw_proposals"] == 6
    assert stats["token_decodable_proposals"] == 5
    assert stats["policy_accepted_proposals"] == 4
    assert stats["duplicate_or_redundant_proposals"] == 1
    assert stats["excluded_seed_identity"] == 1
    assert stats["retained_unique_candidates"] == 2


def test_seed_filenames_are_safe_and_collision_resistant() -> None:
    name = safe_seed_filename(7, "../../A compound", "a" * 64)
    assert name == "seed-000007-A-compound-aaaaaaaa.csv"
    assert "/" not in name


def test_decode_defaults_to_frozen_thousand_proposal_prefix() -> None:
    args = build_parser().parse_args(["decode"])
    assert args.proposal_budget == 1000
    assert args.include_seed is False


def test_package_declares_graph_runtime_dependencies() -> None:
    project = tomllib.loads(
        (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    names = set()
    for requirement in project["dependencies"]:
        name = requirement
        for separator in ("<", ">", "=", "!", "~", "[", ";", " "):
            name = name.split(separator, maxsplit=1)[0]
        names.add(name.lower().replace("_", "-"))
    assert {"torch", "torch-geometric"} <= names


def test_disk_backed_embedding_store_preserves_batches_and_rows(
    tmp_path: Path,
) -> None:
    smiles = ("CCO", "CCN", "CCO")
    records = [
        PendingMolecule(
            input_row=index,
            input_id=("alpha", "alpha", "beta")[index - 1],
            input_smiles=("CCO", "CCN", "OCC")[index - 1],
            canonical_smiles=value,
            molecule_hash=hashlib.sha256(value.encode("utf-8")).hexdigest(),
            atom_count=3,
        )
        for index, value in enumerate(smiles, start=1)
    ]
    expected_vectors = np.arange(
        len(records) * EMBEDDING_DIMENSIONS,
        dtype=np.float32,
    ).reshape(len(records), EMBEDDING_DIMENSIONS)
    store = _DiskBackedEmbeddingStore(tmp_path, EMBEDDING_DIMENSIONS)
    arrays: dict[str, np.ndarray] = {}
    try:
        assert store.observe_input_id("alpha") is False
        assert store.observe_input_id("alpha") is True
        assert store.observe_input_id("beta") is False
        store.append(records[:2], expected_vectors[:2])
        store.append(records[2:], expected_vectors[2:])
        arrays, unique_molecules = store.materialize_arrays()

        assert unique_molecules == 2
        assert store.accepted_count == 3
        assert isinstance(arrays["embeddings"], np.memmap)
        np.testing.assert_array_equal(arrays["embeddings"], expected_vectors)
        np.testing.assert_array_equal(arrays["input_row"], [1, 2, 3])
        np.testing.assert_array_equal(
            arrays["input_id"],
            ["alpha", "alpha", "beta"],
        )
        np.testing.assert_array_equal(
            arrays["canonical_smiles"],
            smiles,
        )

        archive = tmp_path / "roundtrip.npz"
        with archive.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        with np.load(archive, allow_pickle=False) as payload:
            np.testing.assert_array_equal(
                payload["embeddings"],
                expected_vectors,
            )
            np.testing.assert_array_equal(
                payload["input_smiles"],
                ["CCO", "CCN", "OCC"],
            )
    finally:
        _close_memmap_arrays(arrays)
        store.close()
