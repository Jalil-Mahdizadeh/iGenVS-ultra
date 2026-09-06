import numpy as np
from rdkit import Chem

from gmolai_retrain.probes import _sample_indices, _scaffold_clustering_probe


def test_scaffold_clustering_probe_recovers_separated_groups():
    molecules = [Chem.MolFromSmiles("c1ccccc1")] * 6 + [
        Chem.MolFromSmiles("C1CCCCC1")
    ] * 6
    scaffolds = ["c1ccccc1"] * 6 + ["C1CCCCC1"] * 6
    embeddings = np.vstack(
        (
            np.tile(np.asarray([1.0, 0.0]), (6, 1)),
            np.tile(np.asarray([0.0, 1.0]), (6, 1)),
        )
    )
    result = _scaffold_clustering_probe(
        embeddings, molecules, scaffolds, maximum_graphs=12, seed=42
    )
    assert result["available"]
    assert result["graphs"] == 12
    assert result["scaffold_clusters"] == 2
    assert result["kmeans_repetitions"] == 5
    assert result["latent_spherical_kmeans"]["adjusted_rand_index"] == 1.0
    assert result["latent_spherical_kmeans"]["adjusted_rand_index_std"] == 0.0


def test_probe_sampling_is_deterministic_and_not_a_shard_prefix():
    first = _sample_indices(100, 20, seed=42)
    repeated = _sample_indices(100, 20, seed=42)
    assert np.array_equal(first, repeated)
    assert len(np.unique(first)) == 20
    assert not np.array_equal(first, np.arange(20))
