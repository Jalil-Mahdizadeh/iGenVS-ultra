from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

try:
    from igenvs_ultra.fast_policy import canonicalize_policy_chunk
    from rdkit import Chem
except ModuleNotFoundError:
    canonicalize_policy_chunk = None
    Chem = None


PROJECT = Path(__file__).resolve().parents[2]
GMOLAI_SOURCE = PROJECT / "gMolAI-v2.0/src"
if str(GMOLAI_SOURCE) not in sys.path:
    sys.path.insert(0, str(GMOLAI_SOURCE))

try:
    from gmolai_retrain.chem import Rejection, canonicalize
except ModuleNotFoundError:
    Rejection = None
    canonicalize = None


@unittest.skipIf(
    canonicalize_policy_chunk is None or canonicalize is None,
    "RDKit and the released gMolAI source are required",
)
class FastPolicyEquivalenceTests(unittest.TestCase):
    def test_inference_fields_match_released_canonicalizer(self) -> None:
        resolved = json.loads(
            (PROJECT / "gMolAI-v2.0/inference/models/resolved_config.json").read_text()
        )
        data = resolved["data"]
        policy = data["canonicalization"]
        smiles = [
            "",
            "not-a-smiles",
            "CC.O",
            "[Fe]C",
            "C",
            "CCO",
            "C[C@H](O)Cl",
            "c1ccccc1",
            "C" * (int(policy["max_atoms"]) + 1),
        ]
        observed = canonicalize_policy_chunk(
            (7, [(f"mol-{index}", value) for index, value in enumerate(smiles)], policy)
        )
        self.assertEqual([row[0] for row in observed], list(range(7, 7 + len(smiles))))

        for raw, row in zip(smiles, observed):
            reference = canonicalize(
                raw,
                isomeric_smiles=bool(policy["isomeric_smiles"]),
                fragment_policy=str(policy["fragment_policy"]),
                allowed_elements=set(policy["allowed_elements"]),
                min_atoms=int(policy["min_atoms"]),
                max_atoms=int(policy["max_atoms"]),
                buckets=int(data["hash_buckets"]),
                split_cfg=data["split"],
            )
            if isinstance(reference, Rejection):
                self.assertEqual(row[5], reference.reason)
                # Atom count is discarded for rejected rows by both inference
                # paths; canonical SMILES and hashes must remain empty.
                self.assertEqual(row[2:4], ("", ""))
            else:
                self.assertIsNone(row[5])
                self.assertEqual(row[2], reference.smiles)
                self.assertEqual(row[3], reference.molecule_hash)
                self.assertEqual(row[4], reference.atom_count)
                self.assertIsNotNone(Chem.MolFromSmiles(row[2]))


if __name__ == "__main__":
    unittest.main()
