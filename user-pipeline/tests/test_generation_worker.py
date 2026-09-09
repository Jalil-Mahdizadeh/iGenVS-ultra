from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

try:
    import torch
    from igenvs_ultra import generation_worker
except ModuleNotFoundError:
    torch = None
    generation_worker = None


class _Canonicalizer:
    workers = 1

    def canonicalize_tokens(self, _generator, tokens):
        return [f"S{int(row[0])}_{int(row[1])}" for row in tokens]

    def close(self):
        pass


def _engine(state_file):
    assert generation_worker is not None and torch is not None
    engine = generation_worker.PersistentGenerator.__new__(
        generation_worker.PersistentGenerator
    )
    engine.args = argparse.Namespace(
        mode="de-novo",
        max_candidates=None,
        max_candidate_multiplier=10.0,
        temperature=1.0,
        greedy=False,
        top_k=None,
    )
    engine.auto_batch = False
    engine.batch_size = 2
    engine.generator = SimpleNamespace(device=torch.device("cpu"))
    engine.canonicalizer = _Canonicalizer()
    engine.seen = set()
    engine.surplus = deque()
    engine.total_candidates = 0
    engine.total_emitted = 0
    engine.state = None
    engine._state_identity = lambda: "fixture-v1"
    engine._open_state(state_file)
    return engine


@unittest.skipIf(
    torch is None or generation_worker is None,
    "run inside the released iGenVS environment",
)
class PersistentGeneratorRecoveryTests(unittest.TestCase):
    def test_resume_restores_surplus_and_is_idempotent(self) -> None:
        assert torch is not None and generation_worker is not None
        calls = []

        def candidates(_generator, count, **_kwargs):
            seed = int(torch.initial_seed())
            calls.append((seed, count))
            return torch.tensor([[seed, index] for index in range(count)])

        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            generation_worker, "generate_de_novo_batch", side_effect=candidates
        ):
            root = Path(temporary)
            state_file = root / "lane.sqlite3"
            first_output = root / "first.smi"
            first_request = {"output": str(first_output), "count": 1, "seed": 13}

            first = _engine(state_file)
            result = first.generate(first_request)
            self.assertEqual(result["generated"], 1)
            self.assertEqual(first_output.read_text(encoding="utf-8"), "S13_0\n")
            self.assertEqual(list(first.surplus), ["S13_1"])
            self.assertEqual(first.seen, set())
            first.close()

            resumed = _engine(state_file)
            replayed = resumed.generate(first_request)
            self.assertTrue(replayed["replayed_committed_request"])
            second_output = root / "second.smi"
            resumed.generate(
                {"output": str(second_output), "count": 1, "seed": 14}
            )
            self.assertEqual(
                second_output.read_text(encoding="utf-8"), "S13_1\n"
            )
            self.assertEqual(calls, [(13, 2)])
            self.assertEqual(
                resumed.state.execute("SELECT COUNT(*) FROM seen").fetchone()[0],
                2,
            )
            resumed.close()


if __name__ == "__main__":
    unittest.main()
