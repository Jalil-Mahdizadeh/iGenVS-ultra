from __future__ import annotations

import pytest

from igenvs.errors import InputError
from igenvs.tuning import _choose_batch


def test_tuner_selects_reliable_peak_throughput() -> None:
    rows = [
        {"batch_size": 128, "success_fraction": 1.0, "successful": 128, "successful_ligands_per_second": 90.0},
        {"batch_size": 256, "success_fraction": 1.0, "successful": 256, "successful_ligands_per_second": 100.0},
        {"batch_size": 512, "success_fraction": 1.0, "successful": 512, "successful_ligands_per_second": 101.0},
    ]
    assert _choose_batch(rows) == 512


def test_tuner_rejects_unreliable_measurements() -> None:
    rows = [
        {"batch_size": 128, "success_fraction": 0.98, "successful": 126, "successful_ligands_per_second": 100.0}
    ]
    with pytest.raises(InputError):
        _choose_batch(rows)
