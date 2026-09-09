#!/usr/bin/env python3
"""Enter the frozen RL implementation from an iGenVS runtime container."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    root = Path(__file__).resolve().parents[3]
    sources = (
        root / "phase-10-rl-dev/src",
        root / "iGenVS/iGen3/src",
        root / "iGenVS/src",
    )
    for source in sources:
        sys.path.insert(0, str(source))
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [*(str(source) for source in sources), *filter(None, [os.environ.get("PYTHONPATH")])]
    )

    from igenvs_rl.cli import main as rl_main

    return int(rl_main(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
