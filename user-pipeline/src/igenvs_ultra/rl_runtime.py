#!/usr/bin/env python3
"""Enter the frozen RL implementation from an iGenVS runtime container."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] not in {"cli", "validate"}:
        raise SystemExit("usage: rl_runtime.py {cli|validate} [arguments ...]")

    action = arguments.pop(0)
    root = Path(__file__).resolve().parents[3]
    for source in (
        root / "phase-10-rl-dev/src",
        root / "iGenVS/iGen3/src",
        root / "iGenVS/src",
    ):
        sys.path.insert(0, str(source))

    if action == "cli":
        from igenvs_rl.cli import main as rl_main

        return int(rl_main(arguments))

    validator = root / "phase-10-rl-dev/scripts/validate_target.py"
    sys.argv = [str(validator), *arguments]
    runpy.run_path(str(validator), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
