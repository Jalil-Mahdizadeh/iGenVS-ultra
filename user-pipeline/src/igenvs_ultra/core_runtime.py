#!/usr/bin/env python3
"""Run the mounted iGenVS/iGen3 source with the container's dependencies."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] not in {"igenvs", "igen3"}:
        raise ValueError("expected the igenvs or igen3 entry point")
    root = Path(__file__).resolve().parents[3]
    sources = [root / "iGenVS/src", root / "iGenVS/iGen3/src"]
    for source in sources:
        if not source.is_dir():
            raise RuntimeError(f"runtime source is missing: {source}")
    sys.path[:0] = [str(source) for source in sources]
    # sys.path alone does not reach subprocesses such as the RL docking oracle.
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [*(str(source) for source in sources), *filter(None, [os.environ.get("PYTHONPATH")])]
    )
    entry = arguments.pop(0)
    if entry == "igenvs":
        from igenvs.cli import main as cli_main
    else:
        from igen3.cli import main as cli_main
    return int(cli_main(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
