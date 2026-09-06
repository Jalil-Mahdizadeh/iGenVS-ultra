"""Thin script wrapper around the iGen3 randomized SMILES CLI."""

from __future__ import annotations

import sys

from igen3.cli import main


if __name__ == "__main__":
    main(["randomize-smiles", *sys.argv[1:]])
