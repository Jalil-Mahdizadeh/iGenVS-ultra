"""SMILES tokenization helpers for the iGen3 vocabulary."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Vocabulary:
    tokens: list[str]
    pad_token: str = "Z"
    sos_token: str = "G"
    eos_token: str = "E"

    def __post_init__(self) -> None:
        self.char_to_idx = {token: i for i, token in enumerate(self.tokens)}
        self.idx_to_piece = [""] * len(self.tokens)
        for token, idx in self.char_to_idx.items():
            if token in {self.pad_token, self.sos_token, self.eos_token}:
                self.idx_to_piece[idx] = ""
            elif token == "X":
                self.idx_to_piece[idx] = "Cl"
            elif token == "Y":
                self.idx_to_piece[idx] = "Br"
            else:
                self.idx_to_piece[idx] = token

        self.pad_idx = self.char_to_idx[self.pad_token]
        self.sos_idx = self.char_to_idx[self.sos_token]
        self.eos_idx = self.char_to_idx[self.eos_token]

    @property
    def size(self) -> int:
        return len(self.tokens)

    @classmethod
    def from_file(cls, path: Path) -> "Vocabulary":
        with path.open("rb") as fh:
            tokens = pickle.load(fh)
        return cls(tokens=list(tokens))

    def encode_smiles(self, smiles: str, *, add_sos: bool = True, strict: bool = True) -> list[int]:
        """Encode a SMILES string, mapping Cl/Br to the training-time X/Y tokens."""
        encoded_tokens = []
        smi = smiles.strip()
        i = 0
        while i < len(smi):
            if smi.startswith("Cl", i):
                encoded_tokens.append("X")
                i += 2
            elif smi.startswith("Br", i):
                encoded_tokens.append("Y")
                i += 2
            else:
                encoded_tokens.append(smi[i])
                i += 1

        if add_sos and (not encoded_tokens or encoded_tokens[0] != self.sos_token):
            encoded_tokens.insert(0, self.sos_token)

        ids: list[int] = []
        unknown: list[str] = []
        for token in encoded_tokens:
            idx = self.char_to_idx.get(token)
            if idx is None:
                unknown.append(token)
                ids.append(self.pad_idx)
            else:
                ids.append(idx)

        if strict and unknown:
            unique = ", ".join(sorted(set(unknown)))
            raise ValueError(f"SMILES contains token(s) not present in this model vocabulary: {unique}")
        return ids

    def decode_rows(self, rows, *, include_eos: bool = False) -> list[str]:
        """Decode a 2D CPU/numpy-like token array into SMILES strings."""
        decoded: list[str] = []
        for row in rows:
            pieces: list[str] = []
            for token in row:
                idx = int(token)
                if idx == self.eos_idx:
                    if include_eos:
                        pieces.append(self.idx_to_piece[idx])
                    break
                pieces.append(self.idx_to_piece[idx])
            decoded.append("".join(pieces))
        return decoded
