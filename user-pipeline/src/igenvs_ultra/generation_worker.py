#!/usr/bin/env python3
"""Persistent iGen3 worker used by the ultra-screening orchestrator.

The public iGen3 CLI deliberately remains a one-shot command.  Large ultra
screens, however, need to retain the loaded generator and any unused valid
rows from the final CUDA block.  This small JSON-lines service supplies that
lifetime without modifying the released iGen3 package in its container.
"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sqlite3
import sys
import time
import traceback
from typing import Any, Iterable

import torch
from rdkit import rdBase

from igen3.generation import (
    canonicalize_valid_smiles,
    decode_token_batch,
    estimate_batch_upper_bound,
    generate_de_novo_batch,
    write_derivative_file,
)
from igen3.metrics import save_metrics
from igen3.model import load_generator, maybe_compile_generator
from igen3.registry import resolve_model


PROTOCOL_VERSION = 1
CALIBRATION_VERSION = 4
CUDA_SDPA_BATCH_LIMIT = 65_280


def _emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, sort_keys=True, allow_nan=False), flush=True)


def _read_seed_file(path: Path | None) -> list[str]:
    if path is None:
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _round_batch(value: int) -> int:
    if value <= 256:
        return max(1, value)
    return max(256, (value // 256) * 256)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonicalize_chunk(payload: tuple[list[str], bool]) -> list[str | None]:
    values, isomeric_smiles = payload
    return [
        canonicalize_valid_smiles(value, isomeric_smiles=isomeric_smiles)
        for value in values
    ]


def _physical_cores_in_affinity(allowed: set[int]) -> int:
    """Best-effort physical-core count inside the process CPU allowance."""
    try:
        blocks = Path("/proc/cpuinfo").read_text(encoding="utf-8").split("\n\n")
        cores: set[tuple[str, str]] = set()
        matched = 0
        for block in blocks:
            fields = {}
            for line in block.splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    fields[key.strip()] = value.strip()
            if "processor" not in fields or int(fields["processor"]) not in allowed:
                continue
            matched += 1
            cores.add(
                (
                    fields.get("physical id", "0"),
                    fields.get("core id", fields["processor"]),
                )
            )
        if matched:
            return max(1, len(cores))
    except (OSError, ValueError):
        pass
    return max(1, len(allowed))


class ParallelCanonicalizer:
    """Persistent, order-preserving CPU canonicalization for generator rows."""

    def __init__(self, *, expected_count: int) -> None:
        allowed = (
            set(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else set(range(int(os.cpu_count() or 1)))
        )
        affinity_count = len(allowed)
        physical_cores = _physical_cores_in_affinity(allowed)
        # Process startup cannot amortize for a smoke-size screen.  Sustained
        # screens use the CPU affinity assigned to this GPU lane, leaving one
        # physical core for decoding/writing and capping process fan-out at 64.
        # The orchestrator gives lanes disjoint affinities, so this scales down on
        # SMT workstations and remains non-oversubscribed on multi-GPU hosts.
        self.workers = (
            min(64, max(1, physical_cores - 1))
            if expected_count >= 65_536
            else 1
        )
        self.affinity_cpus = affinity_count
        self.physical_cores = physical_cores
        self.executor: ProcessPoolExecutor | None = None
        self._warm_futures: list[Any] = []
        if self.workers > 1:
            self.executor = ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=mp.get_context("spawn"),
            )
            # Process creation overlaps model/calibration GPU work.  Waiting
            # only when rows first need canonicalization removes pool startup
            # from the production critical path without forking a CUDA owner.
            self._warm_futures = [
                self.executor.submit(_canonicalize_chunk, (["C"], True))
                for _ in range(self.workers)
            ]

    def canonicalize_tokens(
        self, generator: Any, tokens: torch.Tensor
    ) -> list[str | None]:
        raw = decode_token_batch(generator, tokens)
        if self.executor is None or len(raw) < self.workers * 64:
            return _canonicalize_chunk((raw, generator.spec.is_isomeric))
        for future in self._warm_futures:
            future.result()
        self._warm_futures.clear()
        chunk_size = math.ceil(len(raw) / self.workers)
        payloads = [
            (raw[start : start + chunk_size], generator.spec.is_isomeric)
            for start in range(0, len(raw), chunk_size)
        ]
        output: list[str | None] = []
        for chunk in self.executor.map(_canonicalize_chunk, payloads):
            output.extend(chunk)
        return output

    def close(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None
        self._warm_futures.clear()


def _fingerprint(
    generator: Any,
    args: argparse.Namespace,
    canonicalizer: ParallelCanonicalizer,
) -> tuple[str, dict[str, Any]]:
    device = generator.device
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        hardware = {
            "device_type": "cuda",
            "name": properties.name,
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_bytes": int(properties.total_memory),
            "multi_processor_count": int(properties.multi_processor_count),
        }
    else:
        hardware = {
            "device_type": device.type,
            "machine": os.uname().machine,
            "cpu_count": (
                len(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else os.cpu_count()
            ),
        }
    value = {
        "calibration_version": CALIBRATION_VERSION,
        "hardware": hardware,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": sys.version.split()[0],
        "rdkit": rdBase.rdkitVersion,
        "model": generator.spec.model_id,
        "model_artifacts": {
            "weights_sha256": _sha256(generator.spec.weights_path(args.model_dir)),
            "vocabulary_sha256": _sha256(generator.spec.vocab_path(args.model_dir)),
        },
        "sequence_length": int(generator.spec.seq_len),
        "dtype": str(generator.dtype),
        "mode": args.mode,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "greedy": bool(args.greedy),
        "compiled": bool(args.compile),
        "compile_mode": args.compile_mode,
        "max_batch_size": int(args.max_batch_size),
        "canonical_workers": canonicalizer.workers,
        "affinity_cpus": canonicalizer.affinity_cpus,
        "physical_cores": canonicalizer.physical_cores,
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), value


def _read_cached_batch(path: Path | None, key: str) -> dict[str, Any] | None:
    if path is None:
        return None
    path = path.with_name(f"{path.stem}-{key}.json")
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    profile = payload.get("profile")
    if payload.get("key") != key or not isinstance(profile, dict):
        return None
    selected = profile.get("selected_batch_size")
    if not isinstance(selected, int) or selected <= 0:
        return None
    return profile


def _write_cached_batch(path: Path | None, key: str, profile: dict[str, Any]) -> None:
    if path is None:
        return
    path = path.with_name(f"{path.stem}-{key}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "key": key,
        "profile": profile,
    }
    temporary = path.with_name(f"{path.name}.{os.getpid()}.partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _recoverable_cuda_batch_error(exc: RuntimeError) -> bool:
    """Return whether a smaller CUDA batch is a valid recovery strategy.

    Very large SDPA launches can exceed a kernel launch dimension before they
    exhaust memory.  CUDA reports that as ``invalid configuration argument``
    rather than OOM.  Treat both as a capacity limit while tuning, but do not
    hide unrelated model or data errors.
    """
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "out of memory",
            "invalid configuration argument",
            "too many resources requested",
            "launch out of resources",
            "failed to allocate",
            "cublas_status_alloc_failed",
        )
    )


def _unique_valid_count(
    generator: Any,
    tokens: torch.Tensor,
    canonicalizer: ParallelCanonicalizer,
) -> int:
    return len(
        {
            canonical
            for canonical in canonicalizer.canonicalize_tokens(generator, tokens)
            if canonical is not None
        }
    )


def _calibrate_batch_size(
    generator: Any,
    args: argparse.Namespace,
    canonicalizer: ParallelCanonicalizer,
    *,
    expected_count: int,
) -> tuple[int, dict[str, Any]]:
    """Measure accepted output throughput over a bounded portable candidate set."""
    key, identity = _fingerprint(generator, args, canonicalizer)
    cache_path = args.profile_cache.resolve() if args.profile_cache else None
    memory_limit = estimate_batch_upper_bound(
        generator, max_batch_size=args.max_batch_size
    )
    implementation_limit = (
        CUDA_SDPA_BATCH_LIMIT
        if generator.device.type == "cuda"
        else args.max_batch_size
    )
    cached = _read_cached_batch(cache_path, key)
    if (
        cached is not None
        and int(cached["selected_batch_size"]) <= args.max_batch_size
        and int(cached["selected_batch_size"]) <= memory_limit
        and int(cached["selected_batch_size"]) <= implementation_limit
    ):
        return int(cached["selected_batch_size"]), {
            **cached,
            "source": "cache",
            "key": key,
            "current_memory_fitting_batch_size": memory_limit,
        }
    # CUDA SDPA currently maps the sequence batch onto a launch dimension with
    # a 65,535-block limit.  Stay at the largest 256-aligned value below that
    # implementation limit; automatic runtime halving remains the fallback for
    # a backend with a tighter constraint.
    # Calibration must improve invocation-to-result time, not only steady
    # throughput.  Four candidates, each warmed and measured, consume five
    # times ``upper`` proposal slots.  Require at least 512 production batches
    # so those trials represent under roughly 1% of the requested workload.
    workload_limit = max(256, int(expected_count))
    upper = _round_batch(min(memory_limit, workload_limit, implementation_limit))
    calibration_minimum_count = 512 * upper
    if expected_count < calibration_minimum_count:
        profile = {
            "source": "amortization_heuristic",
            "key": key,
            "identity": identity,
            "memory_fitting_batch_size": memory_limit,
            "implementation_batch_limit": implementation_limit,
            "calibration_minimum_count": calibration_minimum_count,
            "expected_count": expected_count,
            "selected_batch_size": upper,
            "measurements": [],
        }
        return upper, profile
    else:
        candidates = sorted(
            {
                _round_batch(min(upper, max(512, upper // 4))),
                _round_batch(min(upper, max(512, upper // 2))),
                _round_batch(min(upper, max(512, 3 * upper // 4))),
                upper,
            }
        )

    measurements = []
    for candidate in candidates:
        # CUDA libraries and allocators can have a substantial first-shape
        # cost.  Warm the exact candidate shape before timing it.
        warm = None
        tokens = None
        print(
            f"[iGenVS-ultra] calibrating iGen3 batch {candidate:,}",
            file=sys.stderr,
            flush=True,
        )
        try:
            torch.manual_seed(7_311_927 + candidate)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(7_311_927 + candidate)
            warm = generate_de_novo_batch(
                generator,
                candidate,
                temperature=args.temperature,
                do_sample=not args.greedy,
                top_k=args.top_k,
            )
            # Warm the complete production path, including persistent CPU
            # chemistry workers, before timing accepted output throughput.
            canonicalizer.canonicalize_tokens(generator, warm)
            del warm
            warm = None
            if generator.device.type == "cuda":
                torch.cuda.synchronize(generator.device)
            torch.manual_seed(91_771 + candidate)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(91_771 + candidate)
                torch.cuda.synchronize(generator.device)
            started = time.perf_counter()
            tokens = generate_de_novo_batch(
                generator,
                candidate,
                temperature=args.temperature,
                do_sample=not args.greedy,
                top_k=args.top_k,
            )
            if generator.device.type == "cuda":
                torch.cuda.synchronize(generator.device)
            accepted = _unique_valid_count(generator, tokens, canonicalizer)
            elapsed = time.perf_counter() - started
            measurements.append(
                {
                    "batch_size": candidate,
                    "accepted": accepted,
                    "seconds": elapsed,
                    "accepted_per_second": accepted / max(elapsed, 1.0e-9),
                    "candidates_per_second": candidate / max(elapsed, 1.0e-9),
                    "status": "complete",
                }
            )
        except RuntimeError as exc:
            if not _recoverable_cuda_batch_error(exc):
                raise
            measurements.append(
                {
                    "batch_size": candidate,
                    "status": "unsupported_capacity",
                    "error": str(exc).splitlines()[0],
                }
            )
            print(
                f"[iGenVS-ultra] iGen3 batch {candidate:,} is unsupported; "
                "continuing with measured smaller batches",
                file=sys.stderr,
                flush=True,
            )
        finally:
            del warm, tokens
            if generator.device.type == "cuda":
                torch.cuda.empty_cache()
    successful = [item for item in measurements if item["status"] == "complete"]
    if not successful:
        raise RuntimeError(
            "no automatically selected iGen3 batch completed on this device; "
            "lower --generator-max-batch-size"
        )
    selected = max(
        successful,
        key=lambda item: (float(item["accepted_per_second"]), int(item["batch_size"])),
    )
    profile = {
        "source": "calibration",
        "key": key,
        "identity": identity,
        "memory_fitting_batch_size": memory_limit,
        "implementation_batch_limit": implementation_limit,
        "calibration_minimum_count": calibration_minimum_count,
        "expected_count": expected_count,
        "selected_batch_size": int(selected["batch_size"]),
        "measurements": measurements,
        "created_at_unix": time.time(),
    }
    _write_cached_batch(cache_path, key, profile)
    return int(selected["batch_size"]), profile


class PersistentGenerator:
    """A loaded iGen3 model with cross-request uniqueness and surplus reuse."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.auto_batch = args.batch_size == "auto"
        self.spec = resolve_model(args.model)
        self.generator = load_generator(
            self.spec,
            model_root=args.model_dir,
            device_name=args.device,
            dtype_name=args.dtype,
            compile_model=False,
        )
        self.generator = maybe_compile_generator(
            self.generator, enabled=args.compile, compile_mode=args.compile_mode
        )
        self.canonicalizer = ParallelCanonicalizer(
            expected_count=args.expected_count if args.mode == "de-novo" else 0
        )
        if args.batch_size == "auto":
            self.batch_size, self.calibration = _calibrate_batch_size(
                self.generator,
                args,
                self.canonicalizer,
                expected_count=args.expected_count,
            )
        else:
            self.batch_size = int(args.batch_size)
            self.calibration = {
                "source": "explicit",
                "selected_batch_size": self.batch_size,
            }
        self.seen: set[str] = set()
        self.surplus: deque[str] = deque()
        self.total_candidates = 0
        self.total_emitted = 0
        self.seeds = [*args.seed_smiles, *_read_seed_file(args.seed_file)]
        if args.mode == "derivative" and not self.seeds:
            raise ValueError("derivative generation requires seed SMILES")
        self.state: sqlite3.Connection | None = None
        if args.mode == "de-novo" and args.state_file is not None:
            self._open_state(args.state_file.resolve())

    def _state_identity(self) -> str:
        value = {
            "schema_version": 1,
            "model": self.spec.model_id,
            "weights_sha256": _sha256(self.spec.weights_path(self.args.model_dir)),
            "vocabulary_sha256": _sha256(self.spec.vocab_path(self.args.model_dir)),
            "mode": self.args.mode,
            "temperature": self.args.temperature,
            "top_k": self.args.top_k,
            "greedy": bool(self.args.greedy),
            "dtype": str(self.generator.dtype),
        }
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    def _open_state(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.state = sqlite3.connect(str(path), timeout=60)
        self.state.execute("PRAGMA journal_mode=WAL")
        self.state.execute("PRAGMA synchronous=FULL")
        self.state.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS seen (
                value TEXT PRIMARY KEY
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS surplus (
                position INTEGER PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS requests (
                request_key TEXT PRIMARY KEY,
                output_path TEXT NOT NULL,
                output_sha256 TEXT NOT NULL,
                result_json TEXT NOT NULL
            ) WITHOUT ROWID;
            """
        )
        self.state.execute(
            "CREATE TEMP TABLE IF NOT EXISTS request_seen "
            "(value TEXT PRIMARY KEY) WITHOUT ROWID"
        )
        self.state.execute(
            "CREATE TEMP TABLE IF NOT EXISTS candidate_values "
            "(value TEXT PRIMARY KEY, first_index INTEGER NOT NULL) WITHOUT ROWID"
        )
        identity = self._state_identity()
        stored = self.state.execute(
            "SELECT value FROM metadata WHERE key='identity'"
        ).fetchone()
        if stored is not None and stored[0] != identity:
            raise ValueError(f"persistent generator state has incompatible settings: {path}")
        self.state.execute(
            "INSERT OR IGNORE INTO metadata(key, value) VALUES ('identity', ?)",
            (identity,),
        )
        self.state.commit()
        self._restore_state()

    def _restore_state(self) -> None:
        if self.state is None:
            return
        # SQLite remains the exact global identity authority. Only the bounded
        # current candidate block and surplus queue live in Python memory.
        self.seen.clear()
        self.state.execute("DELETE FROM request_seen")
        self.state.execute("DELETE FROM candidate_values")
        self.surplus = deque(
            str(row[0])
            for row in self.state.execute("SELECT value FROM surplus ORDER BY position")
        )
        values = dict(
            self.state.execute(
                "SELECT key, value FROM metadata WHERE key IN "
                "('total_candidates', 'total_emitted', 'batch_size')"
            )
        )
        self.total_candidates = int(values.get("total_candidates", 0))
        self.total_emitted = int(values.get("total_emitted", 0))
        if "batch_size" in values:
            self.batch_size = int(values["batch_size"])
        self.state.commit()

    @staticmethod
    def _request_key(output: Path, count: int, seed: int) -> str:
        encoded = json.dumps(
            {"output": str(output), "count": count, "seed": seed},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _completed_request(self, key: str, output: Path) -> dict[str, Any] | None:
        if self.state is None:
            return None
        conflict = self.state.execute(
            "SELECT request_key FROM requests WHERE output_path=? AND request_key<>?",
            (str(output), key),
        ).fetchone()
        if conflict is not None:
            raise RuntimeError(
                "persistent generation output was previously committed for a "
                f"different request: {output}"
            )
        row = self.state.execute(
            "SELECT output_path, output_sha256, result_json FROM requests "
            "WHERE request_key=?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        if str(output) != row[0] or not output.is_file() or _sha256(output) != row[1]:
            raise RuntimeError(
                "a committed persistent generation output is missing or changed: "
                f"{output}"
            )
        result = json.loads(row[2])
        result["replayed_committed_request"] = True
        return result

    def _commit_state(
        self,
        key: str,
        output: Path,
        result: dict[str, Any],
    ) -> None:
        assert self.state is not None
        self.state.execute(
            "INSERT OR IGNORE INTO seen(value) SELECT value FROM request_seen"
        )
        self.state.execute("DELETE FROM surplus")
        self.state.executemany(
            "INSERT INTO surplus(position, value) VALUES (?, ?)",
            enumerate(self.surplus),
        )
        self.state.executemany(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            (
                ("total_candidates", str(self.total_candidates)),
                ("total_emitted", str(self.total_emitted)),
                ("batch_size", str(self.batch_size)),
            ),
        )
        self.state.execute(
            "INSERT INTO requests(request_key, output_path, output_sha256, result_json) "
            "VALUES (?, ?, ?, ?)",
            (
                key,
                str(output),
                _sha256(output),
                json.dumps(result, sort_keys=True, allow_nan=False),
            ),
        )
        self.state.commit()

    def _new_canonical_values(self, values: list[str | None]) -> list[str]:
        if self.state is None:
            output = []
            for canonical in values:
                if canonical is None or canonical in self.seen:
                    continue
                self.seen.add(canonical)
                output.append(canonical)
            return output
        self.state.execute("DELETE FROM candidate_values")
        self.state.executemany(
            "INSERT OR IGNORE INTO candidate_values(value, first_index) VALUES (?, ?)",
            (
                (canonical, index)
                for index, canonical in enumerate(values)
                if canonical is not None
            ),
        )
        fresh = [
            str(row[0])
            for row in self.state.execute(
                "SELECT candidate.value FROM candidate_values AS candidate "
                "LEFT JOIN seen ON seen.value = candidate.value "
                "LEFT JOIN request_seen ON request_seen.value = candidate.value "
                "WHERE seen.value IS NULL AND request_seen.value IS NULL "
                "ORDER BY candidate.first_index"
            )
        ]
        self.state.executemany(
            "INSERT INTO request_seen(value) VALUES (?)",
            ((value,) for value in fresh),
        )
        return fresh

    def _persistent_seen_count(self) -> int:
        if self.state is None:
            return len(self.seen)
        row = self.state.execute(
            "SELECT (SELECT COUNT(*) FROM seen) + "
            "(SELECT COUNT(*) FROM request_seen)"
        ).fetchone()
        return int(row[0])

    def _de_novo(self, output: Path, count: int, seed: int) -> dict[str, Any]:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".partial")
        candidates = 0
        emitted = 0
        accepted_from_new_candidates = 0
        candidate_limit = (
            int(self.args.max_candidates)
            if self.args.max_candidates is not None
            else int(math.ceil(count * self.args.max_candidate_multiplier))
        )
        started = time.perf_counter()
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                while emitted < count:
                    while self.surplus and emitted < count:
                        handle.write(self.surplus.popleft())
                        handle.write("\n")
                        emitted += 1
                    if emitted >= count:
                        break
                    if candidates >= candidate_limit:
                        raise RuntimeError(
                            f"iGen3 reached {candidate_limit:,} candidates before producing "
                            f"{count:,} requested rows"
                        )
                    current = min(self.batch_size, candidate_limit - candidates)
                    try:
                        tokens = generate_de_novo_batch(
                            self.generator,
                            current,
                            temperature=self.args.temperature,
                            do_sample=not self.args.greedy,
                            top_k=self.args.top_k,
                        )
                    except RuntimeError as exc:
                        if (
                            not self.auto_batch
                            or not _recoverable_cuda_batch_error(exc)
                            or self.batch_size <= 1
                        ):
                            raise
                        if self.generator.device.type == "cuda":
                            torch.cuda.empty_cache()
                        self.batch_size = _round_batch(max(1, self.batch_size // 2))
                        continue
                    candidates += current
                    self.total_candidates += current
                    canonical_values = self.canonicalizer.canonicalize_tokens(
                        self.generator, tokens
                    )
                    for canonical in self._new_canonical_values(canonical_values):
                        self.surplus.append(canonical)
                        accepted_from_new_candidates += 1
                    del tokens
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, output)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        elapsed = time.perf_counter() - started
        self.total_emitted += emitted
        return {
            "requested": count,
            "generated": emitted,
            "candidates_generated": candidates,
            "accepted_from_new_candidates": accepted_from_new_candidates,
            "surplus_rows": len(self.surplus),
            "seconds": elapsed,
            "smiles_per_second": emitted / max(elapsed, 1.0e-9),
            "candidate_smiles_per_second": candidates / max(elapsed, 1.0e-9),
            "batch_size": self.batch_size,
            "stopped_reason": "target",
            "persistent_seen_rows": self._persistent_seen_count(),
            "canonical_workers": self.canonicalizer.workers,
        }

    def _derivative(self, output: Path, count: int, seed: int) -> dict[str, Any]:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        stats = write_derivative_file(
            self.generator,
            seeds=self.seeds,
            output_path=output,
            batch_size=self.batch_size,
            samples_per_seed=self.args.samples_per_seed,
            count=count,
            temperature=self.args.temperature,
            do_sample=not self.args.greedy,
            top_k=self.args.top_k,
            exclude_seed_molecules=not self.args.include_seed_molecules,
            max_candidates=self.args.max_candidates,
            max_candidate_multiplier=self.args.max_candidate_multiplier,
            stagnation_limit=self.args.stagnation_limit,
            progress=False,
        )
        return {
            "requested": stats.requested,
            "generated": stats.generated,
            "candidates_generated": stats.candidates_generated,
            "seconds": stats.seconds,
            "smiles_per_second": stats.smiles_per_second,
            "candidate_smiles_per_second": stats.candidate_smiles_per_second,
            "batch_size": stats.batch_size,
            "stopped_reason": stats.stopped_reason,
            "surplus_rows": 0,
        }

    def generate(self, request: dict[str, Any]) -> dict[str, Any]:
        output = Path(request["output"]).expanduser().resolve()
        count = int(request["count"])
        seed = int(request["seed"])
        if count <= 0:
            raise ValueError("request count must be positive")
        request_key = self._request_key(output, count, seed)
        completed = self._completed_request(request_key, output)
        if completed is not None:
            return completed
        if self.state is not None:
            self.state.execute("BEGIN IMMEDIATE")
            self.state.execute("DELETE FROM request_seen")
        if self.args.mode == "de-novo":
            try:
                result = self._de_novo(output, count, seed)
            except BaseException:
                if self.state is not None:
                    self.state.rollback()
                    self._restore_state()
                raise
        else:
            result = self._derivative(output, count, seed)
        try:
            if request.get("metrics_dir"):
                _, summary = save_metrics(
                    output,
                    model_id=self.spec.model_id,
                    isomeric_smiles=self.spec.is_isomeric,
                    output_dir=Path(request["metrics_dir"]).resolve(),
                )
                result["metrics"] = summary.to_dict(orient="records")
            if self.state is not None:
                self._commit_state(request_key, output, result)
        except BaseException:
            if self.state is not None:
                self.state.rollback()
                self._restore_state()
            raise
        return result

    def close(self) -> None:
        self.canonicalizer.close()
        if self.state is not None:
            self.state.close()
            self.state = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Persistent iGen3 JSON-lines worker")
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("de-novo", "derivative"), required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--max-batch-size", type=int, default=32_768)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--profile-cache", type=Path)
    parser.add_argument("--state-file", type=Path)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--seed-file", type=Path)
    parser.add_argument("--seed-smiles", action="append", default=[])
    parser.add_argument("--samples-per-seed", type=int, default=1)
    parser.add_argument("--include-seed-molecules", action="store_true")
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--max-candidate-multiplier", type=float, default=50.0)
    parser.add_argument("--stagnation-limit", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.expected_count <= 0 or args.max_batch_size <= 0:
        raise SystemExit("expected count and maximum batch size must be positive")
    if args.batch_size != "auto" and int(args.batch_size) <= 0:
        raise SystemExit("batch size must be positive or auto")
    if args.max_candidate_multiplier < 1:
        raise SystemExit("maximum candidate multiplier must be at least one")
    spec = resolve_model(args.model)
    if args.temperature is None:
        args.temperature = spec.default_temperature(args.mode)
    if args.top_k is None:
        args.top_k = spec.default_top_k
    started = time.perf_counter()
    engine = PersistentGenerator(args)
    _emit(
        {
            "event": "ready",
            "protocol_version": PROTOCOL_VERSION,
            "worker": "iGen3",
            "startup_seconds": time.perf_counter() - started,
            "batch_size": engine.batch_size,
            "calibration": engine.calibration,
        }
    )
    for line in sys.stdin:
        if not line.strip():
            continue
        request: dict[str, Any] = {}
        try:
            request = json.loads(line)
            request_id = request.get("request_id")
            if request.get("command") == "shutdown":
                engine.close()
                _emit({"event": "stopped", "request_id": request_id})
                return 0
            if request.get("command") != "generate":
                raise ValueError("unknown worker command")
            result = engine.generate(request)
            _emit({"event": "result", "request_id": request_id, "result": result})
        except BaseException as exc:
            _emit(
                {
                    "event": "error",
                    "request_id": request.get("request_id"),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
    engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
