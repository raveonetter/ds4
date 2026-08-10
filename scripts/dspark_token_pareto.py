#!/usr/bin/env python3
"""Convert a DSpark quality-matrix run to the speed/token-divergence Pareto metric.

This intentionally leaves prompt generation and arm execution to
run_dspark_quality_matrix.sh. It only changes evaluation:
  - x axis: generation_tps from performance.tsv
  - y axis: first generated token position differing from sequential

The current CLI has no native generated-token trace, so saved generated text is
re-tokenized with the same model and ds4 tokenizer. Tokenization runs after the
timed decode and therefore does not contaminate generation_tps.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass
class Row:
    prompt: str
    arm: str
    confidence: str
    status: str
    tps: float | None
    first_diff: int | None = None
    first_diff_numeric: int = 0
    pareto: bool = False


def die(msg: str) -> None:
    raise SystemExit(f"dspark_token_pareto: {msg}")


def parse_token_ids(text: str, probe: bool = False) -> list[int]:
    """Parse the token-ID vector emitted by ds4 --dump-tokens.

    Current ds4 output is intentionally human-readable.  Its first line is a
    bare JSON-compatible integer vector, followed by one line per token:

        [123, 456, ...]
           123  token-text
           456  token-text

    Older/diagnostic variants are accepted as fallbacks so the postprocessor
    remains usable across nearby experiment branches.
    """
    s = text.strip()
    if not s:
        return []

    # Some builds may emit only the vector.  Accept that directly.
    try:
        obj = json.loads(s)
        if isinstance(obj, list) and all(isinstance(x, int) for x in obj):
            return list(obj)
        if isinstance(obj, dict):
            for key in ("tokens", "token_ids", "ids"):
                value = obj.get(key)
                if isinstance(value, list) and all(isinstance(x, int) for x in value):
                    return list(value)
    except json.JSONDecodeError:
        pass

    lines = text.splitlines()

    # Current ds4 format: the first non-empty line is the complete bare vector,
    # while subsequent lines contain the human-readable token table.
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                obj = None
            if isinstance(obj, list) and all(isinstance(x, int) for x in obj):
                return list(obj)
        break

    # Compatibility with labeled one-line vectors.
    for line in lines:
        m = re.match(
            r"^\s*(?:tokens|token_ids|ids)\s*[:=]\s*(\[[^]]*\])\s*$",
            line,
            re.I,
        )
        if m:
            try:
                obj = json.loads(m.group(1))
            except json.JSONDecodeError:
                obj = None
            if isinstance(obj, list) and all(isinstance(x, int) for x in obj):
                return list(obj)

    # Last-resort compatibility with line-oriented ID dumps.  The final pattern
    # also matches ds4's human table ("   123  token-text"), but current builds
    # should already have returned from the bare-vector path above.
    ids: list[int] = []
    patterns = [
        re.compile(r"^\s*(-?\d+)\s*$"),
        re.compile(
            r"^\s*(?:token(?:_id)?|id)\s*[:=]\s*(-?\d+)(?:\s|$)", re.I
        ),
        re.compile(r"^\s*token\[\d+\]\s*[:=]\s*(-?\d+)(?:\s|$)", re.I),
        re.compile(r"^\s*\d+\s*(?::|\t)\s*(-?\d+)(?:\s|$)"),
        re.compile(r"^\s*(-?\d+)\s{2,}.*$"),
    ]
    for line in lines:
        for pat in patterns:
            m = pat.match(line)
            if m:
                ids.append(int(m.group(1)))
                break
    if ids or probe:
        return ids
    die("unrecognized --dump-tokens format")


def run_tokenizer(ds4: Path, model: Path, text_file: Path) -> list[int]:
    cmd = [
        str(ds4),
        "-m",
        str(model),
        "--raw-prompt",
        "--dump-tokens",
        "--prompt-file",
        str(text_file),
    ]
    p = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if p.returncode != 0:
        die(f"tokenizer failed for {text_file}: {p.stderr.strip()}")
    for stream in (p.stdout, p.stderr):
        ids = parse_token_ids(stream, probe=True)
        if ids:
            return ids
    stdout_head = "\\n".join(p.stdout.splitlines()[:3])
    stderr_head = "\\n".join(p.stderr.splitlines()[:3])
    die(
        f"no parseable token IDs from --dump-tokens for {text_file}; "
        f"stdout_head={stdout_head!r} stderr_head={stderr_head!r}"
    )


def first_divergence(oracle: Sequence[int], candidate: Sequence[int]) -> int | None:
    n = min(len(oracle), len(candidate))
    for i in range(n):
        if oracle[i] != candidate[i]:
            return i + 1
    if len(oracle) != len(candidate):
        return n + 1
    return None


def load_performance(path: Path) -> list[Row]:
    rows: list[Row] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            raw_tps = r.get("generation_tps", "NA")
            tps = None if raw_tps in ("", "NA") else float(raw_tps)
            rows.append(
                Row(r["prompt"], r["arm"], r["confidence"], r["status"], tps)
            )
    if not rows:
        die(f"no rows in {path}")
    return rows


def evaluate(run_dir: Path, ds4: Path, model: Path, rows: list[Row]) -> None:
    prompts = sorted({r.prompt for r in rows})
    for prompt in prompts:
        group = [r for r in rows if r.prompt == prompt]
        oracle_row = next(
            (r for r in group if r.arm == "sequential" and r.status == "PASS"),
            None,
        )
        if oracle_row is None:
            die(f"{prompt}: missing PASS sequential oracle")
        oracle_tokens = run_tokenizer(ds4, model, run_dir / prompt / "sequential.txt")
        if not oracle_tokens:
            die(f"{prompt}: sequential output tokenized to zero tokens")
        horizon = len(oracle_tokens)

        for r in group:
            if r.status != "PASS" or r.tps is None:
                continue
            if r.arm == "sequential":
                r.first_diff = None
            else:
                candidate = run_tokenizer(
                    ds4, model, run_dir / prompt / f"{r.arm}.txt"
                )
                r.first_diff = first_divergence(oracle_tokens, candidate)
            r.first_diff_numeric = horizon + 1 if r.first_diff is None else r.first_diff

        valid = [r for r in group if r.status == "PASS" and r.tps is not None]
        for r in valid:
            r.pareto = not any(
                q is not r
                and q.tps >= r.tps
                and q.first_diff_numeric >= r.first_diff_numeric
                and (q.tps > r.tps or q.first_diff_numeric > r.first_diff_numeric)
                for q in valid
            )


def write_table(path: Path, rows: list[Row]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(
            [
                "prompt",
                "arm",
                "confidence",
                "status",
                "generation_tps",
                "first_diff_token",
                "pareto",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.prompt,
                    r.arm,
                    r.confidence,
                    r.status,
                    "NA" if r.tps is None else f"{r.tps:.2f}",
                    "NA"
                    if r.status != "PASS"
                    else ("NONE" if r.first_diff is None else r.first_diff),
                    "YES" if r.pareto else "NO",
                ]
            )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--ds4-bin", default="./ds4", type=Path)
    ap.add_argument("--model", default="./ds4flash.gguf", type=Path)
    args = ap.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    ds4 = args.ds4_bin.expanduser().resolve()
    model = args.model.expanduser().resolve()
    perf = run_dir / "performance.tsv"
    for p in (ds4, model, perf):
        if not p.exists():
            die(f"missing {p}")

    rows = load_performance(perf)
    evaluate(run_dir, ds4, model, rows)
    out = run_dir / "pareto.tsv"
    write_table(out, rows)

    print("\nPARETO_MATRIX")
    print(out.read_text(encoding="utf-8"), end="")
    print("TOKEN_POSITION_INDEXING\t1-based")
    print("TOKEN_SOURCE\tretokenized_generated_output")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
