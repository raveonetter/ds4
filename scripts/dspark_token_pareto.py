#!/usr/bin/env python3
"""Convert a DSpark quality-matrix run to speed/token-divergence Pareto metrics.

Prompt execution remains in run_dspark_quality_matrix.sh and
run_dspark_family_variants.sh. This postprocessor evaluates:

Per prompt:
  - generation_tps
  - first generated token position differing from sequential

Across prompts for the same arm/configuration:
  - median_generation_tps
  - robust_first_diff_token = earliest divergence on any prompt

The robust table is the deployment-facing frontier. Byte/checkpoint exactness
remains diagnostic only.

The current CLI has no native generated-token trace, so saved generated text is
re-tokenized with the same model and ds4 tokenizer. Tokenization runs after the
measured decode and therefore does not contaminate generation_tps.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
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
    first_diff_numeric: float = 0.0
    pareto: bool = False


@dataclass
class RobustRow:
    arm: str
    confidence: str
    status: str
    prompt_count: int
    median_tps: float | None
    robust_first_diff: int | None
    robust_first_diff_numeric: float
    pareto: bool = False


def die(msg: str) -> None:
    raise SystemExit(f"dspark_token_pareto: {msg}")


def parse_token_ids(text: str, probe: bool = False) -> list[int]:
    """Parse the token-ID vector emitted by ds4 --dump-tokens."""
    s = text.strip()
    if not s:
        return []

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

    # Current ds4 format: first non-empty line is the complete bare vector,
    # followed by a human-readable token table.
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
    """Return the 1-based first differing token, or None through the horizon."""
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


def mark_prompt_pareto(group: list[Row]) -> None:
    valid = [r for r in group if r.status == "PASS" and r.tps is not None]
    for r in valid:
        r.pareto = not any(
            q is not r
            and q.tps >= r.tps
            and q.first_diff_numeric >= r.first_diff_numeric
            and (q.tps > r.tps or q.first_diff_numeric > r.first_diff_numeric)
            for q in valid
        )


def evaluate(run_dir: Path, ds4: Path, model: Path, rows: list[Row]) -> list[str]:
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

            # Within one prompt, NONE means exact through the measured horizon.
            r.first_diff_numeric = (
                math.inf if r.first_diff is None else float(r.first_diff)
            )

        mark_prompt_pareto(group)
    return prompts


def build_robust_rows(rows: list[Row], prompts: Sequence[str]) -> list[RobustRow]:
    prompt_set = set(prompts)
    keys = sorted({(r.arm, r.confidence) for r in rows})
    result: list[RobustRow] = []

    for arm, confidence in keys:
        group = [r for r in rows if r.arm == arm and r.confidence == confidence]
        by_prompt = {r.prompt: r for r in group}
        complete = (
            set(by_prompt) == prompt_set
            and all(r.status == "PASS" and r.tps is not None for r in group)
        )

        if not complete:
            result.append(
                RobustRow(
                    arm=arm,
                    confidence=confidence,
                    status="INCOMPLETE",
                    prompt_count=len(by_prompt),
                    median_tps=None,
                    robust_first_diff=None,
                    robust_first_diff_numeric=0.0,
                )
            )
            continue

        tps_values = [r.tps for r in group if r.tps is not None]
        divergences = [r.first_diff for r in group if r.first_diff is not None]
        robust_first = min(divergences) if divergences else None
        robust_numeric = math.inf if robust_first is None else float(robust_first)

        result.append(
            RobustRow(
                arm=arm,
                confidence=confidence,
                status="PASS",
                prompt_count=len(group),
                median_tps=statistics.median(tps_values),
                robust_first_diff=robust_first,
                robust_first_diff_numeric=robust_numeric,
            )
        )

    valid = [r for r in result if r.status == "PASS" and r.median_tps is not None]
    for r in valid:
        r.pareto = not any(
            q is not r
            and q.median_tps >= r.median_tps
            and q.robust_first_diff_numeric >= r.robust_first_diff_numeric
            and (
                q.median_tps > r.median_tps
                or q.robust_first_diff_numeric > r.robust_first_diff_numeric
            )
            for q in valid
        )
    return result


def write_prompt_table(path: Path, rows: Sequence[Row]) -> None:
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
                "prompt_pareto",
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


def write_robust_table(path: Path, rows: Sequence[RobustRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(
            [
                "arm",
                "confidence",
                "status",
                "prompt_count",
                "median_generation_tps",
                "robust_first_diff_token",
                "robust_pareto",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.arm,
                    r.confidence,
                    r.status,
                    r.prompt_count,
                    "NA" if r.median_tps is None else f"{r.median_tps:.2f}",
                    "NA"
                    if r.status != "PASS"
                    else (
                        "NONE"
                        if r.robust_first_diff is None
                        else r.robust_first_diff
                    ),
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
    prompts = evaluate(run_dir, ds4, model, rows)
    robust_rows = build_robust_rows(rows, prompts)

    prompt_out = run_dir / "pareto.tsv"
    robust_out = run_dir / "robust-pareto.tsv"
    write_prompt_table(prompt_out, rows)
    write_robust_table(robust_out, robust_rows)

    print("\nPARETO_MATRIX")
    print(prompt_out.read_text(encoding="utf-8"), end="")
    print("\nROBUST_PARETO_MATRIX")
    print(robust_out.read_text(encoding="utf-8"), end="")
    print("TOKEN_POSITION_INDEXING\t1-based")
    print("TOKEN_SOURCE\tretokenized_generated_output")
    print("ROBUST_SPEED_AGGREGATION\tmedian_across_prompts")
    print("ROBUST_FIDELITY_AGGREGATION\tminimum_first_diff_across_prompts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
