#!/usr/bin/env python3
"""Aggregate repeated generic DSpark confidence trials.

Primary deployment metrics remain exactly:
  * median generation_tps
  * first generated token divergence vs sequential

Repeated runs are used to reduce timing noise and to verify that the observed
first-divergence position is deterministic.  Per prompt, throughput is the
median across repeats and fidelity is conservative: the earliest divergence
seen in any repeat.  Across prompts, throughput is the median of per-prompt
medians and fidelity is again the earliest divergence on any prompt.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from dspark_token_pareto import first_divergence, run_tokenizer


@dataclass
class Sample:
    prompt: str
    arm: str
    confidence: str
    repeat: int
    status: str
    tps: float | None
    output_file: Path
    first_diff: int | None = None


@dataclass
class PromptRow:
    prompt: str
    arm: str
    confidence: str
    status: str
    repeat_count: int
    median_tps: float | None
    first_diff: int | None
    first_diff_numeric: float
    first_diff_stable: bool
    pareto: bool = False


@dataclass
class RobustRow:
    arm: str
    confidence: str
    status: str
    prompt_count: int
    median_tps: float | None
    first_diff: int | None
    first_diff_numeric: float
    first_diff_stable: bool
    pareto: bool = False


def die(msg: str) -> None:
    raise SystemExit(f"dspark_confidence_pareto: {msg}")


def load_samples(path: Path) -> list[Sample]:
    samples: list[Sample] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            raw_tps = row.get("generation_tps", "NA")
            tps = None if raw_tps in ("", "NA") else float(raw_tps)
            samples.append(
                Sample(
                    prompt=row["prompt"],
                    arm=row["arm"],
                    confidence=row["confidence"],
                    repeat=int(row["repeat"]),
                    status=row["status"],
                    tps=tps,
                    output_file=Path(row["output_file"]),
                )
            )
    if not samples:
        die(f"no samples in {path}")
    return samples


def resolve_output(path: Path, run_dir: Path) -> Path:
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    candidate = run_dir / path
    if candidate.exists():
        return candidate.resolve()
    return path.resolve()


def token_ids_cached(
    path: Path,
    ds4: Path,
    model: Path,
    cache: dict[str, list[int]],
) -> list[int]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        die(f"cannot read generated output {path}: {exc}")
    digest = hashlib.sha256(data).hexdigest()
    cached = cache.get(digest)
    if cached is not None:
        return cached
    ids = run_tokenizer(ds4, model, path)
    cache[digest] = ids
    return ids


def evaluate_samples(
    samples: list[Sample], run_dir: Path, ds4: Path, model: Path
) -> list[str]:
    prompts = sorted({s.prompt for s in samples})
    cache: dict[str, list[int]] = {}

    for prompt in prompts:
        group = [s for s in samples if s.prompt == prompt]
        oracle_samples = sorted(
            [s for s in group if s.arm == "sequential" and s.status == "PASS"],
            key=lambda s: s.repeat,
        )
        if not oracle_samples:
            die(f"{prompt}: no PASS sequential oracle sample")

        oracle_path = resolve_output(oracle_samples[0].output_file, run_dir)
        oracle_tokens = token_ids_cached(oracle_path, ds4, model, cache)
        if not oracle_tokens:
            die(f"{prompt}: sequential oracle tokenized to zero tokens")

        for sample in group:
            if sample.status != "PASS" or sample.tps is None:
                continue
            output = resolve_output(sample.output_file, run_dir)
            tokens = token_ids_cached(output, ds4, model, cache)
            sample.first_diff = first_divergence(oracle_tokens, tokens)

    return prompts


def conservative_first_diff(values: Sequence[int | None]) -> int | None:
    finite = [v for v in values if v is not None]
    return min(finite) if finite else None


def first_diff_stable(values: Sequence[int | None]) -> bool:
    return len(set(values)) <= 1


def mark_pareto(rows: Sequence[PromptRow | RobustRow]) -> None:
    valid = [r for r in rows if r.status == "PASS" and r.median_tps is not None]
    for row in valid:
        row.pareto = not any(
            other is not row
            and other.median_tps >= row.median_tps
            and other.first_diff_numeric >= row.first_diff_numeric
            and (
                other.median_tps > row.median_tps
                or other.first_diff_numeric > row.first_diff_numeric
            )
            for other in valid
        )


def build_prompt_rows(
    samples: list[Sample], expected_repeats: int
) -> list[PromptRow]:
    keys = sorted({(s.prompt, s.arm, s.confidence) for s in samples})
    result: list[PromptRow] = []

    for prompt, arm, confidence in keys:
        group = sorted(
            [
                s
                for s in samples
                if s.prompt == prompt and s.arm == arm and s.confidence == confidence
            ],
            key=lambda s: s.repeat,
        )
        pass_group = [s for s in group if s.status == "PASS" and s.tps is not None]
        complete = len(group) == expected_repeats and len(pass_group) == expected_repeats

        if not complete:
            result.append(
                PromptRow(
                    prompt=prompt,
                    arm=arm,
                    confidence=confidence,
                    status="INCOMPLETE",
                    repeat_count=len(pass_group),
                    median_tps=None,
                    first_diff=None,
                    first_diff_numeric=0.0,
                    first_diff_stable=False,
                )
            )
            continue

        divergences = [s.first_diff for s in pass_group]
        first = conservative_first_diff(divergences)
        result.append(
            PromptRow(
                prompt=prompt,
                arm=arm,
                confidence=confidence,
                status="PASS",
                repeat_count=len(pass_group),
                median_tps=statistics.median(s.tps for s in pass_group if s.tps is not None),
                first_diff=first,
                first_diff_numeric=math.inf if first is None else float(first),
                first_diff_stable=first_diff_stable(divergences),
            )
        )

    for prompt in sorted({r.prompt for r in result}):
        mark_pareto([r for r in result if r.prompt == prompt])
    return result


def build_robust_rows(prompt_rows: list[PromptRow], prompts: Sequence[str]) -> list[RobustRow]:
    prompt_set = set(prompts)
    keys = sorted({(r.arm, r.confidence) for r in prompt_rows})
    result: list[RobustRow] = []

    for arm, confidence in keys:
        group = [r for r in prompt_rows if r.arm == arm and r.confidence == confidence]
        by_prompt = {r.prompt: r for r in group}
        complete = (
            set(by_prompt) == prompt_set
            and all(r.status == "PASS" and r.median_tps is not None for r in group)
        )

        if not complete:
            result.append(
                RobustRow(
                    arm=arm,
                    confidence=confidence,
                    status="INCOMPLETE",
                    prompt_count=len(by_prompt),
                    median_tps=None,
                    first_diff=None,
                    first_diff_numeric=0.0,
                    first_diff_stable=False,
                )
            )
            continue

        first = conservative_first_diff([r.first_diff for r in group])
        result.append(
            RobustRow(
                arm=arm,
                confidence=confidence,
                status="PASS",
                prompt_count=len(group),
                median_tps=statistics.median(
                    r.median_tps for r in group if r.median_tps is not None
                ),
                first_diff=first,
                first_diff_numeric=math.inf if first is None else float(first),
                first_diff_stable=all(r.first_diff_stable for r in group),
            )
        )

    mark_pareto(result)
    return result


def fmt_first(value: int | None, status: str) -> str:
    if status != "PASS":
        return "NA"
    return "NONE" if value is None else str(value)


def write_prompt_table(path: Path, rows: Sequence[PromptRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(
            [
                "prompt",
                "arm",
                "confidence",
                "status",
                "repeats",
                "median_generation_tps",
                "first_diff_token",
                "first_diff_stable",
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
                    r.repeat_count,
                    "NA" if r.median_tps is None else f"{r.median_tps:.2f}",
                    fmt_first(r.first_diff, r.status),
                    "YES" if r.first_diff_stable else "NO",
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
                "first_diff_stable",
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
                    fmt_first(r.first_diff, r.status),
                    "YES" if r.first_diff_stable else "NO",
                    "YES" if r.pareto else "NO",
                ]
            )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--ds4-bin", default="./ds4", type=Path)
    ap.add_argument("--model", default="./ds4flash.gguf", type=Path)
    ap.add_argument("--expected-repeats", required=True, type=int)
    args = ap.parse_args()

    if args.expected_repeats <= 0:
        die("--expected-repeats must be positive")

    run_dir = args.run_dir.expanduser().resolve()
    ds4 = args.ds4_bin.expanduser().resolve()
    model = args.model.expanduser().resolve()
    samples_path = run_dir / "samples.tsv"
    for path in (run_dir, ds4, model, samples_path):
        if not path.exists():
            die(f"missing {path}")

    samples = load_samples(samples_path)
    prompts = evaluate_samples(samples, run_dir, ds4, model)
    prompt_rows = build_prompt_rows(samples, args.expected_repeats)
    robust_rows = build_robust_rows(prompt_rows, prompts)

    prompt_out = run_dir / "prompt-pareto.tsv"
    robust_out = run_dir / "robust-pareto.tsv"
    write_prompt_table(prompt_out, prompt_rows)
    write_robust_table(robust_out, robust_rows)

    print("\nCONFIDENCE_PROMPT_MATRIX")
    print(prompt_out.read_text(encoding="utf-8"), end="")
    print("\nCONFIDENCE_ROBUST_PARETO_MATRIX")
    print(robust_out.read_text(encoding="utf-8"), end="")
    print("TOKEN_POSITION_INDEXING\t1-based")
    print("TOKEN_SOURCE\tretokenized_generated_output")
    print("REPEAT_SPEED_AGGREGATION\tmedian_within_prompt")
    print("ROBUST_SPEED_AGGREGATION\tmedian_of_prompt_medians")
    print("REPEAT_FIDELITY_AGGREGATION\tearliest_first_diff")
    print("ROBUST_FIDELITY_AGGREGATION\tminimum_first_diff_across_prompts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
