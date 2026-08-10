#!/usr/bin/env python3
"""Benchmark DSpark configurations on decode speed and first token divergence.

The sequential arm is the oracle.  Every other arm runs the same prompt and
same generation horizon.  The primary metrics are:

  * generation_tps: throughput reported by ds4's real generation loop.
  * first_diff_token: 1-based first generated-token position that differs from
    the sequential oracle; NONE means equality through the measured horizon.

The current ds4 CLI does not expose generated token IDs directly.  This harness
therefore captures generated stdout and retokenizes it with the same ds4 binary
using --raw-prompt --dump-tokens.  Results explicitly report
``token_source=retokenized_output`` so a future native token trace can replace
this collector without changing the benchmark definition.

Configuration is JSON.  Arms can use different binaries, which makes branch or
family builds (generic, F1, F1+F3, ...) first-class benchmark configurations.
No benchmark prompt is hard-coded or rewritten.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

TPS_RE = re.compile(r"generation:\s*([0-9]+(?:\.[0-9]+)?)\s*tokens/s")


@dataclass
class RunResult:
    prompt: str
    arm: str
    confidence: str
    oracle: bool
    generation_tps: float
    token_ids: list[int]
    stdout: str
    stderr: str
    command: list[str]


@dataclass
class MetricRow:
    prompt: str
    arm: str
    confidence: str
    generation_tps: float
    speedup_vs_sequential: float
    first_diff_token: int | None
    first_diff_token_numeric: int
    matched_prefix_tokens: int
    oracle_tokens: int
    normalized_fidelity: float
    exact_through_horizon: bool
    pareto: bool = False


def fail(message: str) -> "NoReturn":
    raise SystemExit(f"dspark_pareto_bench: {message}")


def load_config(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read config {path}: {exc}")
    if not isinstance(obj, dict):
        fail("top-level config must be a JSON object")
    return obj


def string_list(value: Any, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        fail(f"{field} must be an array of strings")
    return list(value)


def resolve_path(value: str, config_dir: Path) -> str:
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = config_dir / p
    return str(p.resolve())


def prompt_arg(prompt_cfg: dict[str, Any], config_dir: Path, stack: Any) -> tuple[str, list[str]]:
    name = prompt_cfg.get("name")
    if not isinstance(name, str) or not name:
        fail("every prompt needs a non-empty name")

    has_file = "file" in prompt_cfg
    has_text = "text" in prompt_cfg
    if has_file == has_text:
        fail(f"prompt {name!r} must contain exactly one of 'file' or 'text'")

    if has_file:
        value = prompt_cfg["file"]
        if not isinstance(value, str):
            fail(f"prompt {name!r} file must be a string")
        return name, ["--prompt-file", resolve_path(value, config_dir)]

    text = prompt_cfg["text"]
    if not isinstance(text, str):
        fail(f"prompt {name!r} text must be a string")
    f = stack.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False)
    f.write(text)
    f.close()
    stack.paths.append(f.name)
    return name, ["--prompt-file", f.name]


class TempStack:
    def __init__(self) -> None:
        self.paths: list[str] = []

    def NamedTemporaryFile(self, *args: Any, **kwargs: Any) -> Any:
        return tempfile.NamedTemporaryFile(*args, **kwargs)

    def cleanup(self) -> None:
        for path in self.paths:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass


def run_command(command: Sequence[str], timeout: float | None) -> subprocess.CompletedProcess[str]:
    try:
        proc = subprocess.run(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        fail(f"command failed to start/finish: {command!r}: {exc}")
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        fail(f"command returned {proc.returncode}: {command!r}")
    return proc


def parse_tps(stderr: str) -> float:
    matches = TPS_RE.findall(stderr)
    if not matches:
        fail("generation throughput not found in ds4 stderr")
    return float(matches[-1])


def _json_token_ids(text: str) -> list[int] | None:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(obj, list) and all(isinstance(x, int) for x in obj):
        return list(obj)
    if isinstance(obj, dict):
        for key in ("tokens", "token_ids", "ids"):
            value = obj.get(key)
            if isinstance(value, list) and all(isinstance(x, int) for x in value):
                return list(value)
    return None


def parse_dump_tokens(text: str) -> list[int]:
    """Accept the common human and JSON forms emitted by --dump-tokens.

    Deliberately avoids extracting arbitrary integers from token text.  This is
    strict so a changed CLI format fails loudly instead of silently corrupting
    the fidelity metric.
    """
    parsed = _json_token_ids(text.strip())
    if parsed is not None:
        return parsed

    ids: list[int] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        # "123"
        m = re.fullmatch(r"(-?\d+)", line)
        if m:
            ids.append(int(m.group(1)))
            continue
        # "token=123 ...", "id: 123 ...", "token_id=123 ..."
        m = re.match(r"(?:token(?:_id)?|id)\s*[:=]\s*(-?\d+)(?:\s|$)", line, re.I)
        if m:
            ids.append(int(m.group(1)))
            continue
        # "17: 123 ..." or "17\t123 ..." (index then token id)
        m = re.match(r"\d+\s*(?::|\t)\s*(-?\d+)(?:\s|$)", line)
        if m:
            ids.append(int(m.group(1)))
            continue
        # Ignore known summary/header lines; reject anything else below if no
        # token IDs were recognized at all.

    if not ids:
        sample = "\\n".join(text.splitlines()[:8])
        fail(f"could not parse --dump-tokens output; first lines:\n{sample}")
    return ids


def retokenize_generated(
    binary: str,
    model: str,
    generated_text: str,
    tokenizer_args: Sequence[str],
    timeout: float | None,
) -> list[int]:
    # --raw-prompt is required: generated text must be tokenized as bytes/text,
    # never wrapped in a chat template.
    command = [binary, "-m", model, "--raw-prompt", "--dump-tokens", "-p", generated_text]
    command.extend(tokenizer_args)
    proc = run_command(command, timeout)
    # Some ds4 diagnostics use stdout, some builds use stderr.  Prefer stdout,
    # then fall back to stderr without mixing the streams.
    try:
        return parse_dump_tokens(proc.stdout)
    except SystemExit:
        return parse_dump_tokens(proc.stderr)


def first_divergence(oracle: Sequence[int], candidate: Sequence[int]) -> tuple[int | None, int]:
    """Return (1-based first diff position or None, matched prefix count)."""
    common = min(len(oracle), len(candidate))
    for i in range(common):
        if oracle[i] != candidate[i]:
            return i + 1, i
    if len(oracle) != len(candidate):
        return common + 1, common
    return None, len(oracle)


def build_generation_command(
    config: dict[str, Any],
    arm: dict[str, Any],
    prompt_args: Sequence[str],
    config_dir: Path,
) -> tuple[str, str, list[str]]:
    binary_value = arm.get("binary", config.get("binary", "./ds4"))
    model_value = arm.get("model", config.get("model"))
    if not isinstance(binary_value, str) or not binary_value:
        fail("binary must be a non-empty string")
    if not isinstance(model_value, str) or not model_value:
        fail("model must be set globally or per arm")

    binary = resolve_path(binary_value, config_dir)
    model = resolve_path(model_value, config_dir)
    command = [binary, "-m", model]
    command.extend(prompt_args)

    tokens = arm.get("tokens", config.get("tokens"))
    if tokens is not None:
        if not isinstance(tokens, int) or tokens <= 0:
            fail("tokens must be a positive integer")
        command.extend(["--tokens", str(tokens)])

    command.extend(string_list(config.get("common_args"), "common_args"))
    command.extend(string_list(arm.get("args"), f"arm {arm.get('name')} args"))
    return binary, model, command


def run_arm(
    config: dict[str, Any],
    arm: dict[str, Any],
    prompt_name: str,
    prompt_args: Sequence[str],
    config_dir: Path,
    timeout: float | None,
) -> RunResult:
    name = arm.get("name")
    if not isinstance(name, str) or not name:
        fail("every arm needs a non-empty name")
    binary, model, command = build_generation_command(config, arm, prompt_args, config_dir)
    proc = run_command(command, timeout)
    tps = parse_tps(proc.stderr)
    tokenizer_args = string_list(config.get("tokenizer_args"), "tokenizer_args")
    token_ids = retokenize_generated(binary, model, proc.stdout, tokenizer_args, timeout)
    confidence = arm.get("confidence", "NA")
    return RunResult(
        prompt=prompt_name,
        arm=name,
        confidence=str(confidence),
        oracle=bool(arm.get("oracle", False)),
        generation_tps=tps,
        token_ids=token_ids,
        stdout=proc.stdout,
        stderr=proc.stderr,
        command=command,
    )


def mark_pareto(rows: list[MetricRow]) -> None:
    """Mark non-dominated speed/first-divergence points per prompt.

    first_diff_token_numeric is capped by the oracle horizon.  NONE is encoded
    as oracle_tokens + 1, so exact-through-horizon is strictly better than a
    divergence at the last measured token.
    """
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            speed_ge = other.generation_tps >= row.generation_tps
            fidelity_ge = other.first_diff_token_numeric >= row.first_diff_token_numeric
            strict = (
                other.generation_tps > row.generation_tps
                or other.first_diff_token_numeric > row.first_diff_token_numeric
            )
            if speed_ge and fidelity_ge and strict:
                dominated = True
                break
        row.pareto = not dominated


def evaluate_prompt(runs: list[RunResult]) -> list[MetricRow]:
    oracles = [r for r in runs if r.oracle]
    if len(oracles) != 1:
        fail(f"prompt {runs[0].prompt!r} requires exactly one oracle arm, got {len(oracles)}")
    oracle = oracles[0]
    horizon = len(oracle.token_ids)
    if horizon == 0:
        fail(f"prompt {oracle.prompt!r}: oracle produced zero retokenized tokens")

    rows: list[MetricRow] = []
    for run in runs:
        if run.oracle:
            first = None
            matched = horizon
        else:
            first, matched = first_divergence(oracle.token_ids, run.token_ids)
        numeric = first if first is not None else horizon + 1
        fidelity = min(matched, horizon) / horizon
        rows.append(
            MetricRow(
                prompt=run.prompt,
                arm=run.arm,
                confidence=run.confidence,
                generation_tps=run.generation_tps,
                speedup_vs_sequential=run.generation_tps / oracle.generation_tps,
                first_diff_token=first,
                first_diff_token_numeric=numeric,
                matched_prefix_tokens=matched,
                oracle_tokens=horizon,
                normalized_fidelity=fidelity,
                exact_through_horizon=(first is None),
            )
        )
    mark_pareto(rows)
    return rows


def print_matrix(rows: Sequence[MetricRow]) -> None:
    print("PARETO_MATRIX")
    header = (
        "prompt", "arm", "confidence", "generation_tps", "speedup_vs_sequential",
        "first_diff_token", "matched_prefix_tokens", "oracle_tokens",
        "normalized_fidelity", "pareto",
    )
    print("\t".join(header))
    for row in rows:
        first = "NONE" if row.first_diff_token is None else str(row.first_diff_token)
        print(
            "\t".join(
                [
                    row.prompt,
                    row.arm,
                    row.confidence,
                    f"{row.generation_tps:.2f}",
                    f"{row.speedup_vs_sequential:.4f}",
                    first,
                    str(row.matched_prefix_tokens),
                    str(row.oracle_tokens),
                    f"{row.normalized_fidelity:.6f}",
                    "YES" if row.pareto else "NO",
                ]
            )
        )
    print("TOKEN_SOURCE\tretokenized_output")


def write_csv(path: Path, rows: Sequence[MetricRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "prompt", "arm", "confidence", "generation_tps",
                "speedup_vs_sequential", "first_diff_token",
                "first_diff_token_numeric", "matched_prefix_tokens",
                "oracle_tokens", "normalized_fidelity",
                "exact_through_horizon", "pareto", "token_source",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.prompt,
                    row.arm,
                    row.confidence,
                    f"{row.generation_tps:.6f}",
                    f"{row.speedup_vs_sequential:.6f}",
                    "NONE" if row.first_diff_token is None else row.first_diff_token,
                    row.first_diff_token_numeric,
                    row.matched_prefix_tokens,
                    row.oracle_tokens,
                    f"{row.normalized_fidelity:.9f}",
                    int(row.exact_through_horizon),
                    int(row.pareto),
                    "retokenized_output",
                ]
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path, help="benchmark JSON config")
    p.add_argument("--csv", type=Path, help="optional CSV output path")
    p.add_argument("--timeout", type=float, default=None, help="per-process timeout in seconds")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config_dir = config_path.parent
    config = load_config(config_path)

    prompts = config.get("prompts")
    arms = config.get("arms")
    if not isinstance(prompts, list) or not prompts:
        fail("config.prompts must be a non-empty array")
    if not isinstance(arms, list) or not arms:
        fail("config.arms must be a non-empty array")
    if sum(bool(a.get("oracle", False)) for a in arms if isinstance(a, dict)) != 1:
        fail("config.arms must contain exactly one oracle=true arm")

    all_rows: list[MetricRow] = []
    stack = TempStack()
    try:
        for prompt_obj in prompts:
            if not isinstance(prompt_obj, dict):
                fail("each prompt must be an object")
            prompt_name, prompt_args = prompt_arg(prompt_obj, config_dir, stack)
            runs: list[RunResult] = []
            for arm_obj in arms:
                if not isinstance(arm_obj, dict):
                    fail("each arm must be an object")
                run = run_arm(
                    config, arm_obj, prompt_name, prompt_args, config_dir, args.timeout
                )
                runs.append(run)
            all_rows.extend(evaluate_prompt(runs))
    finally:
        stack.cleanup()

    print_matrix(all_rows)
    if args.csv is not None:
        write_csv(args.csv.expanduser().resolve(), all_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
