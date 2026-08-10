#!/usr/bin/env python3
"""DSpark speed/fidelity Pareto benchmark.

Primary objectives:
  1. generation_tps (maximize)
  2. first_diff_token vs sequential oracle (maximize; NONE = exact through horizon)

Prompts are supplied verbatim by config. Arms may vary CLI args, environment
variables, binaries, or models, so family repairs can be benchmarked as knobs.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

TPS_RE = re.compile(r"generation:\s*([0-9]+(?:\.[0-9]+)?)\s*tokens/s")


@dataclass
class Run:
    prompt: str
    arm: str
    confidence: str
    oracle: bool
    tps: float
    tokens: list[int]


@dataclass
class Row:
    prompt: str
    arm: str
    confidence: str
    tps: float
    speedup: float
    first_diff: int | None
    first_diff_numeric: int
    matched_prefix: int
    oracle_tokens: int
    fidelity: float
    exact: bool
    pareto: bool = False


def die(msg: str) -> None:
    raise SystemExit(f"dspark_pareto_bench: {msg}")


def str_list(v: Any, name: str) -> list[str]:
    if v is None:
        return []
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        die(f"{name} must be an array of strings")
    return list(v)


def str_env(v: Any, name: str) -> dict[str, str]:
    if v is None:
        return {}
    if not isinstance(v, dict) or not all(
        isinstance(k, str) and isinstance(x, str) for k, x in v.items()
    ):
        die(f"{name} must be an object of string:string pairs")
    return dict(v)


def resolve(v: str, base: Path) -> str:
    p = Path(v).expanduser()
    return str((p if p.is_absolute() else base / p).resolve())


def run_cmd(
    cmd: Sequence[str], env: dict[str, str], timeout: float | None
) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    merged.update(env)
    try:
        p = subprocess.run(
            list(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=merged,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        die(f"command failed: {cmd!r}: {exc}")
    if p.returncode != 0:
        sys.stderr.write(p.stderr)
        die(f"command returned {p.returncode}: {cmd!r}")
    return p


def parse_tps(stderr: str) -> float:
    m = TPS_RE.findall(stderr)
    if not m:
        die("generation throughput not found in stderr")
    return float(m[-1])


def parse_token_dump(text: str) -> list[int]:
    s = text.strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, list) and all(isinstance(x, int) for x in obj):
            return list(obj)
        if isinstance(obj, dict):
            for key in ("tokens", "token_ids", "ids"):
                v = obj.get(key)
                if isinstance(v, list) and all(isinstance(x, int) for x in v):
                    return list(v)
    except json.JSONDecodeError:
        pass

    out: list[int] = []
    patterns = [
        re.compile(r"^(-?\d+)$"),
        re.compile(
            r"^(?:token(?:_id)?|id)\s*[:=]\s*(-?\d+)(?:\s|$)", re.I
        ),
        re.compile(r"^token\[\d+\]\s*[:=]\s*(-?\d+)(?:\s|$)", re.I),
        re.compile(r"^\d+\s*(?::|\t)\s*(-?\d+)(?:\s|$)"),
    ]
    for raw in text.splitlines():
        line = raw.strip()
        for pat in patterns:
            m = pat.match(line)
            if m:
                out.append(int(m.group(1)))
                break
    if not out:
        die(
            "cannot parse --dump-tokens output; sample:\n"
            + "\n".join(text.splitlines()[:8])
        )
    return out


def retokenize(
    binary: str,
    model: str,
    text: str,
    args: list[str],
    env: dict[str, str],
    timeout: float | None,
) -> list[int]:
    cmd = [
        binary,
        "-m",
        model,
        "--raw-prompt",
        "--dump-tokens",
        "-p",
        text,
        *args,
    ]
    p = run_cmd(cmd, env, timeout)
    for stream in (p.stdout, p.stderr):
        try:
            return parse_token_dump(stream)
        except SystemExit:
            pass
    die("--dump-tokens produced no parseable token IDs")


def first_diff(a: Sequence[int], b: Sequence[int]) -> tuple[int | None, int]:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i + 1, i
    if len(a) != len(b):
        return n + 1, n
    return None, len(a)


def prompt_args(
    p: dict[str, Any], base: Path, temps: list[str]
) -> tuple[str, list[str]]:
    name = p.get("name")
    if not isinstance(name, str) or not name:
        die("every prompt needs name")
    if ("file" in p) == ("text" in p):
        die(f"prompt {name}: set exactly one of file/text")
    if "file" in p:
        if not isinstance(p["file"], str):
            die(f"prompt {name}: file must be string")
        return name, ["--prompt-file", resolve(p["file"], base)]
    if not isinstance(p["text"], str):
        die(f"prompt {name}: text must be string")
    f = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", suffix=".txt", delete=False
    )
    f.write(p["text"])
    f.close()
    temps.append(f.name)
    return name, ["--prompt-file", f.name]


def run_arm(
    cfg: dict[str, Any],
    arm: dict[str, Any],
    pname: str,
    pargs: list[str],
    base: Path,
    timeout: float | None,
) -> Run:
    name = arm.get("name")
    if not isinstance(name, str) or not name:
        die("every arm needs name")
    b = arm.get("binary", cfg.get("binary", "./ds4"))
    m = arm.get("model", cfg.get("model"))
    if not isinstance(b, str) or not isinstance(m, str):
        die(f"arm {name}: binary/model missing")
    binary, model = resolve(b, base), resolve(m, base)

    cmd = [binary, "-m", model, *pargs]
    n = arm.get("tokens", cfg.get("tokens"))
    if n is not None:
        if not isinstance(n, int) or n <= 0:
            die("tokens must be positive integer")
        cmd += ["--tokens", str(n)]
    cmd += str_list(cfg.get("common_args"), "common_args")
    cmd += str_list(arm.get("args"), f"arm {name}.args")

    env = str_env(cfg.get("env"), "env")
    env.update(str_env(arm.get("env"), f"arm {name}.env"))
    p = run_cmd(cmd, env, timeout)
    tps = parse_tps(p.stderr)

    # Current CLI has no generated-token-ID trace. Re-tokenization happens
    # after timing with the same model/tokenizer and is explicitly labeled.
    tok_env = str_env(cfg.get("tokenizer_env"), "tokenizer_env")
    toks = retokenize(
        binary,
        model,
        p.stdout,
        str_list(cfg.get("tokenizer_args"), "tokenizer_args"),
        tok_env,
        timeout,
    )
    return Run(
        pname,
        name,
        str(arm.get("confidence", "NA")),
        bool(arm.get("oracle", False)),
        tps,
        toks,
    )


def rows_for_prompt(runs: list[Run]) -> list[Row]:
    oracle = [r for r in runs if r.oracle]
    if len(oracle) != 1:
        die(f"prompt {runs[0].prompt}: exactly one oracle arm required")
    o = oracle[0]
    horizon = len(o.tokens)
    if horizon == 0:
        die(f"prompt {o.prompt}: oracle produced zero tokens")
    rows: list[Row] = []
    for r in runs:
        d, matched = (None, horizon) if r.oracle else first_diff(o.tokens, r.tokens)
        numeric = horizon + 1 if d is None else d
        rows.append(
            Row(
                r.prompt,
                r.arm,
                r.confidence,
                r.tps,
                r.tps / o.tps,
                d,
                numeric,
                matched,
                horizon,
                min(matched, horizon) / horizon,
                d is None,
            )
        )
    for r in rows:
        r.pareto = not any(
            q is not r
            and q.tps >= r.tps
            and q.first_diff_numeric >= r.first_diff_numeric
            and (q.tps > r.tps or q.first_diff_numeric > r.first_diff_numeric)
            for q in rows
        )
    return rows


def print_rows(rows: list[Row]) -> None:
    print("PARETO_MATRIX")
    print(
        "prompt\tarm\tconfidence\tgeneration_tps\tspeedup_vs_sequential\t"
        "first_diff_token\tmatched_prefix_tokens\toracle_tokens\t"
        "normalized_fidelity\tpareto"
    )
    for r in rows:
        d = "NONE" if r.first_diff is None else str(r.first_diff)
        print(
            f"{r.prompt}\t{r.arm}\t{r.confidence}\t{r.tps:.2f}\t"
            f"{r.speedup:.4f}\t{d}\t{r.matched_prefix}\t{r.oracle_tokens}\t"
            f"{r.fidelity:.6f}\t{'YES' if r.pareto else 'NO'}"
        )
    print("TOKEN_SOURCE\tretokenized_output")


def write_csv(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "prompt",
                "arm",
                "confidence",
                "generation_tps",
                "speedup_vs_sequential",
                "first_diff_token",
                "first_diff_token_numeric",
                "matched_prefix_tokens",
                "oracle_tokens",
                "normalized_fidelity",
                "exact_through_horizon",
                "pareto",
                "token_source",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.prompt,
                    r.arm,
                    r.confidence,
                    f"{r.tps:.6f}",
                    f"{r.speedup:.6f}",
                    "NONE" if r.first_diff is None else r.first_diff,
                    r.first_diff_numeric,
                    r.matched_prefix,
                    r.oracle_tokens,
                    f"{r.fidelity:.9f}",
                    int(r.exact),
                    int(r.pareto),
                    "retokenized_output",
                ]
            )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--timeout", type=float)
    a = ap.parse_args()
    cp = a.config.expanduser().resolve()
    try:
        cfg = json.loads(cp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        die(f"cannot load config: {exc}")
    if not isinstance(cfg, dict):
        die("config must be object")
    prompts, arms = cfg.get("prompts"), cfg.get("arms")
    if (
        not isinstance(prompts, list)
        or not prompts
        or not isinstance(arms, list)
        or not arms
    ):
        die("prompts and arms must be non-empty arrays")
    if sum(bool(x.get("oracle")) for x in arms if isinstance(x, dict)) != 1:
        die("arms must contain exactly one oracle=true")

    rows: list[Row] = []
    temps: list[str] = []
    try:
        for p in prompts:
            if not isinstance(p, dict):
                die("prompt entries must be objects")
            pname, pargs = prompt_args(p, cp.parent, temps)
            runs = [
                run_arm(cfg, arm, pname, pargs, cp.parent, a.timeout)
                for arm in arms
                if isinstance(arm, dict)
            ]
            rows.extend(rows_for_prompt(runs))
    finally:
        for t in temps:
            try:
                os.unlink(t)
            except FileNotFoundError:
                pass
    print_rows(rows)
    if a.csv:
        write_csv(a.csv.expanduser().resolve(), rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
