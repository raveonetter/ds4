#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Sequence


def die(msg: str) -> None:
    raise SystemExit(f"dspark_fast_commit_known_content_combo_ab: {msg}")


def parse_token_ids(text: str) -> list[int]:
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("[") and s.endswith("]"):
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                obj = None
            if isinstance(obj, list) and all(isinstance(x, int) for x in obj):
                return obj
        break
    return []


def tokenize(ds4: Path, model: Path, text_file: Path) -> list[int]:
    proc = subprocess.run(
        [
            str(ds4), "-m", str(model), "--raw-prompt", "--dump-tokens",
            "--prompt-file", str(text_file),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        die(f"tokenizer failed for {text_file}: {proc.stderr.strip()}")
    for stream in (proc.stdout, proc.stderr):
        ids = parse_token_ids(stream)
        if ids:
            return ids
    die(f"no token IDs parsed for {text_file}")


def first_divergence(a: Sequence[int], b: Sequence[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def extract_tps(log: Path) -> float | None:
    vals = re.findall(
        r"([0-9]+(?:\.[0-9]+)?)\s*t/s\b",
        log.read_text(encoding="utf-8", errors="replace"),
    )
    return float(vals[-1]) if vals else None


def e6_gate_hits(log: Path) -> tuple[int, int]:
    text = log.read_text(encoding="utf-8", errors="replace")
    compressor = len(re.findall(
        r"^DS4_DSPARK_E6_PAIR_GATE site=COMPRESSOR$", text, re.MULTILINE
    ))
    indexer = len(re.findall(
        r"^DS4_DSPARK_E6_PAIR_GATE site=INDEXER$", text, re.MULTILINE
    ))
    return compressor, indexer


def fmt_index(v: int | None) -> str:
    return "NONE" if v is None else str(v)


def fmt_ordinal(v: int | None) -> str:
    return "NONE" if v is None else str(v + 1)


def fmt_tps(v: float | None) -> str:
    return "NA" if v is None else f"{v:.3f}"


def pct_delta(new: float | None, old: float | None) -> str:
    if new is None or old is None or old == 0:
        return "NA"
    return f"{((new / old) - 1.0) * 100.0:.3f}"


def moved(candidate: int | None, baseline: int | None) -> bool:
    return baseline is not None and (candidate is None or candidate > baseline)


def analyze(args: argparse.Namespace) -> None:
    oracle_ids = tokenize(args.ds4, args.model, args.oracle)
    arms = {
        "replay": (args.replay_out, args.replay_log),
        "baseline": (args.fast_out, args.fast_log),
        "kv": (args.kv_out, args.kv_log),
        "pair": (args.pair_out, args.pair_log),
        "combo": (args.combo_out, args.combo_log),
    }
    ids: dict[str, list[int]] = {}
    diffs: dict[str, int | None] = {}
    tps: dict[str, float | None] = {}
    for name, (out, log) in arms.items():
        ids[name] = tokenize(args.ds4, args.model, out)
        diffs[name] = first_divergence(oracle_ids, ids[name])
        tps[name] = extract_tps(log)

    baseline_ok = (
        diffs["baseline"] is not None
        and diffs["baseline"] + 1 == args.expected_ordinal1
    )
    replay_ok = diffs["replay"] is None
    pair_c, pair_i = e6_gate_hits(args.pair_log)
    combo_c, combo_i = e6_gate_hits(args.combo_log)
    pair_gate_ok = pair_c > 0 and pair_i > 0
    combo_gate_ok = combo_c > 0 and combo_i > 0

    print(
        "FAST_COMMIT_KNOWN_CONTENT_COMBO_BASELINE "
        f"first_diff_index0={fmt_index(diffs['baseline'])} "
        f"first_diff_ordinal1={fmt_ordinal(diffs['baseline'])} "
        f"expected_ordinal1={args.expected_ordinal1} "
        f"generation_tps={fmt_tps(tps['baseline'])} "
        f"result={'PASS' if baseline_ok else 'FAIL'}"
    )
    print(
        "FAST_COMMIT_KNOWN_CONTENT_COMBO_ORACLE_CONTROL "
        f"replay_first_diff_index0={fmt_index(diffs['replay'])} "
        f"replay_first_diff_ordinal1={fmt_ordinal(diffs['replay'])} "
        f"replay_tps={fmt_tps(tps['replay'])} "
        f"result={'PASS' if replay_ok else 'FAIL'}"
    )
    print(
        "FAST_COMMIT_KNOWN_CONTENT_COMBO_GATE arm=PAIR "
        f"compressor_hits={pair_c} indexer_hits={pair_i} "
        f"result={'PASS' if pair_gate_ok else 'FAIL'}"
    )
    print(
        "FAST_COMMIT_KNOWN_CONTENT_COMBO_GATE arm=COMBO "
        f"compressor_hits={combo_c} indexer_hits={combo_i} "
        f"result={'PASS' if combo_gate_ok else 'FAIL'}"
    )

    labels = {
        "kv": "FAST_COMMIT_KNOWN_CONTENT_KV_FRONTIER",
        "pair": "FAST_COMMIT_KNOWN_CONTENT_PAIR_FRONTIER",
        "combo": "FAST_COMMIT_KNOWN_CONTENT_COMBO_FRONTIER",
    }
    for name in ("kv", "pair", "combo"):
        print(
            f"{labels[name]} "
            f"baseline_first_diff_index0={fmt_index(diffs['baseline'])} "
            f"baseline_first_diff_ordinal1={fmt_ordinal(diffs['baseline'])} "
            f"repair_first_diff_index0={fmt_index(diffs[name])} "
            f"repair_first_diff_ordinal1={fmt_ordinal(diffs[name])} "
            f"baseline_tps={fmt_tps(tps['baseline'])} repair_tps={fmt_tps(tps[name])} "
            f"repair_vs_baseline_tps_delta_pct={pct_delta(tps[name], tps['baseline'])}"
        )

    controls_ok = baseline_ok and replay_ok and pair_gate_ok and combo_gate_ok
    if not controls_ok:
        result = "INCONCLUSIVE"
        next_step = "ADJUDICATE_CONTROL_OR_GATE"
    else:
        kv_moved = moved(diffs["kv"], diffs["baseline"])
        pair_moved = moved(diffs["pair"], diffs["baseline"])
        combo_moved = moved(diffs["combo"], diffs["baseline"])
        if combo_moved and not kv_moved and not pair_moved:
            result = "INTERACTION_PROVEN"
            next_step = "WINDOWED_COMBINED_REPAIR_MINIMIZATION"
        elif combo_moved:
            result = "COMBINED_EFFECT_PROVEN"
            next_step = "MINIMIZE_DOMINANT_AND_INTERACTION_TERMS"
        else:
            result = "INTERACTION_REJECTED"
            next_step = "NEXT_ITERATION_VERIFY_LOGITS_AB"

    print(f"FAST_COMMIT_KNOWN_CONTENT_INTERACTION={result}")
    print(f"FAST_COMMIT_KNOWN_CONTENT_NEXT={next_step}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ds4", required=True, type=Path)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--oracle", required=True, type=Path)
    p.add_argument("--replay-out", required=True, type=Path)
    p.add_argument("--replay-log", required=True, type=Path)
    p.add_argument("--fast-out", required=True, type=Path)
    p.add_argument("--fast-log", required=True, type=Path)
    p.add_argument("--kv-out", required=True, type=Path)
    p.add_argument("--kv-log", required=True, type=Path)
    p.add_argument("--pair-out", required=True, type=Path)
    p.add_argument("--pair-log", required=True, type=Path)
    p.add_argument("--combo-out", required=True, type=Path)
    p.add_argument("--combo-log", required=True, type=Path)
    p.add_argument("--expected-ordinal1", type=int, default=27)
    analyze(p.parse_args())


if __name__ == "__main__":
    main()
