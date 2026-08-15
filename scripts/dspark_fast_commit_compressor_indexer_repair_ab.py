#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Sequence

FAMILY = "DS4_REPAIR_FAMILY_F16_BATCH_EXT_VS_SINGLE_PAIR_MV"


def die(msg: str) -> None:
    raise SystemExit(f"dspark_fast_commit_compressor_indexer_repair_ab: {msg}")


def instrument(source: Path) -> None:
    text = source.read_text(encoding="utf-8")
    sentinel = "ds4_e6_pair_scope_enabled"
    if sentinel in text:
        die(f"source already contains {sentinel}")

    marker = "static bool metal_graph_refresh_ratio4_compressor_state("
    marker_pos = text.find(marker)
    if marker_pos < 0:
        die("ratio-4 compressor refresh helper not found")

    helper = r'''/* E6 diagnostic-only selective Family-7 gate.  The committed production
 * source is not modified: the harness injects this into a detached worktree. */
static bool ds4_e6_pair_scope_enabled(const char *site) {
    const char *env = getenv("DS4_DSPARK_E6_PAIR_REPAIR");
    if (!env || !env[0] || !site) return false;
    return strcasecmp(env, "BOTH") == 0 || strcasecmp(env, site) == 0;
}

'''
    text = text[:marker_pos] + helper + text[marker_pos:]

    # The ratio-4 refresh is shared by attention compressor and indexer. head_dim
    # identifies the caller without changing the production function signature.
    marker_pos = text.find(marker)
    direct_re = re.compile(
        r"if \(ds4_family_repair_runtime_enabled\(\s*"
        + re.escape(FAMILY)
        + r"\s*\)\) \{"
    )
    dm = direct_re.search(text, marker_pos)
    if dm is None:
        die("Family-7 gate inside ratio-4 refresh helper not found")
    direct_new = (
        "if (ds4_family_repair_runtime_enabled(\n"
        f"                {FAMILY}) ||\n"
        "            ds4_e6_pair_scope_enabled(\n"
        "                head_dim == DS4_N_INDEXER_HEAD_DIM ? \"INDEXER\" : \"COMPRESSOR\")) {"
    )
    text = text[: dm.start()] + direct_new + text[dm.end() :]

    # There are two layer-major batch projection gates: attention compressor and
    # indexer. Classify each from the immediately following weight reference and
    # add a diagnostic site override while preserving the normal Family-7 switch.
    assign_re = re.compile(
        r"repair_pair\s*=\s*ds4_family_repair_runtime_enabled\(\s*"
        + re.escape(FAMILY)
        + r"\s*\);"
    )
    matches = list(assign_re.finditer(text))
    if len(matches) != 2:
        die(f"expected 2 layer-major Family-7 pair gates, found {len(matches)}")

    replacements: list[tuple[int, int, str, str]] = []
    seen: set[str] = set()
    for m in matches:
        tail = text[m.end() : m.end() + 2200]
        attn_pos = tail.find("layer->attn_compressor_kv->abs_offset")
        index_pos = tail.find("layer->indexer_compressor_kv->abs_offset")
        if attn_pos >= 0 and (index_pos < 0 or attn_pos < index_pos):
            site = "COMPRESSOR"
        elif index_pos >= 0:
            site = "INDEXER"
        else:
            die("could not classify a layer-major Family-7 pair gate")
        seen.add(site)
        replacement = (
            "repair_pair = ds4_family_repair_runtime_enabled(\n"
            f"                    {FAMILY}) ||\n"
            f"                ds4_e6_pair_scope_enabled(\"{site}\");"
        )
        replacements.append((m.start(), m.end(), replacement, site))

    if seen != {"COMPRESSOR", "INDEXER"}:
        die(f"unexpected Family-7 gate classification: {sorted(seen)}")
    for start, end, replacement, _site in reversed(replacements):
        text = text[:start] + replacement + text[end:]

    source.write_text(text, encoding="utf-8")
    print(
        "FAST_COMMIT_COMPRESSOR_INDEXER_INSTRUMENT "
        "ratio4_gate=1 compressor_projection_gate=1 indexer_projection_gate=1 result=PASS"
    )


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
    if baseline is None:
        return False
    return candidate is None or candidate > baseline


def analyze(args: argparse.Namespace) -> None:
    oracle_ids = tokenize(args.ds4, args.model, args.oracle)
    paths = {
        "replay": (args.replay_out, args.replay_log),
        "baseline": (args.fast_out, args.fast_log),
        "compressor": (args.compressor_out, args.compressor_log),
        "indexer": (args.indexer_out, args.indexer_log),
        "both": (args.both_out, args.both_log),
    }
    diffs: dict[str, int | None] = {}
    tps: dict[str, float | None] = {}
    for name, (out, log) in paths.items():
        diffs[name] = first_divergence(oracle_ids, tokenize(args.ds4, args.model, out))
        tps[name] = extract_tps(log)

    baseline_ok = (
        diffs["baseline"] is not None
        and diffs["baseline"] + 1 == args.expected_ordinal1
    )
    replay_ok = diffs["replay"] is None
    print(
        "FAST_COMMIT_COMPRESSOR_INDEXER_BASELINE "
        f"first_diff_index0={fmt_index(diffs['baseline'])} "
        f"first_diff_ordinal1={fmt_ordinal(diffs['baseline'])} "
        f"expected_ordinal1={args.expected_ordinal1} "
        f"generation_tps={fmt_tps(tps['baseline'])} "
        f"result={'PASS' if baseline_ok else 'FAIL'}"
    )
    print(
        "FAST_COMMIT_COMPRESSOR_INDEXER_ORACLE_CONTROL "
        f"replay_first_diff_index0={fmt_index(diffs['replay'])} "
        f"replay_first_diff_ordinal1={fmt_ordinal(diffs['replay'])} "
        f"replay_tps={fmt_tps(tps['replay'])} "
        f"result={'PASS' if replay_ok else 'FAIL'}"
    )

    for name, label in (
        ("compressor", "FAST_COMMIT_COMPRESSOR_REPAIR_FRONTIER"),
        ("indexer", "FAST_COMMIT_INDEXER_REPAIR_FRONTIER"),
        ("both", "FAST_COMMIT_COMPRESSOR_INDEXER_REPAIR_FRONTIER"),
    ):
        print(
            f"{label} "
            f"baseline_first_diff_index0={fmt_index(diffs['baseline'])} "
            f"baseline_first_diff_ordinal1={fmt_ordinal(diffs['baseline'])} "
            f"repair_first_diff_index0={fmt_index(diffs[name])} "
            f"repair_first_diff_ordinal1={fmt_ordinal(diffs[name])} "
            f"baseline_tps={fmt_tps(tps['baseline'])} repair_tps={fmt_tps(tps[name])} "
            f"repair_vs_baseline_tps_delta_pct={pct_delta(tps[name], tps['baseline'])}"
        )

    if not baseline_ok or not replay_ok:
        source = "INCONCLUSIVE"
        next_step = "ADJUDICATE_CONTROL"
    else:
        c = moved(diffs["compressor"], diffs["baseline"])
        i = moved(diffs["indexer"], diffs["baseline"])
        b = moved(diffs["both"], diffs["baseline"])
        if c and not i:
            source = "COMPRESSOR"
            next_step = "COMPRESSOR_MINIMAL_REPAIR"
        elif i and not c:
            source = "INDEXER"
            next_step = "INDEXER_MINIMAL_REPAIR"
        elif c and i:
            source = "BOTH_INDEPENDENT"
            next_step = "MINIMAL_CAUSAL_SITE_SPLIT"
        elif b:
            source = "INTERACTION"
            next_step = "COMPRESSOR_INDEXER_INTERACTION_LOCALIZATION"
        else:
            source = "NOT_PROVEN"
            next_step = "DOWNSTREAM_FAST_COMMIT_CONTENT_AB"
    print(f"FAST_COMMIT_COMPRESSOR_INDEXER_SOURCE={source}")
    print(f"FAST_COMMIT_COMPRESSOR_INDEXER_NEXT={next_step}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    ip = sub.add_parser("instrument")
    ip.add_argument("--source", type=Path, required=True)

    ap = sub.add_parser("analyze")
    ap.add_argument("--ds4", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--oracle", type=Path, required=True)
    ap.add_argument("--replay-out", type=Path, required=True)
    ap.add_argument("--replay-log", type=Path, required=True)
    ap.add_argument("--fast-out", type=Path, required=True)
    ap.add_argument("--fast-log", type=Path, required=True)
    ap.add_argument("--compressor-out", type=Path, required=True)
    ap.add_argument("--compressor-log", type=Path, required=True)
    ap.add_argument("--indexer-out", type=Path, required=True)
    ap.add_argument("--indexer-log", type=Path, required=True)
    ap.add_argument("--both-out", type=Path, required=True)
    ap.add_argument("--both-log", type=Path, required=True)
    ap.add_argument("--expected-ordinal1", type=int, default=27)

    args = p.parse_args()
    if args.cmd == "instrument":
        instrument(args.source)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
