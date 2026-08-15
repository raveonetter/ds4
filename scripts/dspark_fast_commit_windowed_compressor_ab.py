#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

FAMILY = "DS4_REPAIR_FAMILY_F16_BATCH_EXT_VS_SINGLE_PAIR_MV"
EVENT_PREFIX = "DS4_DSPARK_CONTENT_EVENT"
BATCH_RE = re.compile(
    r"^DS4_DSPARK_E7_BATCH pos0=(\d+) n_tokens=(\d+)$"
)
EVENT_RE = re.compile(
    rf"^{EVENT_PREFIX} path=(\S+) commit_type=(\S+) drafted=(\d+) "
    r"accepted_drafts=(\d+) returned=(\d+) start=(\d+) checkpoint_len=(\d+) ids=(.*)$"
)
RETURN_RE = re.compile(
    r"^DS4_DSPARK_TRACE_RETURN path=(\S+) commit_type=(\S+) drafted=(\d+) "
    r"accepted_drafts=(\d+) returned=(\d+) correction=(\d+) ids=(.*)$"
)
GATE_RE = re.compile(
    r"^DS4_DSPARK_E7_GATE part=(PROJECTION|REFRESH) pos0=(\d+) n_tokens=(\d+)$"
)


def die(msg: str) -> None:
    raise SystemExit(f"dspark_fast_commit_windowed_compressor_ab: {msg}")


def _find_matching_paren(text: str, open_pos: int) -> int:
    depth = 0
    in_string = False
    escaped = False
    for i in range(open_pos, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def instrument(source: Path) -> None:
    text = source.read_text(encoding="utf-8")
    sentinel = "ds4_e7_window_projection_gate"
    if sentinel in text:
        die(f"source already contains {sentinel}")

    marker = "static bool metal_graph_refresh_ratio4_compressor_state("
    marker_pos = text.find(marker)
    if marker_pos < 0:
        die("ratio-4 compressor refresh helper not found")

    helper = r'''/* E7 diagnostic-only causal-window compressor gate.
 * Injected only into the detached test worktree. */
static bool ds4_e7_parse_u32_env(const char *name, uint32_t *out) {
    const char *s = getenv(name);
    char *end = NULL;
    unsigned long v;
    if (!s || !s[0] || !out) return false;
    v = strtoul(s, &end, 10);
    if (end == s || *end != '\0' || v > UINT32_MAX) return false;
    *out = (uint32_t)v;
    return true;
}

static bool ds4_e7_window_match(uint32_t pos0, uint32_t n_tokens) {
    uint32_t target_pos0 = 0;
    uint32_t target_n = 0;
    return ds4_e7_parse_u32_env("DS4_DSPARK_E7_TARGET_POS0", &target_pos0) &&
           ds4_e7_parse_u32_env("DS4_DSPARK_E7_TARGET_N_TOKENS", &target_n) &&
           pos0 == target_pos0 && n_tokens == target_n;
}

static bool ds4_e7_trace_enabled(void) {
    const char *env = getenv("DS4_DSPARK_E7_TRACE");
    return env && env[0] && strcmp(env, "0") != 0;
}

static void ds4_e7_scout_batch(uint32_t pos0, uint32_t n_tokens) {
    static bool have_last = false;
    static uint32_t last_pos0 = 0;
    static uint32_t last_n = 0;
    if (!ds4_e7_trace_enabled()) return;
    if (have_last && last_pos0 == pos0 && last_n == n_tokens) return;
    have_last = true;
    last_pos0 = pos0;
    last_n = n_tokens;
    fprintf(stderr, "DS4_DSPARK_E7_BATCH pos0=%u n_tokens=%u\n",
            pos0, n_tokens);
}

static bool ds4_e7_window_part_enabled(
        const char *part, uint32_t pos0, uint32_t n_tokens) {
    const char *mode = getenv("DS4_DSPARK_E7_COMPRESSOR_REPAIR");
    bool enabled = false;
    if (!part) return false;
    if (strcmp(part, "PROJECTION") == 0) ds4_e7_scout_batch(pos0, n_tokens);
    if (!mode || !mode[0] || !ds4_e7_window_match(pos0, n_tokens)) return false;
    enabled = strcasecmp(mode, "BOTH") == 0 || strcasecmp(mode, part) == 0;
    if (enabled && ds4_e7_trace_enabled()) {
        fprintf(stderr,
                "DS4_DSPARK_E7_GATE part=%s pos0=%u n_tokens=%u\n",
                part, pos0, n_tokens);
    }
    return enabled;
}

static bool ds4_e7_window_projection_gate(uint32_t pos0, uint32_t n_tokens) {
    return ds4_e7_window_part_enabled("PROJECTION", pos0, n_tokens);
}

static bool ds4_e7_window_refresh_gate(uint32_t pos0, uint32_t n_tokens) {
    return ds4_e7_window_part_enabled("REFRESH", pos0, n_tokens);
}

'''
    text = text[:marker_pos] + helper + text[marker_pos:]
    marker_pos = text.find(marker)

    old_sig_tail = "uint32_t          n_tokens) {"
    sig_end = text.find(old_sig_tail, marker_pos, marker_pos + 1400)
    if sig_end < 0:
        die("ratio-4 refresh signature tail not found")
    new_sig_tail = (
        "uint32_t          n_tokens,\n"
        "        const char       *e7_site) {"
    )
    text = text[:sig_end] + new_sig_tail + text[sig_end + len(old_sig_tail):]

    call_token = "metal_graph_refresh_ratio4_compressor_state("
    search_from = marker_pos + len(marker)
    calls: list[tuple[int, int, str]] = []
    pos = search_from
    while True:
        call_pos = text.find(call_token, pos)
        if call_pos < 0:
            break
        open_pos = call_pos + len(call_token) - 1
        close_pos = _find_matching_paren(text, open_pos)
        if close_pos < 0:
            die("unterminated ratio-4 refresh call")
        body = text[open_pos + 1:close_pos]
        has_attn = "layer->attn_compressor_kv" in body
        has_index = "layer->indexer_compressor_kv" in body
        if has_attn == has_index:
            die("could not classify ratio-4 refresh caller")
        site = "COMPRESSOR" if has_attn else "INDEXER"
        calls.append((call_pos, close_pos, site))
        pos = close_pos + 1

    if len(calls) != 4:
        die(f"expected 4 ratio-4 refresh calls, found {len(calls)}")
    if {site for _, _, site in calls} != {"COMPRESSOR", "INDEXER"}:
        die(f"unexpected ratio-4 refresh call classification: {[x[2] for x in calls]}")
    for _call_pos, close_pos, site in reversed(calls):
        text = text[:close_pos] + f', "{site}"' + text[close_pos:]

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
        "            (strcmp(e7_site, \"COMPRESSOR\") == 0 &&\n"
        "             ds4_e7_window_refresh_gate(pos0, n_tokens))) {"
    )
    text = text[:dm.start()] + direct_new + text[dm.end():]

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
        tail = text[m.end():m.end() + 2400]
        attn_pos = tail.find("layer->attn_compressor_kv->abs_offset")
        index_pos = tail.find("layer->indexer_compressor_kv->abs_offset")
        if attn_pos >= 0 and (index_pos < 0 or attn_pos < index_pos):
            site = "COMPRESSOR"
        elif index_pos >= 0:
            site = "INDEXER"
        else:
            die("could not classify a layer-major Family-7 pair gate")
        seen.add(site)
        if site == "COMPRESSOR":
            replacement = (
                "repair_pair = ds4_family_repair_runtime_enabled(\n"
                f"                    {FAMILY}) ||\n"
                "                ds4_e7_window_projection_gate(pos0, n_tokens);"
            )
        else:
            replacement = m.group(0)
        replacements.append((m.start(), m.end(), replacement, site))

    if seen != {"COMPRESSOR", "INDEXER"}:
        die(f"unexpected projection gate classification: {sorted(seen)}")
    for start, end, replacement, _site in reversed(replacements):
        text = text[:start] + replacement + text[end:]

    source.write_text(text, encoding="utf-8")
    print(
        "FAST_COMMIT_WINDOWED_COMPRESSOR_INSTRUMENT "
        "projection_gate=1 refresh_gate=1 "
        f"refresh_calls={len(calls)} result=PASS"
    )


@dataclass(frozen=True)
class Window:
    pos0: int
    n_tokens: int
    event_start: int
    drafted: int
    accepted_drafts: int
    ids: tuple[int, ...]


def _parse_ids(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(",") if x)


def select_window(log: Path, output_json: Path) -> None:
    last_batch: tuple[int, int] | None = None
    batches_seen = 0
    target: Window | None = None
    for raw in log.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        bm = BATCH_RE.match(line)
        if bm:
            last_batch = (int(bm.group(1)), int(bm.group(2)))
            batches_seen += 1
            continue
        em = EVENT_RE.match(line)
        rm = RETURN_RE.match(line)
        if em and em.group(2) == "FULL_ACCEPT_FAST":
            if last_batch is None:
                die("FULL_ACCEPT_FAST observed before any compressor batch scout record")
            target = Window(
                pos0=last_batch[0],
                n_tokens=last_batch[1],
                event_start=int(em.group(6)),
                drafted=int(em.group(3)),
                accepted_drafts=int(em.group(4)),
                ids=_parse_ids(em.group(8)),
            )
            break
        if rm and rm.group(2) == "FULL_ACCEPT_FAST":
            if last_batch is None:
                die("FULL_ACCEPT_FAST observed before any compressor batch scout record")
            target = Window(
                pos0=last_batch[0],
                n_tokens=last_batch[1],
                event_start=-1,
                drafted=int(rm.group(3)),
                accepted_drafts=int(rm.group(4)),
                ids=_parse_ids(rm.group(7)),
            )
            break
    if target is None:
        die("no FULL_ACCEPT_FAST event found in scout log")
    output_json.write_text(
        json.dumps(
            {
                "pos0": target.pos0,
                "n_tokens": target.n_tokens,
                "event_start": target.event_start,
                "drafted": target.drafted,
                "accepted_drafts": target.accepted_drafts,
                "ids": list(target.ids),
                "batches_seen": batches_seen,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        "FAST_COMMIT_WINDOWED_COMPRESSOR_WINDOW "
        f"pos0={target.pos0} n_tokens={target.n_tokens} "
        f"event_start={'NA' if target.event_start < 0 else target.event_start} drafted={target.drafted} "
        f"accepted_drafts={target.accepted_drafts} "
        f"batches_seen_before_event={batches_seen} result=PASS"
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


def gate_hits(log: Path, window: Window) -> tuple[int, int, int]:
    projection = 0
    refresh = 0
    wrong_window = 0
    for raw in log.read_text(encoding="utf-8", errors="replace").splitlines():
        gm = GATE_RE.match(raw.strip())
        if not gm:
            continue
        if int(gm.group(2)) != window.pos0 or int(gm.group(3)) != window.n_tokens:
            wrong_window += 1
        if gm.group(1) == "PROJECTION":
            projection += 1
        else:
            refresh += 1
    return projection, refresh, wrong_window


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


def load_window(path: Path) -> Window:
    obj = json.loads(path.read_text(encoding="utf-8"))
    return Window(
        pos0=int(obj["pos0"]),
        n_tokens=int(obj["n_tokens"]),
        event_start=int(obj["event_start"]),
        drafted=int(obj["drafted"]),
        accepted_drafts=int(obj["accepted_drafts"]),
        ids=tuple(int(x) for x in obj["ids"]),
    )


def analyze(args: argparse.Namespace) -> None:
    window = load_window(args.window_json)
    oracle_ids = tokenize(args.ds4, args.model, args.oracle)
    outputs = {
        "replay": (args.replay_out, args.replay_log),
        "baseline": (args.fast_out, args.fast_log),
        "scout": (args.scout_out, args.scout_log),
        "projection": (args.projection_out, args.projection_log),
        "refresh": (args.refresh_out, args.refresh_log),
        "both": (args.both_out, args.both_log),
    }
    ids: dict[str, list[int]] = {}
    diffs: dict[str, int | None] = {}
    tps: dict[str, float | None] = {}
    for name, (out, log) in outputs.items():
        ids[name] = tokenize(args.ds4, args.model, out)
        diffs[name] = first_divergence(oracle_ids, ids[name])
        tps[name] = extract_tps(log)

    baseline_ok = (
        diffs["baseline"] is not None
        and diffs["baseline"] + 1 == args.expected_ordinal1
    )
    replay_ok = diffs["replay"] is None
    trace_output_exact = ids["baseline"] == ids["scout"]
    trace_frontier_same = diffs["baseline"] == diffs["scout"]

    print(
        "FAST_COMMIT_WINDOWED_COMPRESSOR_BASELINE "
        f"first_diff_index0={fmt_index(diffs['baseline'])} "
        f"first_diff_ordinal1={fmt_ordinal(diffs['baseline'])} "
        f"expected_ordinal1={args.expected_ordinal1} "
        f"generation_tps={fmt_tps(tps['baseline'])} "
        f"result={'PASS' if baseline_ok else 'FAIL'}"
    )
    print(
        "FAST_COMMIT_WINDOWED_COMPRESSOR_ORACLE_CONTROL "
        f"replay_first_diff_index0={fmt_index(diffs['replay'])} "
        f"replay_first_diff_ordinal1={fmt_ordinal(diffs['replay'])} "
        f"replay_tps={fmt_tps(tps['replay'])} "
        f"result={'PASS' if replay_ok else 'FAIL'}"
    )
    print(
        "FAST_COMMIT_WINDOWED_COMPRESSOR_TRACE_CONTROL "
        f"token_output={'EXACT' if trace_output_exact else 'MISMATCH'} "
        f"frontier_same={1 if trace_frontier_same else 0} "
        f"baseline_tps={fmt_tps(tps['baseline'])} scout_tps={fmt_tps(tps['scout'])} "
        f"result={'PASS' if trace_output_exact and trace_frontier_same else 'FAIL'}"
    )

    expected_hits = {
        "projection": lambda p, r: p > 0 and r == 0,
        "refresh": lambda p, r: p == 0 and r > 0,
        "both": lambda p, r: p > 0 and r > 0,
    }
    gate_ok: dict[str, bool] = {}
    early_ok: dict[str, bool] = {}
    for name in ("projection", "refresh", "both"):
        p_hits, r_hits, wrong = gate_hits(outputs[name][1], window)
        gate_ok[name] = expected_hits[name](p_hits, r_hits) and wrong == 0
        vs_baseline = first_divergence(ids["baseline"], ids[name])
        early_ok[name] = vs_baseline is None or (
            diffs["baseline"] is not None and vs_baseline >= diffs["baseline"]
        )
        print(
            "FAST_COMMIT_WINDOWED_COMPRESSOR_GATE "
            f"arm={name.upper()} projection_hits={p_hits} refresh_hits={r_hits} "
            f"wrong_window_hits={wrong} target_pos0={window.pos0} "
            f"target_n_tokens={window.n_tokens} "
            f"result={'PASS' if gate_ok[name] else 'FAIL'}"
        )
        print(
            "FAST_COMMIT_WINDOWED_COMPRESSOR_PRE27_CONTROL "
            f"arm={name.upper()} repair_vs_baseline_first_diff_index0={fmt_index(vs_baseline)} "
            f"repair_vs_baseline_first_diff_ordinal1={fmt_ordinal(vs_baseline)} "
            f"result={'PASS' if early_ok[name] else 'FAIL'}"
        )

    labels = {
        "projection": "FAST_COMMIT_WINDOWED_COMPRESSOR_PROJECTION_FRONTIER",
        "refresh": "FAST_COMMIT_WINDOWED_COMPRESSOR_REFRESH_FRONTIER",
        "both": "FAST_COMMIT_WINDOWED_COMPRESSOR_BOTH_FRONTIER",
    }
    for name in ("projection", "refresh", "both"):
        print(
            f"{labels[name]} "
            f"baseline_first_diff_index0={fmt_index(diffs['baseline'])} "
            f"baseline_first_diff_ordinal1={fmt_ordinal(diffs['baseline'])} "
            f"repair_first_diff_index0={fmt_index(diffs[name])} "
            f"repair_first_diff_ordinal1={fmt_ordinal(diffs[name])} "
            f"baseline_tps={fmt_tps(tps['baseline'])} repair_tps={fmt_tps(tps[name])} "
            f"repair_vs_baseline_tps_delta_pct={pct_delta(tps[name], tps['baseline'])}"
        )

    controls_ok = (
        baseline_ok
        and replay_ok
        and trace_output_exact
        and trace_frontier_same
        and all(gate_ok.values())
        and all(early_ok.values())
    )
    if not controls_ok:
        source = "INCONCLUSIVE"
        next_step = "ADJUDICATE_WINDOW_OR_CONTROL"
    else:
        p = moved(diffs["projection"], diffs["baseline"])
        r = moved(diffs["refresh"], diffs["baseline"])
        b = moved(diffs["both"], diffs["baseline"])
        if (p or r) and not b:
            source = "INCONCLUSIVE"
            next_step = "ADJUDICATE_PROJECTION_REFRESH_COMPOSITION"
        elif p and not r:
            source = "PROJECTION"
            next_step = "PROJECTION_MINIMAL_PRODUCTION_REPAIR"
        elif r and not p:
            source = "REFRESH"
            next_step = "REFRESH_MINIMAL_PRODUCTION_REPAIR"
        elif p and r:
            source = "BOTH_INDEPENDENT"
            next_step = "BENCHMARK_MINIMAL_CHOICES"
        elif b:
            source = "INTERACTION"
            next_step = "PROJECTION_REFRESH_INTERACTION_SPLIT"
        else:
            source = "NOT_PROVEN"
            next_step = "DOWNSTREAM_FAST_COMMIT_CONTENT_AB"
    print(f"FAST_COMMIT_WINDOWED_COMPRESSOR_SOURCE={source}")
    print(f"FAST_COMMIT_WINDOWED_COMPRESSOR_NEXT={next_step}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    ip = sub.add_parser("instrument")
    ip.add_argument("--source", type=Path, required=True)

    sp = sub.add_parser("select-window")
    sp.add_argument("--log", type=Path, required=True)
    sp.add_argument("--output-json", type=Path, required=True)

    ap = sub.add_parser("analyze")
    ap.add_argument("--ds4", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--oracle", type=Path, required=True)
    ap.add_argument("--window-json", type=Path, required=True)
    ap.add_argument("--replay-out", type=Path, required=True)
    ap.add_argument("--replay-log", type=Path, required=True)
    ap.add_argument("--fast-out", type=Path, required=True)
    ap.add_argument("--fast-log", type=Path, required=True)
    ap.add_argument("--scout-out", type=Path, required=True)
    ap.add_argument("--scout-log", type=Path, required=True)
    ap.add_argument("--projection-out", type=Path, required=True)
    ap.add_argument("--projection-log", type=Path, required=True)
    ap.add_argument("--refresh-out", type=Path, required=True)
    ap.add_argument("--refresh-log", type=Path, required=True)
    ap.add_argument("--both-out", type=Path, required=True)
    ap.add_argument("--both-log", type=Path, required=True)
    ap.add_argument("--expected-ordinal1", type=int, default=27)

    args = p.parse_args()
    if args.cmd == "instrument":
        instrument(args.source)
    elif args.cmd == "select-window":
        select_window(args.log, args.output_json)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
