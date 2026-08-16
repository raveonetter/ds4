#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

TARGET_FN = "static int ds4_session_eval_dspark_speculative_argmax("
VERIFY_FN = "metal_graph_verify_suffix_tops("
ENV_NAME = "DS4_DSPARK_E9_TRACE_LOGITS"
ROW_PREFIX = "DS4_DSPARK_E9_LOGITS"
TRACE_PREFIX = "DS4_DSPARK_TRACE_RETURN"


def die(msg: str) -> None:
    raise SystemExit(f"dspark_fast_commit_next_verify_logits_ab: {msg}")


def find_matching_brace(text: str, open_pos: int) -> int:
    depth = 0
    state = "code"
    i = open_pos
    while i < len(text):
        c = text[i]
        n = text[i + 1] if i + 1 < len(text) else ""
        if state == "code":
            if c == '"':
                state = "string"
            elif c == "'":
                state = "char"
            elif c == "/" and n == "/":
                state = "line_comment"
                i += 1
            elif c == "/" and n == "*":
                state = "block_comment"
                i += 1
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return i
        elif state == "string":
            if c == "\\":
                i += 1
            elif c == '"':
                state = "code"
        elif state == "char":
            if c == "\\":
                i += 1
            elif c == "'":
                state = "code"
        elif state == "line_comment":
            if c == "\n":
                state = "code"
        elif state == "block_comment":
            if c == "*" and n == "/":
                state = "code"
                i += 1
        i += 1
    return -1


HELPERS = r'''
/* E9 diagnostic-only: read back every verifier logits row after the batched
 * verification call. Injected only into the detached experiment worktree. */
static uint32_t ds4_e9_verify_ordinal = 0;

static bool ds4_e9_trace_logits_enabled(void) {
    const char *env = getenv("DS4_DSPARK_E9_TRACE_LOGITS");
    return env && env[0] && strcmp(env, "0") != 0;
}

static uint64_t ds4_e9_fnv1a(const void *ptr, size_t n) {
    const unsigned char *p = (const unsigned char *)ptr;
    uint64_t h = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < n; i++) {
        h ^= (uint64_t)p[i];
        h *= UINT64_C(1099511628211);
    }
    return h;
}

static void ds4_e9_emit_logits_row(
        uint32_t verify_ordinal1,
        uint32_t start,
        uint32_t n_tokens,
        uint32_t row,
        int verify_top_id,
        const float *logits) {
    if (!logits) return;
    uint32_t top1 = 0;
    uint32_t top2 = 0;
    float top1_v = logits[0];
    float top2_v = logits[0];
    if (DS4_N_VOCAB > 1) {
        if (logits[1] > logits[0]) {
            top1 = 1;
            top2 = 0;
            top1_v = logits[1];
            top2_v = logits[0];
        } else {
            top1 = 0;
            top2 = 1;
            top1_v = logits[0];
            top2_v = logits[1];
        }
        for (uint32_t i = 2; i < DS4_N_VOCAB; i++) {
            const float v = logits[i];
            if (v > top1_v) {
                top2 = top1;
                top2_v = top1_v;
                top1 = i;
                top1_v = v;
            } else if (v > top2_v) {
                top2 = i;
                top2_v = v;
            }
        }
    }
    const uint64_t hash = ds4_e9_fnv1a(
        logits, (size_t)DS4_N_VOCAB * sizeof(logits[0]));
    fprintf(stderr,
            "DS4_DSPARK_E9_LOGITS verify_ordinal1=%u start=%u n_tokens=%u row=%u "
            "hash=%016llx verify_top_id=%d top1_id=%u top1=%.9g "
            "top2_id=%u top2=%.9g margin=%.17g status=PASS\n",
            verify_ordinal1,
            start,
            n_tokens,
            row,
            (unsigned long long)hash,
            verify_top_id,
            top1,
            (double)top1_v,
            top2,
            (double)top2_v,
            (double)top1_v - (double)top2_v);
}

static bool ds4_e9_verify_suffix_tops(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        const token_vec       *prompt,
        uint32_t               start,
        uint32_t               n_tokens,
        bool                   capture_prefix1,
        bool                   capture_dspark_hidden,
        int                   *row_tops,
        float                 *row_logits,
        ds4_verify_suffix_timing *timing) {
    const bool ok = metal_graph_verify_suffix_tops(
        g, model, weights, prompt, start, n_tokens,
        capture_prefix1, capture_dspark_hidden,
        row_tops, row_logits, timing);
    if (!ok || !ds4_e9_trace_logits_enabled()) return ok;

    const uint32_t verify_ordinal1 = ++ds4_e9_verify_ordinal;
    float *tmp = xmalloc((size_t)DS4_N_VOCAB * sizeof(tmp[0]));
    for (uint32_t row = 0; row < n_tokens; row++) {
        const bool row_ok = metal_graph_read_spec_logits_row(g, row, tmp);
        if (!row_ok) {
            fprintf(stderr,
                    "DS4_DSPARK_E9_LOGITS verify_ordinal1=%u start=%u n_tokens=%u "
                    "row=%u hash=0000000000000000 verify_top_id=%d top1_id=0 "
                    "top1=0 top2_id=0 top2=0 margin=0 status=ERROR\n",
                    verify_ordinal1,
                    start,
                    n_tokens,
                    row,
                    row_tops ? row_tops[row] : -1);
            continue;
        }
        ds4_e9_emit_logits_row(
            verify_ordinal1,
            start,
            n_tokens,
            row,
            row_tops ? row_tops[row] : -1,
            tmp);
    }
    free(tmp);
    fflush(stderr);
    return ok;
}
'''


def instrument(source: Path) -> None:
    text = source.read_text(encoding="utf-8")
    if ROW_PREFIX in text or "ds4_e9_verify_suffix_tops" in text:
        die(f"{source} already contains E9 instrumentation")

    fn_start = text.find(TARGET_FN)
    if fn_start < 0:
        die("DSpark speculative argmax function not found")
    open_pos = text.find("{", fn_start)
    if open_pos < 0:
        die("DSpark speculative argmax opening brace not found")
    fn_end = find_matching_brace(text, open_pos)
    if fn_end < 0:
        die("DSpark speculative argmax closing brace not found")

    fn = text[fn_start:fn_end + 1]
    call_count = fn.count(VERIFY_FN)
    if call_count < 1:
        die("no verifier call found in DSpark speculative argmax")
    fn = fn.replace(VERIFY_FN, "ds4_e9_verify_suffix_tops(")
    patched = text[:fn_start] + HELPERS + "\n" + fn + text[fn_end + 1:]
    source.write_text(patched, encoding="utf-8")
    print(
        "FAST_COMMIT_VERIFY_LOGITS_INSTRUMENT "
        f"verify_call_sites={call_count} result=PASS"
    )


@dataclass(frozen=True)
class Row:
    verify_ordinal1: int
    start: int
    n_tokens: int
    row: int
    hash_hex: str
    verify_top_id: int
    top1_id: int
    top1: float
    top2_id: int
    top2: float
    margin: float
    status: str


@dataclass(frozen=True)
class Event:
    event_index0: int
    path: str
    commit_type: str
    drafted: int
    accepted_drafts: int
    returned: int
    correction: bool
    ids: tuple[int, ...]


ROW_RE = re.compile(
    rf"^{ROW_PREFIX} verify_ordinal1=(\d+) start=(\d+) n_tokens=(\d+) row=(\d+) "
    r"hash=([0-9a-fA-F]+) verify_top_id=(-?\d+) top1_id=(\d+) top1=([^\s]+) "
    r"top2_id=(\d+) top2=([^\s]+) margin=([^\s]+) status=(\S+)$"
)
TRACE_RE = re.compile(
    rf"^{TRACE_PREFIX} path=(\S+) commit_type=(\S+) drafted=(\d+) "
    r"accepted_drafts=(\d+) returned=(\d+) correction=([01]) ids=(.*)$"
)


def parse_rows(log: Path) -> dict[tuple[int, int], Row]:
    rows: dict[tuple[int, int], Row] = {}
    for raw in log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = ROW_RE.match(raw.strip())
        if not m:
            continue
        row = Row(
            verify_ordinal1=int(m.group(1)),
            start=int(m.group(2)),
            n_tokens=int(m.group(3)),
            row=int(m.group(4)),
            hash_hex=m.group(5).lower(),
            verify_top_id=int(m.group(6)),
            top1_id=int(m.group(7)),
            top1=float(m.group(8)),
            top2_id=int(m.group(9)),
            top2=float(m.group(10)),
            margin=float(m.group(11)),
            status=m.group(12),
        )
        key = (row.verify_ordinal1, row.row)
        if key in rows:
            die(f"duplicate E9 row key {key} in {log}")
        rows[key] = row
    if not rows:
        die(f"no {ROW_PREFIX} rows found in {log}")
    return rows


def parse_events(log: Path) -> list[Event]:
    events: list[Event] = []
    for raw in log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = TRACE_RE.match(raw.strip())
        if not m:
            continue
        ids = tuple(int(x) for x in m.group(7).split(",") if x)
        returned = int(m.group(5))
        if len(ids) != returned:
            die(f"trace returned={returned} but ids={len(ids)} in {log}")
        events.append(
            Event(
                event_index0=len(events),
                path=m.group(1),
                commit_type=m.group(2),
                drafted=int(m.group(3)),
                accepted_drafts=int(m.group(4)),
                returned=returned,
                correction=m.group(6) == "1",
                ids=ids,
            )
        )
    if not events:
        die(f"no {TRACE_PREFIX} events found in {log}")
    return events


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


def event_for_verify(events: list[Event], verify_ordinal1: int) -> Event | None:
    idx = verify_ordinal1 - 1
    return events[idx] if 0 <= idx < len(events) else None


def analyze(args: argparse.Namespace) -> None:
    oracle_ids = tokenize(args.ds4, args.model, args.oracle)
    replay_ids = tokenize(args.ds4, args.model, args.replay_out)
    base_ids = tokenize(args.ds4, args.model, args.fast_base_out)
    trace_ids = tokenize(args.ds4, args.model, args.fast_trace_out)

    d_replay = first_divergence(oracle_ids, replay_ids)
    d_base = first_divergence(oracle_ids, base_ids)
    d_trace = first_divergence(oracle_ids, trace_ids)
    trace_vs_base = first_divergence(base_ids, trace_ids)
    t_replay = extract_tps(args.replay_log)
    t_base = extract_tps(args.fast_base_log)
    t_trace = extract_tps(args.fast_trace_log)

    baseline_ok = d_base is not None and d_base + 1 == args.expected_ordinal1
    replay_ok = d_replay is None
    trace_ok = trace_vs_base is None and d_trace == d_base

    fast = parse_rows(args.fast_trace_log)
    replay = parse_rows(args.replay_log)
    fast_events = parse_events(args.fast_trace_log)
    replay_events = parse_events(args.replay_log)

    fast_verify_count = max(k[0] for k in fast)
    replay_verify_count = max(k[0] for k in replay)
    alignment_ok = (
        fast_verify_count == len(fast_events)
        and replay_verify_count == len(replay_events)
        and all(r.status == "PASS" for r in fast.values())
        and all(r.status == "PASS" for r in replay.values())
    )

    self_top_mismatch = sum(
        1 for r in list(fast.values()) + list(replay.values())
        if r.verify_top_id >= 0 and r.verify_top_id != r.top1_id
    )

    causal_idx = next(
        (e.event_index0 for e in fast_events if e.commit_type == "FULL_ACCEPT_FAST"),
        None,
    )
    causal_verify = None if causal_idx is None else causal_idx + 1
    next_verify = None if causal_idx is None else causal_idx + 2

    print(
        "FAST_COMMIT_VERIFY_LOGITS_BASELINE "
        f"first_diff_index0={fmt_index(d_base)} first_diff_ordinal1={fmt_ordinal(d_base)} "
        f"expected_ordinal1={args.expected_ordinal1} generation_tps={fmt_tps(t_base)} "
        f"result={'PASS' if baseline_ok else 'FAIL'}"
    )
    print(
        "FAST_COMMIT_VERIFY_LOGITS_ORACLE_CONTROL "
        f"replay_first_diff_index0={fmt_index(d_replay)} "
        f"replay_first_diff_ordinal1={fmt_ordinal(d_replay)} "
        f"replay_tps={fmt_tps(t_replay)} result={'PASS' if replay_ok else 'FAIL'}"
    )
    print(
        "FAST_COMMIT_VERIFY_LOGITS_TRACE_CONTROL "
        f"token_output={'EXACT' if trace_vs_base is None else 'MISMATCH'} "
        f"frontier_same={1 if d_trace == d_base else 0} "
        f"baseline_tps={fmt_tps(t_base)} trace_tps={fmt_tps(t_trace)} "
        f"trace_tps_delta_pct={pct_delta(t_trace, t_base)} "
        f"result={'PASS' if trace_ok else 'FAIL'}"
    )
    print(
        "FAST_COMMIT_VERIFY_LOGITS_TRACE "
        f"fast_rows={len(fast)} replay_rows={len(replay)} "
        f"fast_verifies={fast_verify_count} replay_verifies={replay_verify_count} "
        f"fast_events={len(fast_events)} replay_events={len(replay_events)} "
        f"verify_top_self_mismatches={self_top_mismatch} "
        f"iter_event_alignment={'PASS' if alignment_ok else 'FAIL'}"
    )
    print(
        "FAST_COMMIT_VERIFY_LOGITS_CAUSAL_WINDOW "
        f"full_accept_event_index0={fmt_index(causal_idx)} "
        f"full_accept_verify_ordinal1={fmt_index(causal_verify)} "
        f"next_verify_ordinal1={fmt_index(next_verify)} "
        f"result={'PASS' if causal_idx is not None else 'FAIL'}"
    )

    common = sorted(set(fast) & set(replay))
    first_hash: tuple[int, int] | None = None
    first_argmax: tuple[int, int] | None = None
    precommit_divergence = False
    shape_mismatch = False

    for key in common:
        f = fast[key]
        r = replay[key]
        same_shape = f.start == r.start and f.n_tokens == r.n_tokens
        shape_mismatch |= not same_shape
        hash_same = f.hash_hex == r.hash_hex
        argmax_same = f.top1_id == r.top1_id
        if first_hash is None and not hash_same:
            first_hash = key
        if first_argmax is None and not argmax_same:
            first_argmax = key
        if causal_verify is not None and key[0] <= causal_verify and not hash_same:
            precommit_divergence = True

        if (
            key[0] in {causal_verify, next_verify}
            or not hash_same
            or not argmax_same
            or not same_shape
        ):
            print(
                "FAST_COMMIT_VERIFY_LOGITS_ROW "
                f"verify_ordinal1={key[0]} row={key[1]} "
                f"fast_start={f.start} replay_start={r.start} "
                f"fast_n_tokens={f.n_tokens} replay_n_tokens={r.n_tokens} "
                f"shape_result={'EXACT' if same_shape else 'MISMATCH'} "
                f"hash_result={'EXACT' if hash_same else 'MISMATCH'} "
                f"argmax_result={'EXACT' if argmax_same else 'MISMATCH'} "
                f"fast_hash={f.hash_hex} replay_hash={r.hash_hex} "
                f"fast_top1_id={f.top1_id} replay_top1_id={r.top1_id} "
                f"fast_verify_top_id={f.verify_top_id} replay_verify_top_id={r.verify_top_id} "
                f"fast_top1={f.top1:.9g} replay_top1={r.top1:.9g} "
                f"fast_top2_id={f.top2_id} replay_top2_id={r.top2_id} "
                f"fast_margin={f.margin:.17g} replay_margin={r.margin:.17g}"
            )

    if first_hash is None:
        print("FAST_COMMIT_VERIFY_LOGITS_FIRST_HASH_DIVERGENCE=NONE")
    else:
        print(
            "FAST_COMMIT_VERIFY_LOGITS_FIRST_HASH_DIVERGENCE "
            f"verify_ordinal1={first_hash[0]} row={first_hash[1]}"
        )
    if first_argmax is None:
        print("FAST_COMMIT_VERIFY_LOGITS_FIRST_ARGMAX_DIVERGENCE=NONE")
    else:
        f = fast[first_argmax]
        r = replay[first_argmax]
        print(
            "FAST_COMMIT_VERIFY_LOGITS_FIRST_ARGMAX_DIVERGENCE "
            f"verify_ordinal1={first_argmax[0]} row={first_argmax[1]} "
            f"fast_top1_id={f.top1_id} replay_top1_id={r.top1_id} "
            f"fast_margin={f.margin:.17g} replay_margin={r.margin:.17g}"
        )

    correction_match = False
    if first_argmax is not None:
        verify_ord, row_idx = first_argmax
        fe = event_for_verify(fast_events, verify_ord)
        revent = event_for_verify(replay_events, verify_ord)
        fast_corr = fe.ids[-1] if fe and fe.correction and fe.ids else None
        replay_corr = (
            revent.ids[-1] if revent and revent.correction and revent.ids else None
        )
        correction_match = (
            fe is not None
            and fe.correction
            and row_idx == fe.accepted_drafts
            and fast_corr == fast[first_argmax].top1_id
        )
        print(
            "FAST_COMMIT_VERIFY_LOGITS_CORRECTION "
            f"verify_ordinal1={verify_ord} row={row_idx} "
            f"fast_commit_type={fe.commit_type if fe else 'MISSING'} "
            f"fast_accepted_drafts={fe.accepted_drafts if fe else 'MISSING'} "
            f"fast_correction_id={fast_corr if fast_corr is not None else 'NONE'} "
            f"replay_commit_type={revent.commit_type if revent else 'MISSING'} "
            f"replay_accepted_drafts={revent.accepted_drafts if revent else 'MISSING'} "
            f"replay_correction_id={replay_corr if replay_corr is not None else 'NONE'} "
            f"correction_row_match={1 if correction_match else 0}"
        )

    controls_ok = (
        baseline_ok and replay_ok and trace_ok and alignment_ok
        and self_top_mismatch == 0 and causal_idx is not None
    )
    if not controls_ok:
        source = "INCONCLUSIVE_CONTROL"
        next_step = "ADJUDICATE_TRACE_OR_ALIGNMENT"
    elif precommit_divergence:
        source = "INCONCLUSIVE_PRECOMMIT_LOGIT_DIVERGENCE"
        next_step = "ADJUDICATE_VERIFIER_DETERMINISM"
    elif shape_mismatch and first_argmax is None:
        source = "VERIFY_INPUT_SHAPE_DIVERGENCE"
        next_step = "NEXT_ITERATION_DRAFT_SCHEDULE_AB"
    elif first_argmax is not None and correction_match:
        source = "NEXT_VERIFY_ARGMAX_CAUSE_LOCALIZED"
        next_step = "LAYERWISE_NEXT_ITERATION_CP_AB"
    elif first_argmax is not None:
        source = "NEXT_VERIFY_ARGMAX_DIVERGENCE_FOUND"
        next_step = "CORRECTION_ROW_MAPPING_ADJUDICATION"
    elif first_hash is not None:
        source = "NEXT_VERIFY_NUMERICAL_DIVERGENCE_ONLY"
        next_step = "CORRECTION_DECISION_LOGIC_AB"
    else:
        source = "VERIFY_LOGITS_EXACT"
        next_step = "NON_LOGIT_CORRECTION_STATE_AB"

    print(f"FAST_COMMIT_VERIFY_LOGITS_COMPARED_ROWS={len(common)}")
    print(f"FAST_COMMIT_VERIFY_LOGITS_SOURCE={source}")
    print(f"FAST_COMMIT_VERIFY_LOGITS_NEXT={next_step}")


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
    ap.add_argument("--fast-base-out", type=Path, required=True)
    ap.add_argument("--fast-base-log", type=Path, required=True)
    ap.add_argument("--fast-trace-out", type=Path, required=True)
    ap.add_argument("--fast-trace-log", type=Path, required=True)
    ap.add_argument("--expected-ordinal1", type=int, default=27)
    args = p.parse_args()
    if args.cmd == "instrument":
        instrument(args.source)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
