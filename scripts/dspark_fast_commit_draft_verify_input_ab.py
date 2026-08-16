#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

TARGET_FN = "static int ds4_session_eval_dspark_speculative_argmax("
VERIFY_FN = "metal_graph_verify_suffix_tops("
VERIFY_PREFIX = "DS4_DSPARK_E10_VERIFY"
STATE_PREFIX = "DS4_DSPARK_STATE_RETURN"


def die(msg: str) -> None:
    raise SystemExit(f"dspark_fast_commit_draft_verify_input_ab: {msg}")


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
/* E10 diagnostic-only: capture the support-model draft block and the exact
 * target verifier window before the commit decision. Injected only into a
 * detached experiment worktree. */
static uint32_t ds4_e10_verify_ordinal = 0;

static bool ds4_e10_trace_enabled(void) {
    const char *env = getenv("DS4_DSPARK_E10_TRACE");
    return env && env[0] && strcmp(env, "0") != 0;
}

static uint64_t ds4_e10_fnv1a(const void *ptr, size_t n) {
    const unsigned char *p = (const unsigned char *)ptr;
    uint64_t h = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < n; i++) {
        h ^= (uint64_t)p[i];
        h *= UINT64_C(1099511628211);
    }
    return h;
}

static uint64_t ds4_e10_read_draft_value(
        const void *base, size_t elem_size, size_t idx) {
    uint64_t v = 0;
    if (!base || elem_size == 0) return 0;
    const unsigned char *p =
        (const unsigned char *)base + idx * elem_size;
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    const size_t n = elem_size < sizeof(v) ? elem_size : sizeof(v);
    memcpy(&v, p, n);
#else
    const size_t n = elem_size < sizeof(v) ? elem_size : sizeof(v);
    for (size_t j = 0; j < n; j++) {
        v = (v << 8) | (uint64_t)p[j];
    }
#endif
    return v;
}

static bool ds4_e10_verify_suffix_tops(
        int                     draft_n,
        const void             *drafts,
        size_t                  draft_elem_size,
        uint64_t                checkpoint_len,
        uint64_t                checkpoint_valid,
        uint64_t                mtp_draft_valid,
        uint64_t                dspark_draft_len,
        uint64_t                dspark_draft_valid,
        uint64_t                mtp_n_raw,
        uint64_t                dspark_cache_start,
        uint64_t                dspark_cache_token_start,
        uint64_t                dspark_cache_len,
        ds4_gpu_graph          *g,
        const ds4_model        *model,
        const ds4_weights      *weights,
        const token_vec        *prompt,
        uint32_t                start,
        uint32_t                n_tokens,
        bool                    capture_prefix1,
        bool                    capture_dspark_hidden,
        int                    *row_tops,
        float                  *row_logits,
        ds4_verify_suffix_timing *timing) {
    const bool ok = metal_graph_verify_suffix_tops(
        g, model, weights, prompt, start, n_tokens,
        capture_prefix1, capture_dspark_hidden,
        row_tops, row_logits, timing);
    if (!ds4_e10_trace_enabled()) return ok;

    const uint32_t ordinal1 = ++ds4_e10_verify_ordinal;
    const size_t draft_bytes =
        (draft_n > 0 && drafts && draft_elem_size > 0)
        ? (size_t)draft_n * draft_elem_size : 0;
    const uint64_t draft_hash =
        draft_bytes ? ds4_e10_fnv1a(drafts, draft_bytes) : 0;

    fprintf(stderr,
            "DS4_DSPARK_E10_VERIFY ordinal1=%u start=%u n_tokens=%u "
            "draft_n=%d draft_elem_size=%zu draft_hash=%016llx "
            "checkpoint_len=%llu checkpoint_valid=%llu "
            "mtp_draft_valid=%llu dspark_draft_len=%llu "
            "dspark_draft_valid=%llu mtp_n_raw=%llu "
            "dspark_cache_start=%llu dspark_cache_token_start=%llu "
            "dspark_cache_len=%llu drafts=",
            ordinal1, start, n_tokens, draft_n, draft_elem_size,
            (unsigned long long)draft_hash,
            (unsigned long long)checkpoint_len,
            (unsigned long long)checkpoint_valid,
            (unsigned long long)mtp_draft_valid,
            (unsigned long long)dspark_draft_len,
            (unsigned long long)dspark_draft_valid,
            (unsigned long long)mtp_n_raw,
            (unsigned long long)dspark_cache_start,
            (unsigned long long)dspark_cache_token_start,
            (unsigned long long)dspark_cache_len);
    for (int i = 0; i < draft_n; i++) {
        const uint64_t token = ds4_e10_read_draft_value(
            drafts, draft_elem_size, (size_t)i);
        fprintf(stderr, "%s%llu", i ? "," : "",
                (unsigned long long)token);
    }
    fprintf(stderr, " status=%s\n", ok ? "PASS" : "ERROR");
    fflush(stderr);
    return ok;
}
'''


def instrument(source: Path) -> None:
    text = source.read_text(encoding="utf-8")
    if VERIFY_PREFIX in text or "ds4_e10_verify_suffix_tops" in text:
        die(f"{source} already contains E10 instrumentation")
    fn_start = text.find(TARGET_FN)
    if fn_start < 0:
        die("DSpark speculative argmax function not found")
    open_pos = text.find("{", fn_start)
    fn_end = find_matching_brace(text, open_pos)
    if open_pos < 0 or fn_end < 0:
        die("could not isolate DSpark speculative argmax function")
    fn = text[fn_start:fn_end + 1]
    call_count = fn.count(VERIFY_FN)
    if call_count < 1:
        die("no verifier call found in DSpark speculative argmax")
    required = [
        "draft_n", "drafts", "s->mtp_draft_valid", "s->dspark_draft_len",
        "s->dspark_draft_valid", "s->graph.mtp_n_raw",
        "s->graph.dspark_cache_start", "s->graph.dspark_cache_token_start",
        "s->graph.dspark_cache_len", "s->checkpoint.len",
        "s->checkpoint_valid",
    ]
    missing = [x for x in required if x not in fn]
    if missing:
        die("missing source anchors: " + ",".join(missing))

    replacement = (
        "ds4_e10_verify_suffix_tops("
        "draft_n, drafts, sizeof(drafts[0]), "
        "(uint64_t)s->checkpoint.len, "
        "(uint64_t)s->checkpoint_valid, "
        "(uint64_t)s->mtp_draft_valid, "
        "(uint64_t)s->dspark_draft_len, "
        "(uint64_t)s->dspark_draft_valid, "
        "(uint64_t)s->graph.mtp_n_raw, "
        "(uint64_t)s->graph.dspark_cache_start, "
        "(uint64_t)s->graph.dspark_cache_token_start, "
        "(uint64_t)s->graph.dspark_cache_len, "
    )
    fn = fn.replace(VERIFY_FN, replacement)
    patched = text[:fn_start] + HELPERS + "\n" + fn + text[fn_end + 1:]
    source.write_text(patched, encoding="utf-8")
    print(
        "FAST_COMMIT_DRAFT_VERIFY_INSTRUMENT "
        f"verify_call_sites={call_count} result=PASS"
    )


@dataclass(frozen=True)
class Verify:
    ordinal1: int
    start: int
    n_tokens: int
    draft_n: int
    draft_elem_size: int
    draft_hash: str
    fields: dict[str, int]
    drafts: tuple[int, ...]
    status: str
    line_no: int


@dataclass
class Event:
    ordinal1: int
    path: str
    commit_type: str
    drafted: int
    accepted_drafts: int
    returned: int
    start: int
    fields: dict[str, int]
    ids: tuple[int, ...]
    line_no: int
    verify: Verify | None = None
    orphan_verify_ordinals: tuple[int, ...] = ()


@dataclass
class ParsedArm:
    verifies: list[Verify] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    trailing_orphans: tuple[int, ...] = ()


VERIFY_RE = re.compile(
    rf"^{VERIFY_PREFIX} ordinal1=(\d+) start=(\d+) n_tokens=(\d+) "
    r"draft_n=(\d+) draft_elem_size=(\d+) draft_hash=([0-9a-fA-F]+) "
    r"(.*?) drafts=([^ ]*) status=(\S+)$"
)
STATE_RE = re.compile(
    rf"^{STATE_PREFIX} path=(\S+) commit_type=(\S+) drafted=(\d+) "
    r"accepted_drafts=(\d+) returned=(\d+) (.*?) ids=(.*)$"
)


def parse_int_list(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(",") if x)


def parse_fields(text: str) -> dict[str, int]:
    return {
        k: int(v)
        for k, v in re.findall(r"([A-Za-z0-9_]+)=(-?\d+)", text)
    }


def parse_arm(log: Path) -> ParsedArm:
    arm = ParsedArm()
    pending: list[Verify] = []
    for line_no, raw in enumerate(
        log.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
    ):
        line = raw.strip()
        vm = VERIFY_RE.match(line)
        if vm:
            fields = parse_fields(vm.group(7))
            v = Verify(
                ordinal1=int(vm.group(1)),
                start=int(vm.group(2)),
                n_tokens=int(vm.group(3)),
                draft_n=int(vm.group(4)),
                draft_elem_size=int(vm.group(5)),
                draft_hash=vm.group(6).lower(),
                fields=fields,
                drafts=parse_int_list(vm.group(8)),
                status=vm.group(9),
                line_no=line_no,
            )
            arm.verifies.append(v)
            pending.append(v)
            continue
        sm = STATE_RE.match(line)
        if not sm:
            continue
        body_fields = parse_fields(sm.group(6))
        if "start" not in body_fields:
            die(f"state return missing absolute start in {log}:{line_no}")
        verify = pending[-1] if pending else None
        orphans = tuple(v.ordinal1 for v in pending[:-1])
        pending.clear()
        arm.events.append(
            Event(
                ordinal1=len(arm.events) + 1,
                path=sm.group(1),
                commit_type=sm.group(2),
                drafted=int(sm.group(3)),
                accepted_drafts=int(sm.group(4)),
                returned=int(sm.group(5)),
                start=body_fields["start"],
                fields=body_fields,
                ids=parse_int_list(sm.group(7)),
                line_no=line_no,
                verify=verify,
                orphan_verify_ordinals=orphans,
            )
        )
    arm.trailing_orphans = tuple(v.ordinal1 for v in pending)
    if not arm.verifies:
        die(f"no {VERIFY_PREFIX} lines found in {log}")
    if not arm.events:
        die(f"no {STATE_PREFIX} lines found in {log}")
    return arm


def event_map(arm: ParsedArm) -> dict[int, Event]:
    out: dict[int, Event] = {}
    for e in arm.events:
        if e.start in out:
            die(f"duplicate absolute start={e.start}; E10 needs a stronger key")
        out[e.start] = e
    return out


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
        capture_output=True, text=True, encoding="utf-8",
        errors="replace", check=False,
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


def fmt_idx(v: int | None) -> str:
    return "NONE" if v is None else str(v)


def fmt_ord(v: int | None) -> str:
    return "NONE" if v is None else str(v + 1)


def fmt_tps(v: float | None) -> str:
    return "NA" if v is None else f"{v:.3f}"


SUPPORT_FIELDS = (
    "mtp_draft_valid",
    "dspark_draft_len",
    "dspark_draft_valid",
    "mtp_n_raw",
    "dspark_cache_start",
    "dspark_cache_token_start",
    "dspark_cache_len",
)


def support_diffs(f: Verify, r: Verify) -> list[str]:
    return [
        name for name in SUPPORT_FIELDS
        if f.fields.get(name) != r.fields.get(name)
    ]


def analyze(args: argparse.Namespace) -> None:
    oracle_ids = tokenize(args.ds4, args.model, args.oracle)
    replay_ids = tokenize(args.ds4, args.model, args.replay_out)
    base_ids = tokenize(args.ds4, args.model, args.fast_base_out)
    trace_ids = tokenize(args.ds4, args.model, args.fast_trace_out)
    d_replay = first_divergence(oracle_ids, replay_ids)
    d_base = first_divergence(oracle_ids, base_ids)
    d_trace = first_divergence(oracle_ids, trace_ids)
    trace_vs_base = first_divergence(base_ids, trace_ids)
    baseline_ok = d_base is not None and d_base + 1 == args.expected_ordinal1
    replay_ok = d_replay is None
    trace_ok = trace_vs_base is None and d_trace == d_base

    fast = parse_arm(args.fast_trace_log)
    replay = parse_arm(args.replay_log)
    fast_map = event_map(fast)
    replay_map = event_map(replay)

    fast_orphans = (
        sum(len(e.orphan_verify_ordinals) for e in fast.events)
        + len(fast.trailing_orphans)
    )
    replay_orphans = (
        sum(len(e.orphan_verify_ordinals) for e in replay.events)
        + len(replay.trailing_orphans)
    )
    fast_unbound = sum(e.verify is None for e in fast.events)
    replay_unbound = sum(e.verify is None for e in replay.events)
    binding_ok = (
        fast_orphans == 0 and replay_orphans == 0
        and fast_unbound == 0 and replay_unbound == 0
        and all(v.status == "PASS" for v in fast.verifies + replay.verifies)
    )

    causal_fast = next(
        (e for e in fast.events if e.commit_type == "FULL_ACCEPT_FAST"), None
    )
    causal_replay = replay_map.get(causal_fast.start) if causal_fast else None
    causal_anchor_ok = (
        causal_fast is not None
        and causal_replay is not None
        and causal_replay.commit_type == "FULL_ACCEPT_REPLAY"
        and causal_fast.ids == causal_replay.ids
        and causal_fast.verify is not None
        and causal_replay.verify is not None
        and causal_fast.verify.drafts == causal_replay.verify.drafts
    )

    controls_ok = (
        baseline_ok and replay_ok and trace_ok and binding_ok and causal_anchor_ok
    )

    print(
        "FAST_COMMIT_DRAFT_VERIFY_BASELINE "
        f"first_diff_index0={fmt_idx(d_base)} first_diff_ordinal1={fmt_ord(d_base)} "
        f"expected_ordinal1={args.expected_ordinal1} "
        f"generation_tps={fmt_tps(extract_tps(args.fast_base_log))} "
        f"result={'PASS' if baseline_ok else 'FAIL'}"
    )
    print(
        "FAST_COMMIT_DRAFT_VERIFY_ORACLE_CONTROL "
        f"replay_first_diff_index0={fmt_idx(d_replay)} "
        f"replay_first_diff_ordinal1={fmt_ord(d_replay)} "
        f"replay_tps={fmt_tps(extract_tps(args.replay_log))} "
        f"result={'PASS' if replay_ok else 'FAIL'}"
    )
    print(
        "FAST_COMMIT_DRAFT_VERIFY_TRACE_CONTROL "
        f"token_output={'EXACT' if trace_vs_base is None else 'MISMATCH'} "
        f"frontier_same={1 if d_trace == d_base else 0} "
        f"result={'PASS' if trace_ok else 'FAIL'}"
    )
    print(
        "FAST_COMMIT_DRAFT_VERIFY_BINDING "
        f"fast_verifies={len(fast.verifies)} replay_verifies={len(replay.verifies)} "
        f"fast_events={len(fast.events)} replay_events={len(replay.events)} "
        f"fast_orphans={fast_orphans} replay_orphans={replay_orphans} "
        f"fast_unbound={fast_unbound} replay_unbound={replay_unbound} "
        f"result={'PASS' if binding_ok else 'FAIL'}"
    )
    if causal_fast is None:
        print("FAST_COMMIT_DRAFT_VERIFY_CAUSAL_ANCHOR result=FAIL reason=no_fast_full_accept")
    else:
        print(
            "FAST_COMMIT_DRAFT_VERIFY_CAUSAL_ANCHOR "
            f"start={causal_fast.start} "
            f"fast_event_ordinal1={causal_fast.ordinal1} "
            f"replay_event_ordinal1={causal_replay.ordinal1 if causal_replay else 'MISSING'} "
            f"fast_draft_n={causal_fast.verify.draft_n if causal_fast.verify else 'MISSING'} "
            f"replay_draft_n={causal_replay.verify.draft_n if causal_replay and causal_replay.verify else 'MISSING'} "
            f"returned_ids_equal={1 if causal_replay and causal_fast.ids == causal_replay.ids else 0} "
            f"draft_ids_equal={1 if causal_replay and causal_fast.verify and causal_replay.verify and causal_fast.verify.drafts == causal_replay.verify.drafts else 0} "
            f"result={'PASS' if causal_anchor_ok else 'FAIL'}"
        )

    fast_starts = set(fast_map)
    replay_starts = set(replay_map)
    all_starts = sorted(fast_starts | replay_starts)
    first_schedule: tuple[int, str] | None = None
    for start in all_starts:
        if start not in fast_map or start not in replay_map:
            first_schedule = (
                start,
                "FAST_ONLY" if start in fast_map else "REPLAY_ONLY",
            )
            break

    first_draft: int | None = None
    first_shape: int | None = None
    first_support: tuple[int, list[str]] | None = None
    compared = 0

    for start in sorted(fast_starts & replay_starts):
        fe = fast_map[start]
        revent = replay_map[start]
        if fe.verify is None or revent.verify is None:
            continue
        fv = fe.verify
        rv = revent.verify
        compared += 1
        draft_same = (
            fv.draft_n == rv.draft_n
            and fv.drafts == rv.drafts
            and fv.draft_hash == rv.draft_hash
        )
        shape_same = fv.start == rv.start and fv.n_tokens == rv.n_tokens
        sdiff = support_diffs(fv, rv)
        if first_draft is None and not draft_same:
            first_draft = start
        if first_shape is None and not shape_same:
            first_shape = start
        if first_support is None and sdiff:
            first_support = (start, sdiff)

        if (
            start == (causal_fast.start if causal_fast else -1)
            or not draft_same or not shape_same or sdiff
        ):
            print(
                "FAST_COMMIT_DRAFT_VERIFY_ROW "
                f"start={start} "
                f"fast_event_ordinal1={fe.ordinal1} "
                f"replay_event_ordinal1={revent.ordinal1} "
                f"fast_verify_start={fv.start} replay_verify_start={rv.start} "
                f"fast_n_tokens={fv.n_tokens} replay_n_tokens={rv.n_tokens} "
                f"verify_shape_result={'EXACT' if shape_same else 'MISMATCH'} "
                f"fast_draft_n={fv.draft_n} replay_draft_n={rv.draft_n} "
                f"draft_result={'EXACT' if draft_same else 'MISMATCH'} "
                f"fast_drafts={','.join(map(str, fv.drafts)) or 'NONE'} "
                f"replay_drafts={','.join(map(str, rv.drafts)) or 'NONE'} "
                f"support_result={'EXACT' if not sdiff else 'MISMATCH'} "
                f"support_fields={','.join(sdiff) if sdiff else 'NONE'}"
            )

    if first_schedule is None:
        print("FAST_COMMIT_DRAFT_VERIFY_FIRST_SCHEDULE_DIVERGENCE=NONE")
    else:
        print(
            "FAST_COMMIT_DRAFT_VERIFY_FIRST_SCHEDULE_DIVERGENCE "
            f"start={first_schedule[0]} presence={first_schedule[1]}"
        )
    print(
        "FAST_COMMIT_DRAFT_VERIFY_FIRST_DRAFT_DIVERGENCE="
        + ("NONE" if first_draft is None else f"start={first_draft}")
    )
    print(
        "FAST_COMMIT_DRAFT_VERIFY_FIRST_VERIFY_SHAPE_DIVERGENCE="
        + ("NONE" if first_shape is None else f"start={first_shape}")
    )
    print(
        "FAST_COMMIT_DRAFT_VERIFY_FIRST_SUPPORT_STATE_DIVERGENCE="
        + (
            "NONE"
            if first_support is None
            else f"start={first_support[0]} fields={','.join(first_support[1])}"
        )
    )

    if not controls_ok:
        source = "INCONCLUSIVE_CONTROL"
        next_step = "ADJUDICATE_E10_CONTROL"
    else:
        candidates = [
            x for x in (
                first_schedule[0] if first_schedule else None,
                first_draft,
                first_shape,
            )
            if x is not None
        ]
        first_any = min(candidates) if candidates else None
        if (
            first_schedule is not None
            and first_schedule[0] == first_any
            and (first_draft is None or first_schedule[0] < first_draft)
            and (first_shape is None or first_schedule[0] < first_shape)
        ):
            source = "SPECULATIVE_CALL_SCHEDULE_DIVERGENCE"
            next_step = "DSPARK_SCHEDULER_AND_DRAFT_VALID_AB"
        elif first_draft is not None and first_draft == first_any:
            if first_support is not None and first_support[0] <= first_draft:
                source = "SUPPORT_STATE_OR_DRAFT_OUTPUT_DIVERGENCE"
                next_step = "SUPPORT_MODEL_STATE_AND_STAGE_OUTPUT_AB"
            else:
                source = "SUPPORT_DRAFT_BLOCK_DIVERGENCE"
                next_step = "SUPPORT_MODEL_STAGE_OUTPUT_AB"
        elif first_shape is not None:
            source = "VERIFY_INPUT_ASSEMBLY_DIVERGENCE"
            next_step = "VERIFY_WINDOW_ASSEMBLY_AB"
        else:
            source = "DRAFT_AND_VERIFY_INPUT_EXACT"
            next_step = "REVISIT_DECISION_OR_MAPPING"
    print(f"FAST_COMMIT_DRAFT_VERIFY_COMPARED_EVENTS={compared}")
    print(f"FAST_COMMIT_DRAFT_VERIFY_SOURCE={source}")
    print(f"FAST_COMMIT_DRAFT_VERIFY_NEXT={next_step}")


def main() -> None:
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    pi = sp.add_parser("instrument")
    pi.add_argument("--source", required=True, type=Path)

    pa = sp.add_parser("analyze")
    pa.add_argument("--ds4", required=True, type=Path)
    pa.add_argument("--model", required=True, type=Path)
    pa.add_argument("--oracle", required=True, type=Path)
    pa.add_argument("--replay-out", required=True, type=Path)
    pa.add_argument("--replay-log", required=True, type=Path)
    pa.add_argument("--fast-base-out", required=True, type=Path)
    pa.add_argument("--fast-base-log", required=True, type=Path)
    pa.add_argument("--fast-trace-out", required=True, type=Path)
    pa.add_argument("--fast-trace-log", required=True, type=Path)
    pa.add_argument("--expected-ordinal1", type=int, default=27)
    args = p.parse_args()
    if args.cmd == "instrument":
        instrument(args.source)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
