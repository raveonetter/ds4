#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import dspark_fast_commit_draft_verify_input_ab as e10

SCHED_FN = "static bool ds4_session_dspark_scheduler_should_run("
SPEC_FN = "static int ds4_session_eval_dspark_speculative_argmax("
SCHED_CALL = "ds4_session_dspark_scheduler_should_run(s)"
STATE_ANCHOR = "const bool dspark_state_ok ="
STATE_PREFIX = "DS4_DSPARK_E11_STATE"
SCHED_PREFIX = "DS4_DSPARK_E11_SCHED"
SPEC_PREFIX = "DS4_DSPARK_E11_SPEC_ENTER"


def die(msg: str) -> None:
    raise SystemExit(f"dspark_fast_commit_scheduler_draft_valid_ab: {msg}")


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
                state = "line_comment"; i += 1
            elif c == "/" and n == "*":
                state = "block_comment"; i += 1
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
                state = "code"; i += 1
        i += 1
    return -1


WRAPPER = r'''
/* E11 diagnostic-only: expose the DSpark scheduler decision and the state
 * that feeds it. Injected only into a detached experiment worktree. */
static bool ds4_e11_scheduler_should_run(ds4_session *s) {
    const unsigned pre_cycles = s ? s->dspark_sched_cycles : 0;
    const unsigned pre_accepted = s ? s->dspark_sched_accepted : 0;
    const unsigned pre_no_draft = s ? s->dspark_sched_no_draft : 0;
    const unsigned pre_skip = s ? s->dspark_sched_skip : 0;
    const unsigned pre_lifetime =
        s ? s->dspark_sched_lifetime_accepted : 0;
    const bool decision = ds4_session_dspark_scheduler_should_run(s);
    const char *env = getenv("DS4_DSPARK_E11_TRACE");
    if (env && env[0] && strcmp(env, "0") != 0) {
        fprintf(stderr,
                "DS4_DSPARK_E11_SCHED decision=%s "
                "pre_cycles=%u post_cycles=%u "
                "pre_accepted=%u post_accepted=%u "
                "pre_no_draft=%u post_no_draft=%u "
                "pre_skip=%u post_skip=%u "
                "pre_lifetime=%u post_lifetime=%u\n",
                decision ? "RUN" : "SKIP",
                pre_cycles, s->dspark_sched_cycles,
                pre_accepted, s->dspark_sched_accepted,
                pre_no_draft, s->dspark_sched_no_draft,
                pre_skip, s->dspark_sched_skip,
                pre_lifetime, s->dspark_sched_lifetime_accepted);
        fflush(stderr);
    }
    return decision;
}
'''

STATE_TRACE = r'''
        {
            const char *ds4_e11_env = getenv("DS4_DSPARK_E11_TRACE");
            if (ds4_e11_env && ds4_e11_env[0] &&
                strcmp(ds4_e11_env, "0") != 0) {
                const int ds4_e11_checkpoint_last =
                    s->checkpoint.len > 0 ?
                    s->checkpoint.v[s->checkpoint.len - 1] : -1;
                fprintf(stderr,
                        "DS4_DSPARK_E11_STATE checkpoint_last=%d "
                        "checkpoint_len=%d state_ok=%d "
                        "checkpoint_valid=%d mtp_draft_valid=%d "
                        "dspark_draft_valid=%d dspark_draft_len=%u "
                        "mtp_n_raw=%u dspark_cache_start=%u "
                        "dspark_cache_token_start=%u dspark_cache_len=%u "
                        "cycles=%u accepted=%u no_draft=%u skip=%u "
                        "lifetime_accepted=%u\n",
                        ds4_e11_checkpoint_last, s->checkpoint.len,
                        (int)dspark_state_ok, (int)s->checkpoint_valid,
                        (int)s->mtp_draft_valid, (int)s->dspark_draft_valid,
                        s->dspark_draft_len, s->graph.mtp_n_raw,
                        s->graph.dspark_cache_start,
                        s->graph.dspark_cache_token_start,
                        s->graph.dspark_cache_len,
                        s->dspark_sched_cycles, s->dspark_sched_accepted,
                        s->dspark_sched_no_draft, s->dspark_sched_skip,
                        s->dspark_sched_lifetime_accepted);
                fflush(stderr);
            }
        }
'''

SPEC_TRACE = r'''
    {
        const char *ds4_e11_env = getenv("DS4_DSPARK_E11_TRACE");
        if (ds4_e11_env && ds4_e11_env[0] &&
            strcmp(ds4_e11_env, "0") != 0) {
            const int ds4_e11_checkpoint_last =
                s->checkpoint.len > 0 ?
                s->checkpoint.v[s->checkpoint.len - 1] : -1;
            fprintf(stderr,
                    "DS4_DSPARK_E11_SPEC_ENTER start=%u "
                    "checkpoint_last=%d checkpoint_len=%d\n",
                    (unsigned)start, ds4_e11_checkpoint_last,
                    s->checkpoint.len);
            fflush(stderr);
        }
    }
'''


def instrument(source: Path) -> None:
    text = source.read_text(encoding="utf-8")
    if STATE_PREFIX in text or "ds4_e11_scheduler_should_run" in text:
        die(f"{source} already contains E11 instrumentation")

    sched_start = text.find(SCHED_FN)
    if sched_start < 0:
        die("scheduler function not found")
    sched_open = text.find("{", sched_start)
    sched_end = find_matching_brace(text, sched_open)
    if sched_open < 0 or sched_end < 0:
        die("could not isolate scheduler function")

    call_positions = []
    p = 0
    while True:
        p = text.find(SCHED_CALL, p)
        if p < 0:
            break
        if not (sched_start <= p <= sched_end):
            call_positions.append(p)
        p += len(SCHED_CALL)
    if len(call_positions) != 1:
        die(f"expected exactly one scheduler call site, found {len(call_positions)}")

    call_pos = call_positions[0]
    text = text[:call_pos] + "ds4_e11_scheduler_should_run(s)" + text[call_pos + len(SCHED_CALL):]

    sched_start = text.find(SCHED_FN)
    sched_open = text.find("{", sched_start)
    sched_end = find_matching_brace(text, sched_open)
    text = text[:sched_end + 1] + "\n" + WRAPPER + text[sched_end + 1:]

    if text.count(STATE_ANCHOR) != 1:
        die(f"expected exactly one dspark_state_ok anchor, found {text.count(STATE_ANCHOR)}")
    state_pos = text.find(STATE_ANCHOR)
    state_semicolon = text.find(";", state_pos)
    if state_semicolon < 0:
        die("dspark_state_ok assignment terminator not found")
    text = text[:state_semicolon + 1] + STATE_TRACE + text[state_semicolon + 1:]

    spec_start = text.find(SPEC_FN)
    if spec_start < 0:
        die("speculative argmax function not found")
    spec_open = text.find("{", spec_start)
    spec_end = find_matching_brace(text, spec_open)
    if spec_open < 0 or spec_end < 0:
        die("could not isolate speculative argmax function")
    spec = text[spec_start:spec_end + 1]
    m = re.search(r"\b(?:uint32_t|size_t|int)\s+start\s*=", spec)
    if not m:
        die("absolute start declaration not found in speculative function")
    start_semicolon = spec.find(";", m.end())
    if start_semicolon < 0:
        die("start declaration terminator not found")
    spec = spec[:start_semicolon + 1] + SPEC_TRACE + spec[start_semicolon + 1:]
    text = text[:spec_start] + spec + text[spec_end + 1:]

    source.write_text(text, encoding="utf-8")
    print(
        "FAST_COMMIT_SCHEDULER_DRAFT_VALID_INSTRUMENT "
        "scheduler_call_sites=1 state_gate_sites=1 spec_entry_sites=1 result=PASS"
    )


@dataclass
class Gate:
    ordinal1: int
    line_no: int
    state: dict[str, int]
    decision: str | None = None
    sched: dict[str, int] | None = None
    spec_start: int | None = None


STATE_RE = re.compile(rf"^{STATE_PREFIX} (.*)$")
SCHED_RE = re.compile(rf"^{SCHED_PREFIX} decision=(RUN|SKIP) (.*)$")
SPEC_RE = re.compile(
    rf"^{SPEC_PREFIX} start=(\d+) checkpoint_last=(-?\d+) checkpoint_len=(\d+)$"
)


def parse_fields(text: str) -> dict[str, int]:
    return {k: int(v) for k, v in re.findall(r"([A-Za-z0-9_]+)=(-?\d+)", text)}


def parse_gates(log: Path) -> list[Gate]:
    gates: list[Gate] = []
    pending_run: list[Gate] = []
    for line_no, raw in enumerate(
        log.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
    ):
        line = raw.strip()
        m = STATE_RE.match(line)
        if m:
            state = parse_fields(m.group(1))
            required = {
                "checkpoint_last", "checkpoint_len", "state_ok",
                "checkpoint_valid", "mtp_draft_valid", "dspark_draft_valid",
                "cycles", "accepted", "no_draft", "skip",
            }
            missing = required - state.keys()
            if missing:
                die(f"missing E11 state fields in {log}:{line_no}: {sorted(missing)}")
            gates.append(Gate(len(gates) + 1, line_no, state))
            continue

        m = SCHED_RE.match(line)
        if m:
            if not gates:
                die(f"scheduler trace without state trace in {log}:{line_no}")
            g = gates[-1]
            if g.decision is not None:
                die(f"duplicate scheduler trace for gate {g.ordinal1} in {log}")
            g.decision = m.group(1)
            g.sched = parse_fields(m.group(2))
            if g.decision == "RUN":
                pending_run.append(g)
            continue

        m = SPEC_RE.match(line)
        if m:
            if not pending_run:
                die(f"spec entry without preceding RUN scheduler gate in {log}:{line_no}")
            g = pending_run.pop(0)
            g.spec_start = int(m.group(1))

    if not gates:
        die(f"no {STATE_PREFIX} traces found in {log}")
    if pending_run:
        die(
            f"{len(pending_run)} RUN scheduler gates in {log} were not followed "
            "by speculative function entry"
        )
    return gates


def spec_map(gates: list[Gate]) -> dict[int, Gate]:
    out: dict[int, Gate] = {}
    for g in gates:
        if g.spec_start is None:
            continue
        if g.spec_start in out:
            die(f"duplicate speculative absolute start={g.spec_start}")
        out[g.spec_start] = g
    return out


def first_set_divergence(a: set[int], b: set[int]) -> tuple[int | None, str]:
    for start in sorted(a | b):
        if (start in a) != (start in b):
            return start, "FAST_ONLY" if start in a else "REPLAY_ONLY"
    return None, "NONE"


def first_field_diff(
    a: dict[str, int], b: dict[str, int], fields: Sequence[str]
) -> list[str]:
    return [f for f in fields if a.get(f) != b.get(f)]


def analyze(
    ds4: Path,
    model: Path,
    oracle: Path,
    replay_out: Path,
    replay_log: Path,
    fast_base_out: Path,
    fast_base_log: Path,
    fast_trace_out: Path,
    fast_trace_log: Path,
    expected_ordinal1: int,
) -> None:
    oracle_ids = e10.tokenize(ds4, model, oracle)
    replay_ids = e10.tokenize(ds4, model, replay_out)
    fast_base_ids = e10.tokenize(ds4, model, fast_base_out)
    fast_trace_ids = e10.tokenize(ds4, model, fast_trace_out)

    baseline_diff = e10.first_divergence(oracle_ids, fast_base_ids)
    replay_diff = e10.first_divergence(oracle_ids, replay_ids)
    trace_diff = e10.first_divergence(oracle_ids, fast_trace_ids)
    trace_same = fast_base_ids == fast_trace_ids and baseline_diff == trace_diff
    baseline_ok = baseline_diff is not None and baseline_diff + 1 == expected_ordinal1
    replay_ok = replay_diff is None

    print(
        "FAST_COMMIT_SCHEDULER_DRAFT_VALID_BASELINE "
        f"first_diff_index0={e10.fmt_idx(baseline_diff)} "
        f"first_diff_ordinal1={e10.fmt_ord(baseline_diff)} "
        f"expected_ordinal1={expected_ordinal1} "
        f"generation_tps={e10.fmt_tps(e10.extract_tps(fast_base_log))} "
        f"result={'PASS' if baseline_ok else 'FAIL'}"
    )
    print(
        "FAST_COMMIT_SCHEDULER_DRAFT_VALID_ORACLE_CONTROL "
        f"replay_first_diff_index0={e10.fmt_idx(replay_diff)} "
        f"replay_first_diff_ordinal1={e10.fmt_ord(replay_diff)} "
        f"replay_tps={e10.fmt_tps(e10.extract_tps(replay_log))} "
        f"result={'PASS' if replay_ok else 'FAIL'}"
    )
    print(
        "FAST_COMMIT_SCHEDULER_DRAFT_VALID_TRACE_CONTROL "
        f"token_output={'EXACT' if fast_base_ids == fast_trace_ids else 'MISMATCH'} "
        f"frontier_same={1 if baseline_diff == trace_diff else 0} "
        f"result={'PASS' if trace_same else 'FAIL'}"
    )

    if not (baseline_ok and replay_ok and trace_same):
        print("FAST_COMMIT_SCHEDULER_DRAFT_VALID_SOURCE=INCONCLUSIVE")
        print("FAST_COMMIT_SCHEDULER_DRAFT_VALID_NEXT=ADJUDICATE_CONTROL")
        return

    fast = parse_gates(fast_trace_log)
    replay = parse_gates(replay_log)
    fm, rm = spec_map(fast), spec_map(replay)
    start, presence = first_set_divergence(set(fm), set(rm))
    print(
        "FAST_COMMIT_SCHEDULER_DRAFT_VALID_BINDING "
        f"fast_gates={len(fast)} replay_gates={len(replay)} "
        f"fast_specs={len(fm)} replay_specs={len(rm)} result=PASS"
    )
    if start is None:
        print("FAST_COMMIT_SCHEDULER_DRAFT_VALID_FIRST_SCHEDULE_DIVERGENCE=NONE")
        print("FAST_COMMIT_SCHEDULER_DRAFT_VALID_SOURCE=NOT_REPRODUCED")
        print("FAST_COMMIT_SCHEDULER_DRAFT_VALID_NEXT=ADJUDICATE_E10")
        return

    print(
        "FAST_COMMIT_SCHEDULER_DRAFT_VALID_FIRST_SCHEDULE_DIVERGENCE "
        f"start={start} presence={presence}"
    )

    common_before = [s for s in sorted(set(fm) & set(rm)) if s < start]
    anchor = common_before[-1] if common_before else None
    if anchor is None:
        print("FAST_COMMIT_SCHEDULER_DRAFT_VALID_SOURCE=INCONCLUSIVE")
        print("FAST_COMMIT_SCHEDULER_DRAFT_VALID_NEXT=DECODE_CONTROL_FLOW_AB")
        return

    fg_anchor, rg_anchor = fm[anchor], rm[anchor]
    fi = fast.index(fg_anchor)
    ri = replay.index(rg_anchor)
    print(
        "FAST_COMMIT_SCHEDULER_DRAFT_VALID_CAUSAL_ANCHOR "
        f"start={anchor} fast_gate_ordinal1={fg_anchor.ordinal1} "
        f"replay_gate_ordinal1={rg_anchor.ordinal1} result=PASS"
    )

    validity_fields = (
        "state_ok", "checkpoint_valid", "mtp_draft_valid",
        "dspark_draft_valid",
    )
    support_fields = (
        "dspark_draft_len", "mtp_n_raw", "dspark_cache_start",
        "dspark_cache_token_start", "dspark_cache_len",
    )
    sched_fields = (
        "pre_cycles", "pre_accepted", "pre_no_draft",
        "pre_skip", "pre_lifetime",
    )

    classification = None
    detail = None
    max_off = min(len(fast) - fi, len(replay) - ri, 12)
    for off in range(max_off):
        f = fast[fi + off]
        r = replay[ri + off]
        vd = first_field_diff(f.state, r.state, validity_fields)
        sd = first_field_diff(f.state, r.state, support_fields)
        fd = first_field_diff(f.sched or {}, r.sched or {}, sched_fields)
        print(
            "FAST_COMMIT_SCHEDULER_DRAFT_VALID_ROW "
            f"offset={off} fast_gate_ordinal1={f.ordinal1} "
            f"replay_gate_ordinal1={r.ordinal1} "
            f"fast_checkpoint_last={f.state.get('checkpoint_last')} "
            f"replay_checkpoint_last={r.state.get('checkpoint_last')} "
            f"fast_state_ok={f.state.get('state_ok')} "
            f"replay_state_ok={r.state.get('state_ok')} "
            f"fast_decision={f.decision or 'RESET'} "
            f"replay_decision={r.decision or 'RESET'} "
            f"fast_spec_start={f.spec_start if f.spec_start is not None else 'NONE'} "
            f"replay_spec_start={r.spec_start if r.spec_start is not None else 'NONE'} "
            f"validity_diff={','.join(vd) if vd else 'NONE'} "
            f"support_diff={','.join(sd) if sd else 'NONE'} "
            f"scheduler_diff={','.join(fd) if fd else 'NONE'}"
        )
        if classification is None:
            if vd:
                classification = "DRAFT_VALID_STATE_DIVERGENCE"
                detail = ",".join(vd)
            elif f.decision != r.decision:
                if fd:
                    classification = "SCHEDULER_COUNTER_DIVERGENCE"
                    detail = ",".join(fd)
                else:
                    classification = "SCHEDULER_POLICY_DIVERGENCE"
                    detail = "decision"
            elif f.spec_start != r.spec_start and (
                f.spec_start == start or r.spec_start == start
            ):
                classification = "SPECULATIVE_ENTRY_PRESENCE_DIVERGENCE"
                detail = "spec_entry"

        if f.spec_start == start or r.spec_start == start:
            if classification is None:
                classification = "SPECULATIVE_ENTRY_PRESENCE_DIVERGENCE"
                detail = "spec_entry"
            break

    if classification is None:
        classification = "CONTROL_FLOW_PRESENCE_DIVERGENCE"
        detail = "unaligned_gate_sequence"

    next_map = {
        "DRAFT_VALID_STATE_DIVERGENCE": "DRAFT_VALID_INVALIDATION_CAUSAL_AB",
        "SCHEDULER_COUNTER_DIVERGENCE": "SCHEDULER_COUNTER_CAUSAL_AB",
        "SCHEDULER_POLICY_DIVERGENCE": "SCHEDULER_POLICY_CAUSAL_AB",
        "SPECULATIVE_ENTRY_PRESENCE_DIVERGENCE": "SPECULATIVE_ENTRY_CONTROL_AB",
        "CONTROL_FLOW_PRESENCE_DIVERGENCE": "DECODE_CONTROL_FLOW_AB",
    }
    print(
        "FAST_COMMIT_SCHEDULER_DRAFT_VALID_CAUSE "
        f"class={classification} fields={detail}"
    )
    print(f"FAST_COMMIT_SCHEDULER_DRAFT_VALID_SOURCE={classification}")
    print(f"FAST_COMMIT_SCHEDULER_DRAFT_VALID_NEXT={next_map[classification]}")


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
        analyze(
            args.ds4, args.model, args.oracle,
            args.replay_out, args.replay_log,
            args.fast_base_out, args.fast_base_log,
            args.fast_trace_out, args.fast_trace_log,
            args.expected_ordinal1,
        )


if __name__ == "__main__":
    main()
