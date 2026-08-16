#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import dspark_fast_commit_scheduler_draft_valid_ab as base


SCHED_FN = "static bool ds4_session_dspark_scheduler_should_skip("
SCHED_CALL = "ds4_session_dspark_scheduler_should_skip(s)"
SPEC_FN = "static int ds4_session_eval_dspark_speculative_argmax("
NO_DRAFT_GATE = "if (!s || !s->dspark_draft_valid || s->dspark_draft_len == 0) {"


WRAPPER = r'''
/* E11 compatibility instrumentation for the current should-skip scheduler.
 * The production predicate mutates skip state, so evaluate it exactly once
 * and expose the inverse as the experiment's positive RUN/SKIP decision. */
static bool ds4_e11_scheduler_should_run(ds4_session *s) {
    const unsigned pre_cycles = s ? s->dspark_sched_cycles : 0;
    const unsigned pre_accepted = s ? s->dspark_sched_accepted : 0;
    const unsigned pre_no_draft = s ? s->dspark_sched_no_draft : 0;
    const unsigned pre_skip = s ? s->dspark_sched_skip : 0;
    const unsigned pre_lifetime =
        s ? s->dspark_sched_lifetime_accepted : 0;

    const bool should_skip = ds4_session_dspark_scheduler_should_skip(s);
    const bool decision = !should_skip;

    const unsigned post_cycles = s ? s->dspark_sched_cycles : 0;
    const unsigned post_accepted = s ? s->dspark_sched_accepted : 0;
    const unsigned post_no_draft = s ? s->dspark_sched_no_draft : 0;
    const unsigned post_skip = s ? s->dspark_sched_skip : 0;
    const unsigned post_lifetime =
        s ? s->dspark_sched_lifetime_accepted : 0;

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
                pre_cycles, post_cycles,
                pre_accepted, post_accepted,
                pre_no_draft, post_no_draft,
                pre_skip, post_skip,
                pre_lifetime, post_lifetime);
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
                s && s->checkpoint.len > 0 ?
                s->checkpoint.v[s->checkpoint.len - 1] : -1;
            const bool ds4_e11_state_ok =
                s && s->dspark_draft_valid && s->dspark_draft_len != 0;
            fprintf(stderr,
                    "DS4_DSPARK_E11_STATE checkpoint_last=%d "
                    "checkpoint_len=%d state_ok=%d "
                    "checkpoint_valid=%d mtp_draft_valid=%d "
                    "dspark_draft_valid=%d dspark_draft_len=%u "
                    "mtp_n_raw=%u dspark_cache_start=%u "
                    "dspark_cache_token_start=%u dspark_cache_len=%u "
                    "cycles=%u accepted=%u no_draft=%u skip=%u "
                    "lifetime_accepted=%u\n",
                    ds4_e11_checkpoint_last,
                    s ? s->checkpoint.len : -1,
                    ds4_e11_state_ok ? 1 : 0,
                    s ? (int)s->checkpoint_valid : 0,
                    s ? (int)s->mtp_draft_valid : 0,
                    s ? (int)s->dspark_draft_valid : 0,
                    s ? s->dspark_draft_len : 0u,
                    s ? s->graph.mtp_n_raw : 0u,
                    s ? s->graph.dspark_cache_start : 0u,
                    s ? s->graph.dspark_cache_token_start : 0u,
                    s ? s->graph.dspark_cache_len : 0u,
                    s ? s->dspark_sched_cycles : 0u,
                    s ? s->dspark_sched_accepted : 0u,
                    s ? s->dspark_sched_no_draft : 0u,
                    s ? s->dspark_sched_skip : 0u,
                    s ? s->dspark_sched_lifetime_accepted : 0u);
            fflush(stderr);
        }
    }
'''


def die(msg: str) -> None:
    raise SystemExit(f"dspark_fast_commit_scheduler_draft_valid_ab_compat: {msg}")


def instrument(source: Path) -> None:
    text = source.read_text(encoding="utf-8")
    if base.STATE_PREFIX in text or "ds4_e11_scheduler_should_run" in text:
        die(f"{source} already contains E11 instrumentation")

    sched_start = text.find(SCHED_FN)
    if sched_start < 0:
        die("current should-skip scheduler function not found")
    sched_open = text.find("{", sched_start)
    sched_end = base.find_matching_brace(text, sched_open)
    if sched_open < 0 or sched_end < 0:
        die("could not isolate current scheduler function")

    call_positions: list[int] = []
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
    replacement = "!ds4_e11_scheduler_should_run(s)"
    text = text[:call_pos] + replacement + text[call_pos + len(SCHED_CALL):]

    sched_start = text.find(SCHED_FN)
    sched_open = text.find("{", sched_start)
    sched_end = base.find_matching_brace(text, sched_open)
    text = text[:sched_end + 1] + "\n" + WRAPPER + text[sched_end + 1:]

    spec_start = text.find(SPEC_FN)
    if spec_start < 0:
        die("speculative argmax function not found")
    spec_open = text.find("{", spec_start)
    spec_end = base.find_matching_brace(text, spec_open)
    if spec_open < 0 or spec_end < 0:
        die("could not isolate speculative argmax function")
    spec = text[spec_start:spec_end + 1]

    if spec.count(NO_DRAFT_GATE) != 1:
        die(
            "expected exactly one DSpark no-draft gate in speculative function, "
            f"found {spec.count(NO_DRAFT_GATE)}"
        )
    gate_pos = spec.find(NO_DRAFT_GATE)
    spec = spec[:gate_pos] + STATE_TRACE + spec[gate_pos:]

    m = re.search(r"\b(?:uint32_t|size_t|int)\s+start\s*=", spec)
    if not m:
        die("absolute start declaration not found in speculative function")
    start_semicolon = spec.find(";", m.end())
    if start_semicolon < 0:
        die("start declaration terminator not found")
    spec = (
        spec[:start_semicolon + 1]
        + base.SPEC_TRACE
        + spec[start_semicolon + 1:]
    )
    text = text[:spec_start] + spec + text[spec_end + 1:]

    source.write_text(text, encoding="utf-8")
    print(
        "FAST_COMMIT_SCHEDULER_DRAFT_VALID_INSTRUMENT "
        "scheduler_api=SHOULD_SKIP_INVERTED "
        "scheduler_call_sites=1 state_gate_sites=1 spec_entry_sites=1 result=PASS"
    )


def parse_gates(log: Path) -> list[base.Gate]:
    gates: list[base.Gate] = []
    pending_sched: list[tuple[str, dict[str, int], int]] = []
    spec_candidate: base.Gate | None = None

    for line_no, raw in enumerate(
        log.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
    ):
        line = raw.strip()

        m = base.SCHED_RE.match(line)
        if m:
            pending_sched.append(
                (m.group(1), base.parse_fields(m.group(2)), line_no)
            )
            continue

        m = base.STATE_RE.match(line)
        if m:
            state = base.parse_fields(m.group(1))
            required = {
                "checkpoint_last", "checkpoint_len", "state_ok",
                "checkpoint_valid", "mtp_draft_valid", "dspark_draft_valid",
                "cycles", "accepted", "no_draft", "skip",
            }
            missing = required - state.keys()
            if missing:
                die(
                    f"missing E11 state fields in {log}:{line_no}: "
                    f"{sorted(missing)}"
                )

            g = base.Gate(len(gates) + 1, line_no, state)
            if pending_sched:
                decision, sched, _sched_line = pending_sched.pop(0)
                g.decision = decision
                g.sched = sched
            gates.append(g)

            if g.decision == "RUN" and state.get("state_ok") == 1:
                spec_candidate = g
            else:
                spec_candidate = None
            continue

        m = base.SPEC_RE.match(line)
        if m:
            if spec_candidate is None:
                for candidate in reversed(gates):
                    if (
                        candidate.spec_start is None
                        and candidate.decision == "RUN"
                        and candidate.state.get("state_ok") == 1
                    ):
                        spec_candidate = candidate
                        break
            if spec_candidate is None:
                die(
                    f"spec entry without an eligible scheduler/state gate "
                    f"in {log}:{line_no}"
                )
            spec_candidate.spec_start = int(m.group(1))
            spec_candidate = None

    if not gates:
        die(f"no {base.STATE_PREFIX} traces found in {log}")

    return gates


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
        return

    base.parse_gates = parse_gates
    base.analyze(
        args.ds4,
        args.model,
        args.oracle,
        args.replay_out,
        args.replay_log,
        args.fast_base_out,
        args.fast_base_log,
        args.fast_trace_out,
        args.fast_trace_log,
        args.expected_ordinal1,
    )


if __name__ == "__main__":
    main()
