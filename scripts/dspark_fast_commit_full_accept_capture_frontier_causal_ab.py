#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import dspark_fast_commit_draft_verify_input_ab as e10
import dspark_fast_commit_scheduler_draft_valid_ab as e11_base
import dspark_fast_commit_scheduler_draft_valid_ab_compat as e11
import dspark_fast_commit_draft_valid_invalidation_causal_ab as e12

SPEC_FN = "static int ds4_session_eval_dspark_speculative_argmax("
TRACE_PREFIX = "DS4_DSPARK_E13_CAPTURE"
INTERVENTION_ENV = "DS4_DSPARK_E13_INVALIDATE_VERIFIED_BATCH"

TRACE_HELPER = r'''
static void ds4_e13_trace_capture_frontier(const char *phase,
                                            ds4_session *s,
                                            int start,
                                            int draft_n,
                                            int commit_drafts) {
    const char *env = getenv("DS4_DSPARK_E13_TRACE");
    if (!env || !env[0] || strcmp(env, "0") == 0 || !s) return;
    ds4_gpu_graph *g = &s->graph;
    fprintf(stderr,
            "DS4_DSPARK_E13_CAPTURE phase=%s start=%d draft_n=%d "
            "commit_drafts=%d checkpoint_len=%d checkpoint_valid=%d "
            "capture_valid=%d capture_mask=%u capture_checkpoint_len=%u "
            "batch_valid=%d batch_mask=%u batch_start=%u batch_tokens=%u "
            "cache_start=%u cache_token_start=%u cache_len=%u\n",
            phase,
            start,
            draft_n,
            commit_drafts,
            s->checkpoint.len,
            (int)s->checkpoint_valid,
            (int)g->dspark_capture_valid,
            g->dspark_capture_mask,
            g->dspark_capture_checkpoint_len,
            (int)g->dspark_capture_batch_valid,
            g->dspark_capture_batch_mask,
            g->dspark_capture_batch_start,
            g->dspark_capture_batch_tokens,
            g->dspark_cache_start,
            g->dspark_cache_token_start,
            g->dspark_cache_len);
    fflush(stderr);
}

'''

VERIFY_TRACE = r'''
    ds4_e13_trace_capture_frontier("VERIFY_DONE",
                                   s,
                                   start,
                                   draft_n,
                                   commit_drafts);

'''

FAST_POST_NOTE = r'''
            ds4_e13_trace_capture_frontier("FAST_POST_NOTE",
                                           s,
                                           start,
                                           draft_n,
                                           commit_drafts);
            {
                const char *ds4_e13_fix =
                    getenv("DS4_DSPARK_E13_INVALIDATE_VERIFIED_BATCH");
                if (ds4_e13_fix && ds4_e13_fix[0] &&
                    strcmp(ds4_e13_fix, "0") != 0) {
                    metal_graph_dspark_capture_batch_invalidate(&s->graph);
                    ds4_e13_trace_capture_frontier("FAST_POST_INTERVENTION",
                                                   s,
                                                   start,
                                                   draft_n,
                                                   commit_drafts);
                }
            }
'''

REPLAY_RESTORE_TRACE = r'''
    ds4_e13_trace_capture_frontier("REPLAY_POST_RESTORE",
                                   s,
                                   start,
                                   draft_n,
                                   commit_drafts);

'''

REPLAY_FINAL_TRACE = r'''
        ds4_e13_trace_capture_frontier("REPLAY_FINAL",
                                       s,
                                       start,
                                       draft_n,
                                       commit_drafts);
'''


def die(msg: str) -> None:
    raise SystemExit(f"dspark_fast_commit_full_accept_capture_frontier_causal_ab: {msg}")


def insert_before_once(text: str, anchor: str, payload: str, label: str) -> str:
    count = text.count(anchor)
    if count != 1:
        die(f"expected exactly one {label} anchor, found {count}")
    pos = text.find(anchor)
    return text[:pos] + payload + text[pos:]


def insert_after_once(text: str, anchor: str, payload: str, label: str) -> str:
    count = text.count(anchor)
    if count != 1:
        die(f"expected exactly one {label} anchor, found {count}")
    pos = text.find(anchor) + len(anchor)
    return text[:pos] + payload + text[pos:]


def instrument(source: Path) -> None:
    # E13 deliberately keeps E11/E12 observation enabled so the intervention
    # can be bound back to the already-proven start=199 schedule/proposer gate.
    e12.instrument(source)
    text = source.read_text(encoding="utf-8")
    if TRACE_PREFIX in text:
        die(f"{source} already contains E13 instrumentation")

    fn_start = text.find(SPEC_FN)
    if fn_start < 0:
        die("DSpark speculative function not found")
    fn_open = text.find("{", fn_start)
    fn_end = e11_base.find_matching_brace(text, fn_open)
    if fn_open < 0 or fn_end < 0:
        die("could not isolate DSpark speculative function")

    text = text[:fn_start] + TRACE_HELPER + text[fn_start:]
    fn_start += len(TRACE_HELPER)
    fn_end += len(TRACE_HELPER)
    fn = text[fn_start:fn_end + 1]

    fn = insert_before_once(
        fn,
        "    /* Full accept, non-TP, opt-in: skip the rollback and the sequential",
        VERIFY_TRACE,
        "post-verify full-accept",
    )

    fast_anchor = (
        "            s->checkpoint_valid = true;\n"
        "            ds4_session_dspark_capture_note_checkpoint(s);\n"
    )
    fn = insert_after_once(fn, fast_anchor, FAST_POST_NOTE, "fast commit capture note")

    replay_restore_anchor = (
        "    if (!ok) {\n"
        "        if (tp_verify_sent &&\n"
    )
    fn = insert_before_once(
        fn,
        replay_restore_anchor,
        REPLAY_RESTORE_TRACE,
        "post-rollback replay frontier",
    )

    replay_final_anchor = (
        "        s->checkpoint_valid = true;\n"
        "        ds4_session_dspark_capture_note_checkpoint(s);\n"
    )
    fn = insert_after_once(
        fn,
        replay_final_anchor,
        REPLAY_FINAL_TRACE,
        "replay final capture note",
    )

    text = text[:fn_start] + fn + text[fn_end + 1:]
    source.write_text(text, encoding="utf-8")
    print(
        "FAST_COMMIT_CAPTURE_FRONTIER_INSTRUMENT "
        "e11_schedule_trace=1 e12_proposer_trace=1 "
        "verify_trace=1 fast_commit_trace=1 replay_trace=1 "
        "intervention=INVALIDATE_VERIFIED_BATCH probe_mode=OFF result=PASS"
    )


@dataclass
class CaptureEvent:
    phase: str
    start: int
    line_no: int
    fields: dict[str, int]


TRACE_RE = re.compile(rf"^{TRACE_PREFIX} phase=([A-Z_]+) (.*)$")


def parse_capture(log: Path) -> list[CaptureEvent]:
    events: list[CaptureEvent] = []
    for line_no, raw in enumerate(
        log.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
    ):
        m = TRACE_RE.match(raw.strip())
        if not m:
            continue
        phase = m.group(1)
        fields = e11_base.parse_fields(m.group(2))
        required = {
            "start", "draft_n", "commit_drafts", "checkpoint_len",
            "checkpoint_valid", "capture_valid", "capture_mask",
            "capture_checkpoint_len", "batch_valid", "batch_mask",
            "batch_start", "batch_tokens", "cache_start",
            "cache_token_start", "cache_len",
        }
        missing = required - fields.keys()
        if missing:
            die(f"missing E13 fields in {log}:{line_no}: {sorted(missing)}")
        events.append(CaptureEvent(phase, fields["start"], line_no, fields))
    if not events:
        die(f"no {TRACE_PREFIX} traces found in {log}")
    return events


def phase_map(events: list[CaptureEvent], start: int) -> dict[str, CaptureEvent]:
    out: dict[str, CaptureEvent] = {}
    for event in events:
        if event.start != start:
            continue
        if event.phase in out:
            die(f"duplicate E13 phase={event.phase} start={start}")
        out[event.phase] = event
    return out


BATCH_FIELDS = ("batch_valid", "batch_mask", "batch_start", "batch_tokens")
ROW_FIELDS = ("capture_valid", "capture_mask", "capture_checkpoint_len")
CACHE_FIELDS = ("cache_start", "cache_token_start", "cache_len")


def diffs(a: dict[str, int], b: dict[str, int], fields: tuple[str, ...]) -> list[str]:
    return [name for name in fields if a.get(name) != b.get(name)]


def fmt_fields(values: list[str]) -> str:
    return ",".join(values) if values else "NONE"


def fmt_capture(event: CaptureEvent | None) -> str:
    if event is None:
        return "MISSING"
    f = event.fields
    return (
        f"row={f['capture_valid']}:{f['capture_mask']}:{f['capture_checkpoint_len']} "
        f"batch={f['batch_valid']}:{f['batch_mask']}:{f['batch_start']}:{f['batch_tokens']} "
        f"cache={f['cache_start']}:{f['cache_token_start']}:{f['cache_len']}"
    )


def pct(new: float | None, old: float | None) -> str:
    if new is None or old is None or old == 0:
        return "NA"
    return f"{(new / old - 1.0) * 100.0:.3f}"


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
    intervention_out: Path,
    intervention_log: Path,
    expected_ordinal1: int,
) -> None:
    oracle_ids = e10.tokenize(ds4, model, oracle)
    replay_ids = e10.tokenize(ds4, model, replay_out)
    fast_base_ids = e10.tokenize(ds4, model, fast_base_out)
    fast_trace_ids = e10.tokenize(ds4, model, fast_trace_out)
    intervention_ids = e10.tokenize(ds4, model, intervention_out)

    baseline_diff = e10.first_divergence(oracle_ids, fast_base_ids)
    replay_diff = e10.first_divergence(oracle_ids, replay_ids)
    trace_diff = e10.first_divergence(oracle_ids, fast_trace_ids)
    intervention_diff = e10.first_divergence(oracle_ids, intervention_ids)

    baseline_ok = baseline_diff is not None and baseline_diff + 1 == expected_ordinal1
    replay_ok = replay_diff is None
    trace_ok = fast_base_ids == fast_trace_ids and baseline_diff == trace_diff

    fast_tps = e10.extract_tps(fast_base_log)
    replay_tps = e10.extract_tps(replay_log)
    intervention_tps = e10.extract_tps(intervention_log)

    print(
        "FAST_COMMIT_CAPTURE_FRONTIER_BASELINE "
        f"first_diff_index0={e10.fmt_idx(baseline_diff)} "
        f"first_diff_ordinal1={e10.fmt_ord(baseline_diff)} "
        f"expected_ordinal1={expected_ordinal1} "
        f"generation_tps={e10.fmt_tps(fast_tps)} "
        f"result={'PASS' if baseline_ok else 'FAIL'}"
    )
    print(
        "FAST_COMMIT_CAPTURE_FRONTIER_ORACLE_CONTROL "
        f"replay_first_diff_index0={e10.fmt_idx(replay_diff)} "
        f"replay_first_diff_ordinal1={e10.fmt_ord(replay_diff)} "
        f"replay_tps={e10.fmt_tps(replay_tps)} "
        f"result={'PASS' if replay_ok else 'FAIL'}"
    )
    print(
        "FAST_COMMIT_CAPTURE_FRONTIER_TRACE_CONTROL "
        f"token_output={'EXACT' if fast_base_ids == fast_trace_ids else 'MISMATCH'} "
        f"frontier_same={1 if baseline_diff == trace_diff else 0} "
        f"result={'PASS' if trace_ok else 'FAIL'}"
    )
    print(
        "FAST_COMMIT_CAPTURE_FRONTIER_INTERVENTION_OUTPUT "
        f"first_diff_index0={e10.fmt_idx(intervention_diff)} "
        f"first_diff_ordinal1={e10.fmt_ord(intervention_diff)} "
        f"generation_tps={e10.fmt_tps(intervention_tps)} "
        f"vs_fast_tps_delta_pct={pct(intervention_tps, fast_tps)} "
        f"vs_replay_tps_delta_pct={pct(intervention_tps, replay_tps)} "
        f"result={'PASS' if intervention_diff is None else 'FAIL'}"
    )

    if not (baseline_ok and replay_ok and trace_ok):
        print("FAST_COMMIT_CAPTURE_FRONTIER_SOURCE=INCONCLUSIVE_CONTROL")
        print("FAST_COMMIT_CAPTURE_FRONTIER_NEXT=ADJUDICATE_CONTROL")
        return

    fast_gates = e11.parse_gates(fast_trace_log)
    replay_gates = e11.parse_gates(replay_log)
    intervention_gates = e11.parse_gates(intervention_log)
    fast_specs = e11_base.spec_map(fast_gates)
    replay_specs = e11_base.spec_map(replay_gates)
    intervention_specs = e11_base.spec_map(intervention_gates)
    schedule_start, presence = e11_base.first_set_divergence(set(fast_specs), set(replay_specs))
    if schedule_start is None:
        print("FAST_COMMIT_CAPTURE_FRONTIER_SOURCE=E12_SCHEDULE_NOT_REPRODUCED")
        print("FAST_COMMIT_CAPTURE_FRONTIER_NEXT=ADJUDICATE_E12")
        return
    intervention_presence_at_target = (
        (schedule_start in intervention_specs) == (schedule_start in replay_specs)
    )
    intervention_schedule_start, intervention_presence = e11_base.first_set_divergence(
        set(intervention_specs), set(replay_specs)
    )
    print(
        "FAST_COMMIT_CAPTURE_FRONTIER_SCHEDULE "
        f"baseline_first_divergence_start={schedule_start} presence={presence} "
        f"intervention_matches_replay_at_target={1 if intervention_presence_at_target else 0} "
        f"intervention_first_divergence_start="
        f"{intervention_schedule_start if intervention_schedule_start is not None else 'NONE'} "
        f"intervention_presence={intervention_presence}"
    )

    fast_capture = parse_capture(fast_trace_log)
    replay_capture = parse_capture(replay_log)
    intervention_capture = parse_capture(intervention_log)

    full_accept_starts = [
        e.start for e in fast_capture
        if e.phase == "FAST_POST_NOTE" and
           e.fields["draft_n"] > 0 and
           e.fields["draft_n"] == e.fields["commit_drafts"] and
           e.start < schedule_start
    ]
    if not full_accept_starts:
        print("FAST_COMMIT_CAPTURE_FRONTIER_SOURCE=FULL_ACCEPT_BINDING_FAILED")
        print("FAST_COMMIT_CAPTURE_FRONTIER_NEXT=TRACE_FULL_ACCEPT_BINDING")
        return
    anchor_start = max(full_accept_starts)
    fm = phase_map(fast_capture, anchor_start)
    rm = phase_map(replay_capture, anchor_start)
    im = phase_map(intervention_capture, anchor_start)

    required_fast = {"VERIFY_DONE", "FAST_POST_NOTE"}
    required_replay = {"VERIFY_DONE", "REPLAY_POST_RESTORE", "REPLAY_FINAL"}
    required_intervention = {"VERIFY_DONE", "FAST_POST_NOTE", "FAST_POST_INTERVENTION"}
    lifecycle_ok = (
        required_fast <= fm.keys() and
        required_replay <= rm.keys() and
        required_intervention <= im.keys()
    )
    print(
        "FAST_COMMIT_CAPTURE_FRONTIER_BINDING "
        f"full_accept_start={anchor_start} schedule_start={schedule_start} "
        f"distance={schedule_start - anchor_start} "
        f"result={'PASS' if lifecycle_ok else 'FAIL'}"
    )
    if not lifecycle_ok:
        print("FAST_COMMIT_CAPTURE_FRONTIER_SOURCE=LIFECYCLE_TRACE_INCOMPLETE")
        print("FAST_COMMIT_CAPTURE_FRONTIER_NEXT=TRACE_LIFECYCLE_AUDIT")
        return

    fast_verify = fm["VERIFY_DONE"]
    replay_verify = rm["VERIFY_DONE"]
    fast_post = fm["FAST_POST_NOTE"]
    replay_restore = rm["REPLAY_POST_RESTORE"]
    replay_final = rm["REPLAY_FINAL"]
    intervention_pre = im["FAST_POST_NOTE"]
    intervention_post = im["FAST_POST_INTERVENTION"]

    verify_batch_diff = diffs(fast_verify.fields, replay_verify.fields, BATCH_FIELDS)
    verify_row_diff = diffs(fast_verify.fields, replay_verify.fields, ROW_FIELDS)
    branch_batch_diff = diffs(fast_post.fields, replay_final.fields, BATCH_FIELDS)
    intervention_batch_diff = diffs(intervention_post.fields, replay_final.fields, BATCH_FIELDS)
    intervention_row_diff = diffs(intervention_post.fields, replay_final.fields, ROW_FIELDS)
    intervention_cache_diff = diffs(intervention_post.fields, replay_final.fields, CACHE_FIELDS)

    print(
        "FAST_COMMIT_CAPTURE_FRONTIER_LIFECYCLE "
        f"start={anchor_start} "
        f"fast_verify=\"{fmt_capture(fast_verify)}\" "
        f"replay_verify=\"{fmt_capture(replay_verify)}\" "
        f"fast_post=\"{fmt_capture(fast_post)}\" "
        f"replay_post_restore=\"{fmt_capture(replay_restore)}\" "
        f"replay_final=\"{fmt_capture(replay_final)}\" "
        f"intervention_pre=\"{fmt_capture(intervention_pre)}\" "
        f"intervention_post=\"{fmt_capture(intervention_post)}\""
    )
    print(
        "FAST_COMMIT_CAPTURE_FRONTIER_BRANCH_AB "
        f"verify_batch_diff={fmt_fields(verify_batch_diff)} "
        f"verify_row_diff={fmt_fields(verify_row_diff)} "
        f"fast_vs_replay_final_batch_diff={fmt_fields(branch_batch_diff)} "
        f"intervention_vs_replay_final_batch_diff={fmt_fields(intervention_batch_diff)} "
        f"intervention_vs_replay_final_row_diff={fmt_fields(intervention_row_diff)} "
        f"intervention_vs_replay_final_cache_diff={fmt_fields(intervention_cache_diff)}"
    )

    replay_props = e12.event_map(e12.parse_proposers(replay_log))
    intervention_props = e12.event_map(e12.parse_proposers(intervention_log))
    proposer_bound = schedule_start in replay_props and schedule_start in intervention_props
    proposer_capture_diff: list[str] = ["MISSING_TARGET"]
    proposer_cache_diff: list[str] = ["MISSING_TARGET"]
    if proposer_bound:
        proposer_capture_diff = e12.diff_fields(
            intervention_props[schedule_start].entry,
            replay_props[schedule_start].entry,
            e12.CAPTURE_FIELDS,
        )
        proposer_cache_diff = e12.diff_fields(
            intervention_props[schedule_start].entry,
            replay_props[schedule_start].entry,
            e12.ENTRY_CACHE_FIELDS,
        )
    print(
        "FAST_COMMIT_CAPTURE_FRONTIER_NEXT_PROPOSER "
        f"start={schedule_start} bound={1 if proposer_bound else 0} "
        f"capture_diff={fmt_fields(proposer_capture_diff)} "
        f"cache_diff={fmt_fields(proposer_cache_diff)}"
    )

    pre_branch_exact = not verify_batch_diff and not verify_row_diff
    branch_created = bool(branch_batch_diff)
    metadata_repaired = (
        not intervention_batch_diff and
        not intervention_row_diff and
        not intervention_cache_diff
    )
    next_entry_repaired = proposer_bound and not proposer_capture_diff and not proposer_cache_diff
    output_repaired = intervention_diff is None
    cause_proven = (
        pre_branch_exact and
        branch_created and
        metadata_repaired and
        next_entry_repaired and
        intervention_presence_at_target and
        output_repaired
    )

    print(
        "FAST_COMMIT_CAPTURE_FRONTIER_INTERVENTION "
        f"pre_branch_exact={1 if pre_branch_exact else 0} "
        f"branch_created_batch_divergence={1 if branch_created else 0} "
        f"metadata_repaired={1 if metadata_repaired else 0} "
        f"next_entry_repaired={1 if next_entry_repaired else 0} "
        f"schedule_target_repaired={1 if intervention_presence_at_target else 0} "
        f"token_stream_repaired={1 if output_repaired else 0} "
        f"result={'PASS' if cause_proven else 'FAIL'}"
    )

    if cause_proven:
        print("FAST_COMMIT_CAPTURE_FRONTIER_CAUSE=PROVEN")
        print("FAST_COMMIT_CAPTURE_FRONTIER_SOURCE=RETAINED_VERIFIED_SUFFIX_BATCH_CAPTURE")
        print("FAST_COMMIT_CAPTURE_FRONTIER_REPAIR_CANDIDATE=INVALIDATE_VERIFIED_BATCH_ON_FAST_FULL_ACCEPT")
        print("FAST_COMMIT_CAPTURE_FRONTIER_NEXT=PRODUCTION_REPAIR_PERF_AB")
    else:
        print("FAST_COMMIT_CAPTURE_FRONTIER_CAUSE=PARTIAL")
        if not pre_branch_exact:
            print("FAST_COMMIT_CAPTURE_FRONTIER_SOURCE=VERIFIER_CAPTURE_DIVERGENCE")
            print("FAST_COMMIT_CAPTURE_FRONTIER_NEXT=VERIFIER_CAPTURE_INPUT_AB")
        elif not metadata_repaired or not next_entry_repaired:
            print("FAST_COMMIT_CAPTURE_FRONTIER_SOURCE=CAPTURE_METADATA_NOT_SUFFICIENT")
            print("FAST_COMMIT_CAPTURE_FRONTIER_NEXT=CAPTURE_PAYLOAD_OR_FRONTIER_AUDIT")
        elif not output_repaired:
            print("FAST_COMMIT_CAPTURE_FRONTIER_SOURCE=CAPTURE_SCHEDULE_CAUSE_ONLY")
            print("FAST_COMMIT_CAPTURE_FRONTIER_NEXT=FAST_COMMIT_TARGET_STATE_AB")
        else:
            print("FAST_COMMIT_CAPTURE_FRONTIER_SOURCE=CONTROL_FLOW_RESIDUAL")
            print("FAST_COMMIT_CAPTURE_FRONTIER_NEXT=POST_INTERVENTION_SCHEDULE_AUDIT")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
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
    ap.add_argument("--intervention-out", type=Path, required=True)
    ap.add_argument("--intervention-log", type=Path, required=True)
    ap.add_argument("--expected-ordinal1", type=int, default=27)
    args = parser.parse_args()
    if args.cmd == "instrument":
        instrument(args.source)
        return
    analyze(
        args.ds4,
        args.model,
        args.oracle,
        args.replay_out,
        args.replay_log,
        args.fast_base_out,
        args.fast_base_log,
        args.fast_trace_out,
        args.fast_trace_log,
        args.intervention_out,
        args.intervention_log,
        args.expected_ordinal1,
    )


if __name__ == "__main__":
    main()
