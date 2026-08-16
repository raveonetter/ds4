#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import dspark_fast_commit_draft_verify_input_ab as e10
import dspark_fast_commit_scheduler_draft_valid_ab as e11_base
import dspark_fast_commit_scheduler_draft_valid_ab_compat as e11

PROPOSER_FN = "static bool ds4_session_prepare_dspark_draft_impl("
ENTRY_PREFIX = "DS4_DSPARK_E12_ENTRY"
READY_PREFIX = "DS4_DSPARK_E12_READY"
CACHE_PREFIX = "DS4_DSPARK_E12_CACHE"
CHAIN_PREFIX = "DS4_DSPARK_E12_CHAIN"
RESULT_PREFIX = "DS4_DSPARK_E12_RESULT"

ENTRY_TRACE = r"""
    {
        const char *ds4_e12_env = getenv("DS4_DSPARK_E12_TRACE");
        if (ds4_e12_env && ds4_e12_env[0] &&
            strcmp(ds4_e12_env, "0") != 0) {
            const int ds4_e12_checkpoint_last =
                s && s->checkpoint.len > 0 ?
                s->checkpoint.v[s->checkpoint.len - 1] : -1;
            fprintf(stderr,
                    "DS4_DSPARK_E12_ENTRY start=%u token=%d "
                    "checkpoint_last=%d checkpoint_len=%d "
                    "checkpoint_valid=%d mtp_draft_valid=%d "
                    "pre_dspark_draft_valid=%d pre_dspark_draft_len=%u "
                    "capture_enabled=%d capture_valid=%d "
                    "capture_checkpoint_len=%u capture_batch_valid=%d "
                    "capture_batch_start=%u capture_batch_tokens=%u "
                    "mtp_n_raw=%u cache_start=%u cache_token_start=%u "
                    "cache_len=%u sched_cycles=%u sched_accepted=%u "
                    "sched_no_draft=%u sched_skip=%u lifetime_accepted=%u\n",
                    (unsigned)(pos + 1u), token,
                    ds4_e12_checkpoint_last,
                    s ? s->checkpoint.len : -1,
                    s ? (int)s->checkpoint_valid : 0,
                    s ? (int)s->mtp_draft_valid : 0,
                    s ? (int)s->dspark_draft_valid : 0,
                    s ? s->dspark_draft_len : 0u,
                    s ? (int)s->graph.dspark_capture_enabled : 0,
                    s ? (int)s->graph.dspark_capture_valid : 0,
                    s ? s->graph.dspark_capture_checkpoint_len : 0u,
                    s ? (int)s->graph.dspark_capture_batch_valid : 0,
                    s ? s->graph.dspark_capture_batch_start : 0u,
                    s ? s->graph.dspark_capture_batch_tokens : 0u,
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
"""

READY_TRACE = r"""
        {
            const char *ds4_e12_env = getenv("DS4_DSPARK_E12_TRACE");
            if (ds4_e12_env && ds4_e12_env[0] &&
                strcmp(ds4_e12_env, "0") != 0) {
                fprintf(stderr,
                        "DS4_DSPARK_E12_READY start=%u "
                        "capture_ok=%d batch_capture_ok=%d "
                        "stage0_ready=%d runtime_fused_stage0_setup=%d "
                        "capture_valid=%d capture_checkpoint_len=%u "
                        "capture_batch_valid=%d capture_batch_start=%u "
                        "capture_batch_tokens=%u cache_start=%u "
                        "cache_token_start=%u cache_len=%u\n",
                        (unsigned)(pos + 1u),
                        capture_ok ? 1 : 0,
                        batch_capture_ok ? 1 : 0,
                        stage0_ready ? 1 : 0,
                        runtime_fused_stage0_setup ? 1 : 0,
                        (int)s->graph.dspark_capture_valid,
                        s->graph.dspark_capture_checkpoint_len,
                        (int)s->graph.dspark_capture_batch_valid,
                        s->graph.dspark_capture_batch_start,
                        s->graph.dspark_capture_batch_tokens,
                        s->graph.dspark_cache_start,
                        s->graph.dspark_cache_token_start,
                        s->graph.dspark_cache_len);
                fflush(stderr);
            }
        }
"""

CACHE_TRACE = r"""
        {
            const char *ds4_e12_env = getenv("DS4_DSPARK_E12_TRACE");
            if (ds4_e12_env && ds4_e12_env[0] &&
                strcmp(ds4_e12_env, "0") != 0) {
                fprintf(stderr,
                        "DS4_DSPARK_E12_CACHE start=%u "
                        "draft_cache_ready=%d initial_cache_ready=%d "
                        "initial_cache_ok=%d initial_cache_rows=%u "
                        "cache_window_ok=%d cache_start=%u "
                        "cache_token_start=%u cache_len=%u "
                        "capture_batch_start=%u capture_batch_tokens=%u\n",
                        (unsigned)(pos + 1u),
                        draft_cache_ready ? 1 : 0,
                        initial_cache_ready ? 1 : 0,
                        initial_cache_ok ? 1 : 0,
                        initial_cache_rows,
                        cache_window_ok ? 1 : 0,
                        s->graph.dspark_cache_start,
                        s->graph.dspark_cache_token_start,
                        s->graph.dspark_cache_len,
                        s->graph.dspark_capture_batch_start,
                        s->graph.dspark_capture_batch_tokens);
                fflush(stderr);
            }
        }
"""

CHAIN_TRACE = r"""
        {
            const char *ds4_e12_env = getenv("DS4_DSPARK_E12_TRACE");
            if (ds4_e12_env && ds4_e12_env[0] &&
                strcmp(ds4_e12_env, "0") != 0) {
                fprintf(stderr,
                        "DS4_DSPARK_E12_CHAIN start=%u "
                        "stage0_ok=%d stage_input_ok=%d draft_cache_ready=%d "
                        "cache_window_ok=%d stage_chain_ready=%d "
                        "stage_chain_ok=%d stage_chain_done=%u "
                        "stage_cache_start=%u stage_cache_rows=%u "
                        "cache_start=%u cache_token_start=%u cache_len=%u\n",
                        (unsigned)(pos + 1u),
                        stage0_ok ? 1 : 0,
                        stage_input_ok ? 1 : 0,
                        draft_cache_ready ? 1 : 0,
                        cache_window_ok ? 1 : 0,
                        stage_chain_ready ? 1 : 0,
                        stage_chain_ok ? 1 : 0,
                        stage_chain_done,
                        stage_cache_start,
                        stage_cache_rows,
                        s->graph.dspark_cache_start,
                        s->graph.dspark_cache_token_start,
                        s->graph.dspark_cache_len);
                fflush(stderr);
            }
        }
"""

RESULT_TRACE = r"""
        {
            const char *ds4_e12_env = getenv("DS4_DSPARK_E12_TRACE");
            if (ds4_e12_env && ds4_e12_env[0] &&
                strcmp(ds4_e12_env, "0") != 0) {
                fprintf(stderr,
                        "DS4_DSPARK_E12_RESULT start=%u "
                        "stage0_ok=%d stage_input_ok=%d draft_cache_ready=%d "
                        "cache_window_ok=%d stage_chain_ok=%d "
                        "base_logits_ok=%d markov_ready=%d markov_ok=%d "
                        "confidence_ok=%d confidence_len=%u "
                        "confidence_prefix=%u markov_proposal_len=%u "
                        "fake_argmax_ok=%d draft_len=%u draft_valid=%d "
                        "cache_start=%u cache_token_start=%u cache_len=%u\n",
                        (unsigned)(pos + 1u),
                        stage0_ok ? 1 : 0,
                        stage_input_ok ? 1 : 0,
                        draft_cache_ready ? 1 : 0,
                        cache_window_ok ? 1 : 0,
                        stage_chain_ok ? 1 : 0,
                        base_logits_ok ? 1 : 0,
                        markov_ready ? 1 : 0,
                        markov_ok ? 1 : 0,
                        confidence_ok ? 1 : 0,
                        confidence_len,
                        confidence_prefix_len,
                        markov_proposal_len,
                        fake_argmax_ok ? 1 : 0,
                        s->dspark_draft_len,
                        s->dspark_draft_valid ? 1 : 0,
                        s->graph.dspark_cache_start,
                        s->graph.dspark_cache_token_start,
                        s->graph.dspark_cache_len);
                fflush(stderr);
            }
        }
"""

def die(msg: str) -> None:
    raise SystemExit(f"dspark_fast_commit_draft_valid_invalidation_causal_ab: {msg}")

def insert_before_once(text: str, anchor: str, payload: str, label: str) -> str:
    if text.count(anchor) != 1:
        die(f"expected exactly one {label} anchor, found {text.count(anchor)}")
    pos = text.find(anchor)
    return text[:pos] + payload + text[pos:]

def insert_after_statement_once(text: str, anchor: str, payload: str, label: str) -> str:
    if text.count(anchor) != 1:
        die(f"expected exactly one {label} anchor, found {text.count(anchor)}")
    pos = text.find(anchor)
    semi = text.find(";", pos)
    if semi < 0:
        die(f"{label} statement terminator not found")
    return text[:semi + 1] + payload + text[semi + 1:]

def instrument(source: Path) -> None:
    e11.instrument(source)
    text = source.read_text(encoding="utf-8")
    if ENTRY_PREFIX in text:
        die(f"{source} already contains E12 instrumentation")

    fn_start = text.find(PROPOSER_FN)
    if fn_start < 0:
        die("DSpark proposer implementation not found")
    fn_open = text.find("{", fn_start)
    fn_end = e11_base.find_matching_brace(text, fn_open)
    if fn_open < 0 or fn_end < 0:
        die("could not isolate DSpark proposer implementation")

    fn = text[fn_start:fn_end + 1]
    fn = insert_before_once(fn, "    s->dspark_draft_valid = false;", ENTRY_TRACE, "proposer entry")
    fn = insert_after_statement_once(fn, "        const bool runtime_fused_stage0_setup =", READY_TRACE, "runtime_fused_stage0_setup")
    fn = insert_after_statement_once(fn, "        DS4_DSPARK_PROP_ADD(propose_cache_ms, cache_t0)", CACHE_TRACE, "cache phase")
    fn = insert_after_statement_once(fn, "        DS4_DSPARK_PROP_ADD(propose_chain_ms, chain_t0)", CHAIN_TRACE, "stage-chain phase")
    fn = insert_before_once(fn, "        if (probe_log) {", RESULT_TRACE, "proposer result")

    text = text[:fn_start] + fn + text[fn_end + 1:]
    source.write_text(text, encoding="utf-8")
    print(
        "FAST_COMMIT_DRAFT_VALID_INVALIDATION_INSTRUMENT "
        "e11_schedule_trace=1 proposer_entry=1 proposer_ready=1 "
        "proposer_cache=1 proposer_chain=1 proposer_result=1 "
        "probe_mode=OFF result=PASS"
    )

@dataclass
class ProposerEvent:
    start: int
    line_no: int
    entry: dict[str, int]
    ready: dict[str, int] | None = None
    cache: dict[str, int] | None = None
    chain: dict[str, int] | None = None
    result: dict[str, int] | None = None

ENTRY_RE = re.compile(rf"^{ENTRY_PREFIX} (.*)$")
READY_RE = re.compile(rf"^{READY_PREFIX} (.*)$")
CACHE_RE = re.compile(rf"^{CACHE_PREFIX} (.*)$")
CHAIN_RE = re.compile(rf"^{CHAIN_PREFIX} (.*)$")
RESULT_RE = re.compile(rf"^{RESULT_PREFIX} (.*)$")

def _attach(current: ProposerEvent | None, fields: dict[str, int], phase: str, log: Path, line_no: int) -> ProposerEvent:
    if current is None:
        die(f"{phase} trace without proposer entry in {log}:{line_no}")
    start = fields.get("start")
    if start != current.start:
        die(f"{phase} start mismatch in {log}:{line_no}: entry={current.start} phase={start}")
    if getattr(current, phase) is not None:
        die(f"duplicate {phase} trace for start={current.start} in {log}")
    setattr(current, phase, fields)
    return current

def parse_proposers(log: Path) -> list[ProposerEvent]:
    events: list[ProposerEvent] = []
    current: ProposerEvent | None = None
    for line_no, raw in enumerate(log.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        line = raw.strip()
        m = ENTRY_RE.match(line)
        if m:
            fields = e11_base.parse_fields(m.group(1))
            required = {
                "start", "token", "checkpoint_last", "checkpoint_len",
                "checkpoint_valid", "capture_valid", "capture_checkpoint_len",
                "capture_batch_valid", "capture_batch_start", "capture_batch_tokens",
                "mtp_n_raw", "cache_start", "cache_token_start", "cache_len",
                "sched_cycles", "sched_accepted", "sched_no_draft", "sched_skip",
            }
            missing = required - fields.keys()
            if missing:
                die(f"missing E12 entry fields in {log}:{line_no}: {sorted(missing)}")
            current = ProposerEvent(fields["start"], line_no, fields)
            events.append(current)
            continue
        for regex, phase in (
            (READY_RE, "ready"),
            (CACHE_RE, "cache"),
            (CHAIN_RE, "chain"),
            (RESULT_RE, "result"),
        ):
            m = regex.match(line)
            if m:
                current = _attach(current, e11_base.parse_fields(m.group(1)), phase, log, line_no)
                if phase == "result":
                    current = None
                break

    if not events:
        die(f"no {ENTRY_PREFIX} traces found in {log}")
    return events

def event_map(events: list[ProposerEvent]) -> dict[int, ProposerEvent]:
    out: dict[int, ProposerEvent] = {}
    for event in events:
        if event.start in out:
            die(f"duplicate proposer start={event.start}")
        out[event.start] = event
    return out

def diff_fields(a: dict[str, int] | None, b: dict[str, int] | None, fields: tuple[str, ...]) -> list[str]:
    if a is None or b is None:
        return ["PHASE_PRESENCE"] if a is not b else []
    return [name for name in fields if a.get(name) != b.get(name)]

TOKEN_FIELDS = ("token", "checkpoint_last", "checkpoint_len", "checkpoint_valid")
CAPTURE_FIELDS = (
    "capture_enabled", "capture_valid", "capture_checkpoint_len",
    "capture_batch_valid", "capture_batch_start", "capture_batch_tokens",
)
ENTRY_CACHE_FIELDS = ("mtp_n_raw", "cache_start", "cache_token_start", "cache_len")
SCHED_FIELDS = ("sched_cycles", "sched_accepted", "sched_no_draft", "sched_skip", "lifetime_accepted")
READY_FIELDS = ("capture_ok", "batch_capture_ok", "stage0_ready", "runtime_fused_stage0_setup")
CACHE_FIELDS = (
    "draft_cache_ready", "initial_cache_ready", "initial_cache_ok",
    "initial_cache_rows", "cache_window_ok", "cache_start",
    "cache_token_start", "cache_len", "capture_batch_start", "capture_batch_tokens",
)
CHAIN_FIELDS = (
    "stage0_ok", "stage_input_ok", "draft_cache_ready", "cache_window_ok",
    "stage_chain_ready", "stage_chain_ok", "stage_chain_done",
    "stage_cache_start", "stage_cache_rows", "cache_start", "cache_token_start", "cache_len",
)
RESULT_FIELDS = (
    "stage0_ok", "stage_input_ok", "draft_cache_ready", "cache_window_ok",
    "stage_chain_ok", "base_logits_ok", "markov_ready", "markov_ok",
    "confidence_ok", "confidence_len", "confidence_prefix",
    "markov_proposal_len", "fake_argmax_ok", "draft_len", "draft_valid",
    "cache_start", "cache_token_start", "cache_len",
)

def classify_event(fast: ProposerEvent, replay: ProposerEvent) -> tuple[str | None, str, list[str]]:
    groups = [
        ("ENTRY_TOKEN", "TOKEN_OR_CHECKPOINT_DIVERGENCE", diff_fields(fast.entry, replay.entry, TOKEN_FIELDS)),
        ("ENTRY_CAPTURE", "CAPTURE_FRONTIER_DIVERGENCE", diff_fields(fast.entry, replay.entry, CAPTURE_FIELDS)),
        ("ENTRY_CACHE", "DSPARK_CACHE_FRONTIER_DIVERGENCE", diff_fields(fast.entry, replay.entry, ENTRY_CACHE_FIELDS)),
        ("ENTRY_SCHED", "SCHEDULER_STATE_DIVERGENCE", diff_fields(fast.entry, replay.entry, SCHED_FIELDS)),
        ("READY", "PROPOSER_READINESS_DIVERGENCE", diff_fields(fast.ready, replay.ready, READY_FIELDS)),
        ("CACHE", "CACHE_SEED_WINDOW_DIVERGENCE", diff_fields(fast.cache, replay.cache, CACHE_FIELDS)),
        ("CHAIN", "STAGE_CHAIN_DIVERGENCE", diff_fields(fast.chain, replay.chain, CHAIN_FIELDS)),
        ("RESULT", "PROPOSAL_GENERATION_DIVERGENCE", diff_fields(fast.result, replay.result, RESULT_FIELDS)),
    ]
    for phase, cls, diffs in groups:
        if diffs:
            return cls, phase, diffs
    return None, "NONE", []

def fmt_diff(values: list[str]) -> str:
    return ",".join(values) if values else "NONE"

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
    baseline_ok = baseline_diff is not None and baseline_diff + 1 == expected_ordinal1
    replay_ok = replay_diff is None
    trace_same = fast_base_ids == fast_trace_ids and baseline_diff == trace_diff

    print(
        "FAST_COMMIT_DRAFT_VALID_INVALIDATION_BASELINE "
        f"first_diff_index0={e10.fmt_idx(baseline_diff)} "
        f"first_diff_ordinal1={e10.fmt_ord(baseline_diff)} "
        f"expected_ordinal1={expected_ordinal1} "
        f"generation_tps={e10.fmt_tps(e10.extract_tps(fast_base_log))} "
        f"result={'PASS' if baseline_ok else 'FAIL'}"
    )
    print(
        "FAST_COMMIT_DRAFT_VALID_INVALIDATION_ORACLE_CONTROL "
        f"replay_first_diff_index0={e10.fmt_idx(replay_diff)} "
        f"replay_first_diff_ordinal1={e10.fmt_ord(replay_diff)} "
        f"replay_tps={e10.fmt_tps(e10.extract_tps(replay_log))} "
        f"result={'PASS' if replay_ok else 'FAIL'}"
    )
    print(
        "FAST_COMMIT_DRAFT_VALID_INVALIDATION_TRACE_CONTROL "
        f"token_output={'EXACT' if fast_base_ids == fast_trace_ids else 'MISMATCH'} "
        f"frontier_same={1 if baseline_diff == trace_diff else 0} "
        f"result={'PASS' if trace_same else 'FAIL'}"
    )
    if not (baseline_ok and replay_ok and trace_same):
        print("FAST_COMMIT_DRAFT_VALID_INVALIDATION_SOURCE=INCONCLUSIVE")
        print("FAST_COMMIT_DRAFT_VALID_INVALIDATION_NEXT=ADJUDICATE_CONTROL")
        return

    fast_gates = e11.parse_gates(fast_trace_log)
    replay_gates = e11.parse_gates(replay_log)
    fast_specs = e11_base.spec_map(fast_gates)
    replay_specs = e11_base.spec_map(replay_gates)
    schedule_start, presence = e11_base.first_set_divergence(set(fast_specs), set(replay_specs))
    print(
        "FAST_COMMIT_DRAFT_VALID_INVALIDATION_SCHEDULE_BINDING "
        f"fast_specs={len(fast_specs)} replay_specs={len(replay_specs)} "
        f"first_divergence_start={schedule_start if schedule_start is not None else 'NONE'} "
        f"presence={presence} result=PASS"
    )
    if schedule_start is None:
        print("FAST_COMMIT_DRAFT_VALID_INVALIDATION_SOURCE=NOT_REPRODUCED")
        print("FAST_COMMIT_DRAFT_VALID_INVALIDATION_NEXT=ADJUDICATE_E11")
        return

    fast_events = parse_proposers(fast_trace_log)
    replay_events = parse_proposers(replay_log)
    fm = event_map(fast_events)
    rm = event_map(replay_events)

    have_target = schedule_start in fm and schedule_start in rm
    print(
        "FAST_COMMIT_DRAFT_VALID_INVALIDATION_PROPOSER_BINDING "
        f"fast_events={len(fast_events)} replay_events={len(replay_events)} "
        f"fast_has_schedule_start={1 if schedule_start in fm else 0} "
        f"replay_has_schedule_start={1 if schedule_start in rm else 0} "
        f"result={'PASS' if have_target else 'FAIL'}"
    )
    if not have_target:
        print("FAST_COMMIT_DRAFT_VALID_INVALIDATION_SOURCE=INCONCLUSIVE")
        print("FAST_COMMIT_DRAFT_VALID_INVALIDATION_NEXT=PROPOSER_BINDING_AUDIT")
        return

    common_upto = [start for start in sorted(set(fm) & set(rm)) if start <= schedule_start]
    first_div_start: int | None = None
    first_class: str | None = None
    first_phase = "NONE"
    first_fields: list[str] = []

    for start in common_upto:
        f = fm[start]
        r = rm[start]
        cls, phase, fields = classify_event(f, r)
        token_diff = diff_fields(f.entry, r.entry, TOKEN_FIELDS)
        capture_diff = diff_fields(f.entry, r.entry, CAPTURE_FIELDS)
        entry_cache_diff = diff_fields(f.entry, r.entry, ENTRY_CACHE_FIELDS)
        sched_diff = diff_fields(f.entry, r.entry, SCHED_FIELDS)
        ready_diff = diff_fields(f.ready, r.ready, READY_FIELDS)
        cache_diff = diff_fields(f.cache, r.cache, CACHE_FIELDS)
        chain_diff = diff_fields(f.chain, r.chain, CHAIN_FIELDS)
        result_diff = diff_fields(f.result, r.result, RESULT_FIELDS)
        print(
            "FAST_COMMIT_DRAFT_VALID_INVALIDATION_ROW "
            f"start={start} "
            f"fast_token={f.entry.get('token')} replay_token={r.entry.get('token')} "
            f"fast_checkpoint_last={f.entry.get('checkpoint_last')} "
            f"replay_checkpoint_last={r.entry.get('checkpoint_last')} "
            f"token_diff={fmt_diff(token_diff)} "
            f"capture_diff={fmt_diff(capture_diff)} "
            f"entry_cache_diff={fmt_diff(entry_cache_diff)} "
            f"scheduler_diff={fmt_diff(sched_diff)} "
            f"ready_diff={fmt_diff(ready_diff)} "
            f"cache_diff={fmt_diff(cache_diff)} "
            f"chain_diff={fmt_diff(chain_diff)} "
            f"result_diff={fmt_diff(result_diff)}"
        )
        if cls is not None and first_div_start is None:
            first_div_start = start
            first_class = cls
            first_phase = phase
            first_fields = fields

    if first_div_start is None:
        print("FAST_COMMIT_DRAFT_VALID_INVALIDATION_FIRST_PROPOSER_DIVERGENCE=NONE")
        print("FAST_COMMIT_DRAFT_VALID_INVALIDATION_SOURCE=CONTROL_FLOW_BINDING_DIVERGENCE")
        print("FAST_COMMIT_DRAFT_VALID_INVALIDATION_NEXT=SPEC_ENTRY_CONTROL_AB")
        return

    print(
        "FAST_COMMIT_DRAFT_VALID_INVALIDATION_FIRST_PROPOSER_DIVERGENCE "
        f"start={first_div_start} phase={first_phase} "
        f"class={first_class} fields={fmt_diff(first_fields)}"
    )
    if first_div_start != schedule_start:
        print(
            "FAST_COMMIT_DRAFT_VALID_INVALIDATION_CAUSAL_DISTANCE "
            f"proposer_start={first_div_start} schedule_start={schedule_start} "
            f"delta_tokens={schedule_start - first_div_start}"
        )

    next_map = {
        "TOKEN_OR_CHECKPOINT_DIVERGENCE": "TOKEN_HISTORY_CAUSAL_AUDIT",
        "CAPTURE_FRONTIER_DIVERGENCE": "FULL_ACCEPT_CAPTURE_FRONTIER_CAUSAL_AB",
        "DSPARK_CACHE_FRONTIER_DIVERGENCE": "FULL_ACCEPT_DSPARK_CACHE_FRONTIER_CAUSAL_AB",
        "SCHEDULER_STATE_DIVERGENCE": "SCHEDULER_STATE_CAUSAL_AB",
        "PROPOSER_READINESS_DIVERGENCE": "PROPOSER_READINESS_CAUSAL_AB",
        "CACHE_SEED_WINDOW_DIVERGENCE": "CACHE_SEED_WINDOW_CAUSAL_AB",
        "STAGE_CHAIN_DIVERGENCE": "STAGE_CHAIN_CAUSAL_AB",
        "PROPOSAL_GENERATION_DIVERGENCE": "PROPOSAL_GENERATION_CAUSAL_AB",
    }
    print(
        "FAST_COMMIT_DRAFT_VALID_INVALIDATION_CAUSE "
        f"class={first_class} phase={first_phase} fields={fmt_diff(first_fields)}"
    )
    print(f"FAST_COMMIT_DRAFT_VALID_INVALIDATION_SOURCE={first_class}")
    print(
        "FAST_COMMIT_DRAFT_VALID_INVALIDATION_NEXT="
        f"{next_map.get(first_class, 'PROPOSER_CONTROL_FLOW_AUDIT')}"
    )

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
    analyze(
        args.ds4, args.model, args.oracle,
        args.replay_out, args.replay_log,
        args.fast_base_out, args.fast_base_log,
        args.fast_trace_out, args.fast_trace_log,
        args.expected_ordinal1,
    )

if __name__ == "__main__":
    main()
