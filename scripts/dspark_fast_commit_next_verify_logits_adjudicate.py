#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

ROW_RE = re.compile(
    r"^DS4_DSPARK_E9_LOGITS event_ordinal1=(\d+) verify_call_seq=(\d+) "
    r"start=(\d+) n_tokens=(\d+) row=(\d+) hash=([0-9a-fA-F]+) "
    r"verify_top_id=(-?\d+) top1_id=(\d+) top1=([^\s]+) "
    r"top2_id=(\d+) top2=([^\s]+) margin=([^\s]+) status=(\S+)$"
)
TRACE_RE = re.compile(
    r"^DS4_DSPARK_TRACE_RETURN path=(\S+) commit_type=(\S+) drafted=(\d+) "
    r"accepted_drafts=(\d+) returned=(\d+) correction=([01]) ids=(.*)$"
)


@dataclass(frozen=True)
class Row:
    event_ordinal1: int
    call_seq: int
    start: int
    n_tokens: int
    row: int
    hash_hex: str
    verify_top: int
    top1: int
    top1_v: float
    top2: int
    top2_v: float
    margin: float
    status: str

    @property
    def verify_top_valid(self) -> bool:
        # metal_graph_verify_suffix_tops_impl initializes only
        # top_rows = n_tokens - 1 slots. The final logits row is the
        # continuation row and deliberately has no row_tops entry.
        return self.row + 1 < self.n_tokens


@dataclass(frozen=True)
class Event:
    ordinal1: int
    path: str
    commit_type: str
    drafted: int
    accepted_drafts: int
    returned: int
    correction: bool
    ids: tuple[int, ...]


def die(msg: str) -> None:
    raise SystemExit(f"e9_adjudicate: {msg}")


def marker(lines: list[str], prefix: str) -> str | None:
    for line in lines:
        if line.startswith(prefix):
            return line
    return None


def field(line: str | None, name: str) -> str | None:
    if not line:
        return None
    m = re.search(rf"(?:^|\s){re.escape(name)}=([^\s]+)", line)
    return m.group(1) if m else None


def parse_rows(path: Path) -> dict[tuple[int, int, int], Row]:
    out: dict[tuple[int, int, int], Row] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = ROW_RE.match(raw.strip())
        if not m:
            continue
        r = Row(
            event_ordinal1=int(m.group(1)),
            call_seq=int(m.group(2)),
            start=int(m.group(3)),
            n_tokens=int(m.group(4)),
            row=int(m.group(5)),
            hash_hex=m.group(6).lower(),
            verify_top=int(m.group(7)),
            top1=int(m.group(8)),
            top1_v=float(m.group(9)),
            top2=int(m.group(10)),
            top2_v=float(m.group(11)),
            margin=float(m.group(12)),
            status=m.group(13),
        )
        key = (r.event_ordinal1, r.call_seq, r.row)
        if key in out:
            die(f"duplicate row {key} in {path}")
        out[key] = r
    if not out:
        die(f"no E9 logits rows in {path}")
    return out


def parse_events(path: Path) -> list[Event]:
    out: list[Event] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = TRACE_RE.match(raw.strip())
        if not m:
            continue
        ids = tuple(int(x) for x in m.group(7).split(",") if x)
        returned = int(m.group(5))
        if len(ids) != returned:
            die(f"event returned={returned}, ids={len(ids)} in {path}")
        out.append(
            Event(
                ordinal1=len(out) + 1,
                path=m.group(1),
                commit_type=m.group(2),
                drafted=int(m.group(3)),
                accepted_drafts=int(m.group(4)),
                returned=returned,
                correction=m.group(6) == "1",
                ids=ids,
            )
        )
    if not out:
        die(f"no commit events in {path}")
    return out


def event_at(events: list[Event], ordinal1: int) -> Event | None:
    i = ordinal1 - 1
    return events[i] if 0 <= i < len(events) else None


def fmt_key(key: tuple[int, int, int] | None) -> str:
    if key is None:
        return "NONE"
    return f"event_ordinal1={key[0]} verify_call_seq={key[1]} row={key[2]}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dir", type=Path)
    args = ap.parse_args()

    d = args.result_dir.expanduser().resolve()
    summary = d / "next-verify-logits-ab-summary.log"
    fast_log = d / "E9_FAST_TRACE.log"
    replay_log = d / "E9_REPLAY_TRACE.log"
    for p in (summary, fast_log, replay_log):
        if not p.is_file():
            die(f"missing {p}")

    lines = summary.read_text(encoding="utf-8", errors="replace").splitlines()
    baseline = marker(lines, "FAST_COMMIT_VERIFY_LOGITS_BASELINE ")
    oracle = marker(lines, "FAST_COMMIT_VERIFY_LOGITS_ORACLE_CONTROL ")
    trace_control = marker(lines, "FAST_COMMIT_VERIFY_LOGITS_TRACE_CONTROL ")
    trace = marker(lines, "FAST_COMMIT_VERIFY_LOGITS_TRACE ")
    causal = marker(lines, "FAST_COMMIT_VERIFY_LOGITS_CAUSAL_WINDOW ")

    fast = parse_rows(fast_log)
    replay = parse_rows(replay_log)
    fast_events = parse_events(fast_log)
    replay_events = parse_events(replay_log)

    legacy_invalid_final = [
        (arm, r)
        for arm, rows in (("FAST", fast), ("REPLAY", replay))
        for r in rows.values()
        if not r.verify_top_valid and r.verify_top >= 0
    ]
    valid_self_top_bad = [
        (arm, r)
        for arm, rows in (("FAST", fast), ("REPLAY", replay))
        for r in rows.values()
        if r.status != "PASS"
        or (r.verify_top_valid and r.verify_top >= 0 and r.verify_top != r.top1)
    ]

    gates = {
        "baseline": field(baseline, "result") == "PASS",
        "oracle": field(oracle, "result") == "PASS",
        "trace_control": field(trace_control, "result") == "PASS",
        "alignment": field(trace, "event_alignment") == "PASS",
        "self_top_contract": len(valid_self_top_bad) == 0,
        "causal_window": field(causal, "result") == "PASS",
    }
    failed = [name for name, ok in gates.items() if not ok]

    print(
        "FAST_COMMIT_VERIFY_LOGITS_CONTRACT "
        "top_rows=n_tokens-1 acceptance_index=row_tops[i-1] "
        f"legacy_invalid_final_row_reads={len(legacy_invalid_final)} "
        f"valid_self_top_mismatches={len(valid_self_top_bad)}"
    )
    print(
        "FAST_COMMIT_VERIFY_LOGITS_CONTROL_DETAIL "
        + " ".join(f"{name}={'PASS' if ok else 'FAIL'}" for name, ok in gates.items())
    )
    print(
        "FAST_COMMIT_VERIFY_LOGITS_ALIGNMENT_DETAIL "
        f"fast_rows={len(fast)} replay_rows={len(replay)} "
        f"fast_events={len(fast_events)} replay_events={len(replay_events)} "
        f"fast_extra_verify_calls={len({(k[0], k[1]) for k in fast if k[1] > 0})} "
        f"replay_extra_verify_calls={len({(k[0], k[1]) for k in replay if k[1] > 0})}"
    )
    for arm, r in legacy_invalid_final[:12]:
        print(
            "FAST_COMMIT_VERIFY_LOGITS_IGNORED_FINAL_ROW_TOP "
            f"arm={arm} event_ordinal1={r.event_ordinal1} row={r.row} "
            f"n_tokens={r.n_tokens} legacy_verify_top_id={r.verify_top} "
            f"readback_top1_id={r.top1}"
        )
    for arm, r in valid_self_top_bad[:12]:
        print(
            "FAST_COMMIT_VERIFY_LOGITS_VALID_TOP_BAD_ROW "
            f"arm={arm} event_ordinal1={r.event_ordinal1} row={r.row} "
            f"n_tokens={r.n_tokens} verify_top_id={r.verify_top} "
            f"readback_top1_id={r.top1} status={r.status}"
        )

    causal_ord_s = field(causal, "full_accept_event_ordinal1")
    next_ord_s = field(causal, "next_event_ordinal1")
    causal_ord = int(causal_ord_s) if causal_ord_s and causal_ord_s != "NONE" else None
    next_ord = int(next_ord_s) if next_ord_s and next_ord_s != "NONE" else None

    common = sorted(k for k in (set(fast) & set(replay)) if k[1] == 0)
    first_hash: tuple[int, int, int] | None = None
    first_argmax: tuple[int, int, int] | None = None
    first_shape: tuple[int, int, int] | None = None
    precommit_hash_div = False

    for key in common:
        f = fast[key]
        r = replay[key]
        same_shape = f.start == r.start and f.n_tokens == r.n_tokens
        if first_shape is None and not same_shape:
            first_shape = key
        if first_hash is None and f.hash_hex != r.hash_hex:
            first_hash = key
        if first_argmax is None and f.top1 != r.top1:
            first_argmax = key
        if causal_ord is not None and key[0] <= causal_ord and f.hash_hex != r.hash_hex:
            precommit_hash_div = True

        if (
            key[0] in {causal_ord, next_ord}
            or f.hash_hex != r.hash_hex
            or f.top1 != r.top1
            or not same_shape
        ):
            print(
                "FAST_COMMIT_VERIFY_LOGITS_REANALYZED_ROW "
                f"event_ordinal1={key[0]} verify_call_seq={key[1]} row={key[2]} "
                f"fast_start={f.start} replay_start={r.start} "
                f"fast_n_tokens={f.n_tokens} replay_n_tokens={r.n_tokens} "
                f"shape_result={'EXACT' if same_shape else 'MISMATCH'} "
                f"hash_result={'EXACT' if f.hash_hex == r.hash_hex else 'MISMATCH'} "
                f"argmax_result={'EXACT' if f.top1 == r.top1 else 'MISMATCH'} "
                f"fast_top1_id={f.top1} replay_top1_id={r.top1} "
                f"fast_top2_id={f.top2} replay_top2_id={r.top2} "
                f"fast_margin={f.margin:.17g} replay_margin={r.margin:.17g}"
            )

    print(f"FAST_COMMIT_VERIFY_LOGITS_FIRST_HASH_DIVERGENCE {fmt_key(first_hash)}")
    print(f"FAST_COMMIT_VERIFY_LOGITS_FIRST_ARGMAX_DIVERGENCE {fmt_key(first_argmax)}")
    print(f"FAST_COMMIT_VERIFY_LOGITS_FIRST_SHAPE_DIVERGENCE {fmt_key(first_shape)}")

    rejection_row_match = False
    if first_argmax is not None:
        event_ord, _, row_idx = first_argmax
        fe = event_at(fast_events, event_ord)
        revent = event_at(replay_events, event_ord)
        expected_reject_row = (
            fe.accepted_drafts - 1
            if fe is not None and 0 < fe.accepted_drafts < fe.drafted
            else None
        )
        rejection_row_match = expected_reject_row == row_idx
        print(
            "FAST_COMMIT_VERIFY_LOGITS_REJECTION_MAPPING "
            f"event_ordinal1={event_ord} row={row_idx} "
            f"fast_commit_type={fe.commit_type if fe else 'MISSING'} "
            f"fast_drafted={fe.drafted if fe else 'MISSING'} "
            f"fast_accepted_drafts={fe.accepted_drafts if fe else 'MISSING'} "
            f"expected_rejection_row={expected_reject_row if expected_reject_row is not None else 'NONE'} "
            f"replay_commit_type={revent.commit_type if revent else 'MISSING'} "
            f"rejection_row_match={1 if rejection_row_match else 0}"
        )

    controls_ok = not failed
    if not controls_ok:
        source = "INCONCLUSIVE_CONTROL"
        next_step = "FIX_LISTED_CONTROL"
    elif precommit_hash_div:
        source = "INCONCLUSIVE_PRECOMMIT_LOGIT_DIVERGENCE"
        next_step = "ADJUDICATE_VERIFIER_DETERMINISM"
    elif first_shape is not None and first_argmax is None:
        source = "VERIFY_INPUT_SHAPE_DIVERGENCE"
        next_step = "NEXT_ITERATION_DRAFT_SCHEDULE_AB"
    elif first_argmax is not None and next_ord is not None and first_argmax[0] == next_ord and rejection_row_match:
        source = "NEXT_VERIFY_ARGMAX_CAUSE_LOCALIZED"
        next_step = "LAYERWISE_NEXT_ITERATION_CP_AB"
    elif first_argmax is not None:
        source = "NEXT_VERIFY_ARGMAX_DIVERGENCE_FOUND"
        next_step = "ROW_EVENT_MAPPING_ADJUDICATION"
    elif first_hash is not None:
        source = "NEXT_VERIFY_NUMERICAL_DIVERGENCE_ONLY"
        next_step = "CORRECTION_DECISION_LOGIC_AB"
    else:
        source = "VERIFY_LOGITS_EXACT"
        next_step = "NON_LOGIT_CORRECTION_STATE_AB"

    print(f"FAST_COMMIT_VERIFY_LOGITS_COMPARED_ROWS={len(common)}")
    print(f"FAST_COMMIT_VERIFY_LOGITS_CONTROL_FAILED={','.join(failed) if failed else 'NONE'}")
    print(f"FAST_COMMIT_VERIFY_LOGITS_CONTROL_VERDICT={'CONTROL_PASS' if controls_ok else 'CONTROL_FAIL'}")
    print(f"FAST_COMMIT_VERIFY_LOGITS_SOURCE={source}")
    print(f"FAST_COMMIT_VERIFY_LOGITS_NEXT={next_step}")


if __name__ == "__main__":
    main()
