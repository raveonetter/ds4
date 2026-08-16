#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
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
        return self.row + 1 < self.n_tokens


@dataclass
class VerifyGroup:
    event_ordinal1: int
    rows: dict[tuple[int, int], Row] = field(default_factory=dict)
    first_line: int = 0
    last_line: int = 0

    def add(self, line_no: int, row: Row) -> None:
        key = (row.call_seq, row.row)
        if key in self.rows:
            die(
                f"duplicate E9 row event={row.event_ordinal1} "
                f"call={row.call_seq} row={row.row}"
            )
        self.rows[key] = row
        if self.first_line == 0:
            self.first_line = line_no
        self.last_line = line_no

    def primary_rows(self) -> dict[int, Row]:
        return {r.row: r for r in self.rows.values() if r.call_seq == 0}

    def primary_shape(self) -> tuple[int, int] | None:
        rows = list(self.primary_rows().values())
        if not rows:
            return None
        starts = {r.start for r in rows}
        ntoks = {r.n_tokens for r in rows}
        if len(starts) != 1 or len(ntoks) != 1:
            die(f"inconsistent primary E9 shape for event {self.event_ordinal1}")
        return next(iter(starts)), next(iter(ntoks))


@dataclass
class Event:
    ordinal1: int
    path: str
    commit_type: str
    drafted: int
    accepted_drafts: int
    returned: int
    correction: bool
    ids: tuple[int, ...]
    token_start0: int
    line_no: int
    verify_event_ordinal1: int | None = None
    orphan_verify_ordinals: tuple[int, ...] = ()


@dataclass
class ParsedArm:
    rows: dict[tuple[int, int, int], Row]
    groups: dict[int, VerifyGroup]
    events: list[Event]
    trailing_orphans: tuple[int, ...]


def die(msg: str) -> None:
    raise SystemExit(f"e9_adjudicate: {msg}")


def marker(lines: list[str], prefix: str) -> str | None:
    for line in lines:
        if line.startswith(prefix):
            return line
    return None


def field_value(line: str | None, name: str) -> str | None:
    if not line:
        return None
    m = re.search(rf"(?:^|\s){re.escape(name)}=([^\s]+)", line)
    return m.group(1) if m else None


def parse_row(m: re.Match[str]) -> Row:
    return Row(
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


def parse_arm(path: Path) -> ParsedArm:
    rows: dict[tuple[int, int, int], Row] = {}
    groups: dict[int, VerifyGroup] = {}
    events: list[Event] = []
    pending: list[int] = []
    seen_pending: set[int] = set()
    token_cursor = 0

    for line_no, raw in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
    ):
        s = raw.strip()
        rm = ROW_RE.match(s)
        if rm:
            r = parse_row(rm)
            key = (r.event_ordinal1, r.call_seq, r.row)
            if key in rows:
                die(f"duplicate row {key} in {path}")
            rows[key] = r
            group = groups.setdefault(
                r.event_ordinal1, VerifyGroup(r.event_ordinal1)
            )
            group.add(line_no, r)
            if r.event_ordinal1 not in seen_pending:
                pending.append(r.event_ordinal1)
                seen_pending.add(r.event_ordinal1)
            continue

        tm = TRACE_RE.match(s)
        if not tm:
            continue
        ids = tuple(int(x) for x in tm.group(7).split(",") if x)
        returned = int(tm.group(5))
        if len(ids) != returned:
            die(f"event returned={returned}, ids={len(ids)} in {path}")

        # E9 rows are emitted just after verifier execution; the commit trace
        # is emitted just before return. Bind each return to the nearest
        # preceding unbound verifier group. Older pending groups are explicit
        # orphans, never an implicit ordinal shift.
        verify_ord = pending[-1] if pending else None
        orphans = tuple(pending[:-1]) if pending else ()
        pending.clear()
        seen_pending.clear()
        events.append(
            Event(
                ordinal1=len(events) + 1,
                path=tm.group(1),
                commit_type=tm.group(2),
                drafted=int(tm.group(3)),
                accepted_drafts=int(tm.group(4)),
                returned=returned,
                correction=tm.group(6) == "1",
                ids=ids,
                token_start0=token_cursor,
                line_no=line_no,
                verify_event_ordinal1=verify_ord,
                orphan_verify_ordinals=orphans,
            )
        )
        token_cursor += returned

    if not rows:
        die(f"no E9 logits rows in {path}")
    if not events:
        die(f"no commit events in {path}")
    return ParsedArm(rows, groups, events, tuple(pending))


def group_for_event(arm: ParsedArm, event: Event) -> VerifyGroup | None:
    if event.verify_event_ordinal1 is None:
        return None
    return arm.groups.get(event.verify_event_ordinal1)


def events_by_token_start(arm: ParsedArm) -> dict[int, Event]:
    out: dict[int, Event] = {}
    for event in arm.events:
        if event.token_start0 in out:
            die(f"duplicate generated token_start0={event.token_start0}")
        out[event.token_start0] = event
    return out


def event_shape(arm: ParsedArm, event: Event | None) -> tuple[int, int] | None:
    if event is None:
        return None
    group = group_for_event(arm, event)
    return None if group is None else group.primary_shape()


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

    fast = parse_arm(fast_log)
    replay = parse_arm(replay_log)

    all_rows = [("FAST", r) for r in fast.rows.values()] + [
        ("REPLAY", r) for r in replay.rows.values()
    ]
    legacy_invalid_final = [
        (arm, r)
        for arm, r in all_rows
        if not r.verify_top_valid and r.verify_top >= 0
    ]
    valid_self_top_bad = [
        (arm, r)
        for arm, r in all_rows
        if r.status != "PASS"
        or (r.verify_top_valid and r.verify_top >= 0 and r.verify_top != r.top1)
    ]

    fast_orphans = (
        sum(len(e.orphan_verify_ordinals) for e in fast.events)
        + len(fast.trailing_orphans)
    )
    replay_orphans = (
        sum(len(e.orphan_verify_ordinals) for e in replay.events)
        + len(replay.trailing_orphans)
    )
    fast_unbound = sum(e.verify_event_ordinal1 is None for e in fast.events)
    replay_unbound = sum(e.verify_event_ordinal1 is None for e in replay.events)

    fast_by_pos = events_by_token_start(fast)
    replay_by_pos = events_by_token_start(replay)
    causal_fast = next(
        (e for e in fast.events if e.commit_type == "FULL_ACCEPT_FAST"), None
    )
    causal_replay = (
        replay_by_pos.get(causal_fast.token_start0) if causal_fast else None
    )
    causal_fast_shape = event_shape(fast, causal_fast)
    causal_replay_shape = event_shape(replay, causal_replay)
    causal_anchor_ok = (
        causal_fast is not None
        and causal_replay is not None
        and causal_replay.commit_type == "FULL_ACCEPT_REPLAY"
        and causal_fast.ids == causal_replay.ids
        and causal_fast_shape is not None
        and causal_fast_shape == causal_replay_shape
    )

    gates = {
        "baseline": field_value(baseline, "result") == "PASS",
        "oracle": field_value(oracle, "result") == "PASS",
        "trace_control": field_value(trace_control, "result") == "PASS",
        "self_top_contract": len(valid_self_top_bad) == 0,
        "log_order_binding": (
            fast_orphans == 0
            and replay_orphans == 0
            and fast_unbound == 0
            and replay_unbound == 0
        ),
        "causal_anchor": causal_anchor_ok,
    }
    failed = [name for name, ok in gates.items() if not ok]

    print(
        "FAST_COMMIT_VERIFY_LOGITS_CONTRACT "
        "top_rows=n_tokens-1 acceptance_index=row_tops[i-1] "
        "cross_arm_key=generated_token_start0 "
        "log_binding=nearest_preceding_verify_group "
        f"legacy_invalid_final_row_reads={len(legacy_invalid_final)} "
        f"valid_self_top_mismatches={len(valid_self_top_bad)}"
    )
    print(
        "FAST_COMMIT_VERIFY_LOGITS_CONTROL_DETAIL "
        + " ".join(
            f"{name}={'PASS' if ok else 'FAIL'}" for name, ok in gates.items()
        )
    )
    print(
        "FAST_COMMIT_VERIFY_LOGITS_BINDING_DETAIL "
        f"fast_verify_groups={len(fast.groups)} fast_commit_events={len(fast.events)} "
        f"fast_orphan_verify_groups={fast_orphans} fast_unbound_returns={fast_unbound} "
        f"replay_verify_groups={len(replay.groups)} replay_commit_events={len(replay.events)} "
        f"replay_orphan_verify_groups={replay_orphans} replay_unbound_returns={replay_unbound}"
    )

    for arm_name, arm in (("FAST", fast), ("REPLAY", replay)):
        for event in arm.events:
            shape = event_shape(arm, event)
            print(
                "FAST_COMMIT_VERIFY_LOGITS_EVENT_BINDING "
                f"arm={arm_name} commit_event_ordinal1={event.ordinal1} "
                f"verify_event_ordinal1={event.verify_event_ordinal1 if event.verify_event_ordinal1 is not None else 'NONE'} "
                f"token_start0={event.token_start0} returned={event.returned} "
                f"commit_type={event.commit_type} drafted={event.drafted} "
                f"accepted_drafts={event.accepted_drafts} "
                f"verify_start={shape[0] if shape else 'NONE'} "
                f"n_tokens={shape[1] if shape else 'NONE'}"
            )

    if causal_fast is not None:
        print(
            "FAST_COMMIT_VERIFY_LOGITS_CAUSAL_ANCHOR "
            f"token_start0={causal_fast.token_start0} "
            f"fast_commit_event_ordinal1={causal_fast.ordinal1} "
            f"fast_verify_event_ordinal1={causal_fast.verify_event_ordinal1 if causal_fast.verify_event_ordinal1 is not None else 'NONE'} "
            f"replay_commit_event_ordinal1={causal_replay.ordinal1 if causal_replay else 'NONE'} "
            f"replay_verify_event_ordinal1={causal_replay.verify_event_ordinal1 if causal_replay and causal_replay.verify_event_ordinal1 is not None else 'NONE'} "
            f"ids_equal={1 if causal_replay and causal_fast.ids == causal_replay.ids else 0} "
            f"shape_equal={1 if causal_fast_shape is not None and causal_fast_shape == causal_replay_shape else 0} "
            f"result={'PASS' if causal_anchor_ok else 'FAIL'}"
        )

    common_positions = sorted(set(fast_by_pos) & set(replay_by_pos))
    first_shape: tuple[int, int, int] | None = None
    first_hash: tuple[int, int, int, int] | None = None
    first_argmax: tuple[int, int, int, int] | None = None
    compared_rows = 0
    causal_precommit_hash_div = False

    for token_start0 in common_positions:
        fe = fast_by_pos[token_start0]
        revent = replay_by_pos[token_start0]
        fg = group_for_event(fast, fe)
        rg = group_for_event(replay, revent)
        if fg is None or rg is None:
            continue
        fshape = fg.primary_shape()
        rshape = rg.primary_shape()
        if fshape is None or rshape is None:
            continue
        same_shape = fshape == rshape
        if first_shape is None and not same_shape:
            first_shape = (token_start0, fe.ordinal1, revent.ordinal1)

        frows = fg.primary_rows()
        rrows = rg.primary_rows()
        for row_idx in sorted(set(frows) & set(rrows)):
            fr = frows[row_idx]
            rr = rrows[row_idx]
            compared_rows += 1
            hash_same = fr.hash_hex == rr.hash_hex
            argmax_same = fr.top1 == rr.top1
            if first_hash is None and not hash_same:
                first_hash = (
                    token_start0, fe.ordinal1, revent.ordinal1, row_idx
                )
            if first_argmax is None and not argmax_same:
                first_argmax = (
                    token_start0, fe.ordinal1, revent.ordinal1, row_idx
                )
            if (
                causal_fast is not None
                and token_start0 <= causal_fast.token_start0
                and not hash_same
            ):
                causal_precommit_hash_div = True

            if (
                token_start0 == (
                    causal_fast.token_start0 if causal_fast else -1
                )
                or not same_shape
                or not hash_same
                or not argmax_same
            ):
                print(
                    "FAST_COMMIT_VERIFY_LOGITS_ALIGNED_ROW "
                    f"token_start0={token_start0} "
                    f"fast_event_ordinal1={fe.ordinal1} "
                    f"replay_event_ordinal1={revent.ordinal1} row={row_idx} "
                    f"fast_start={fr.start} replay_start={rr.start} "
                    f"fast_n_tokens={fr.n_tokens} replay_n_tokens={rr.n_tokens} "
                    f"shape_result={'EXACT' if same_shape else 'MISMATCH'} "
                    f"hash_result={'EXACT' if hash_same else 'MISMATCH'} "
                    f"argmax_result={'EXACT' if argmax_same else 'MISMATCH'} "
                    f"fast_top1_id={fr.top1} replay_top1_id={rr.top1} "
                    f"fast_top2_id={fr.top2} replay_top2_id={rr.top2} "
                    f"fast_margin={fr.margin:.17g} replay_margin={rr.margin:.17g}"
                )

    if first_hash is None:
        print("FAST_COMMIT_VERIFY_LOGITS_FIRST_HASH_DIVERGENCE=NONE")
    else:
        print(
            "FAST_COMMIT_VERIFY_LOGITS_FIRST_HASH_DIVERGENCE "
            f"token_start0={first_hash[0]} "
            f"fast_event_ordinal1={first_hash[1]} "
            f"replay_event_ordinal1={first_hash[2]} row={first_hash[3]}"
        )
    if first_argmax is None:
        print("FAST_COMMIT_VERIFY_LOGITS_FIRST_ARGMAX_DIVERGENCE=NONE")
    else:
        print(
            "FAST_COMMIT_VERIFY_LOGITS_FIRST_ARGMAX_DIVERGENCE "
            f"token_start0={first_argmax[0]} "
            f"fast_event_ordinal1={first_argmax[1]} "
            f"replay_event_ordinal1={first_argmax[2]} row={first_argmax[3]}"
        )
    if first_shape is None:
        print("FAST_COMMIT_VERIFY_LOGITS_FIRST_SHAPE_DIVERGENCE=NONE")
    else:
        print(
            "FAST_COMMIT_VERIFY_LOGITS_FIRST_SHAPE_DIVERGENCE "
            f"token_start0={first_shape[0]} "
            f"fast_event_ordinal1={first_shape[1]} "
            f"replay_event_ordinal1={first_shape[2]}"
        )

    rejection_match = False
    rejection_prefix_equal = False
    rejection_shape_equal = False
    if first_argmax is not None:
        token_start0, fast_ord, replay_ord, row_idx = first_argmax
        fe = fast.events[fast_ord - 1]
        revent = replay.events[replay_ord - 1]
        expected_reject_row = (
            fe.accepted_drafts - 1
            if fe.correction and 0 < fe.accepted_drafts < fe.drafted
            else None
        )
        rejection_match = expected_reject_row == row_idx
        prefix_n = fe.accepted_drafts
        rejection_prefix_equal = (
            prefix_n > 0
            and len(fe.ids) >= prefix_n
            and len(revent.ids) >= prefix_n
            and fe.ids[:prefix_n] == revent.ids[:prefix_n]
        )
        fshape = event_shape(fast, fe)
        rshape = event_shape(replay, revent)
        rejection_shape_equal = fshape is not None and fshape == rshape
        print(
            "FAST_COMMIT_VERIFY_LOGITS_REJECTION_MAPPING "
            f"token_start0={token_start0} row={row_idx} "
            f"fast_commit_event_ordinal1={fast_ord} "
            f"replay_commit_event_ordinal1={replay_ord} "
            f"fast_commit_type={fe.commit_type} "
            f"fast_drafted={fe.drafted} "
            f"fast_accepted_drafts={fe.accepted_drafts} "
            f"expected_rejection_row={expected_reject_row if expected_reject_row is not None else 'NONE'} "
            f"accepted_prefix_equal={1 if rejection_prefix_equal else 0} "
            f"verify_shape_equal={1 if rejection_shape_equal else 0} "
            f"rejection_row_match={1 if rejection_match else 0}"
        )

    controls_ok = not failed
    first_argmax_after_causal = (
        first_argmax is not None
        and causal_fast is not None
        and first_argmax[0] > causal_fast.token_start0
    )
    if not controls_ok:
        source = "INCONCLUSIVE_CONTROL"
        next_step = "FIX_LISTED_CONTROL"
    elif causal_precommit_hash_div:
        source = "INCONCLUSIVE_PRECOMMIT_LOGIT_DIVERGENCE"
        next_step = "ADJUDICATE_LOG_BINDING_OR_VERIFIER_DETERMINISM"
    elif (
        first_shape is not None
        and causal_fast is not None
        and first_shape[0] > causal_fast.token_start0
    ):
        source = "NEXT_ITERATION_VERIFY_SHAPE_DIVERGENCE"
        next_step = "DRAFT_BLOCK_AND_VERIFY_INPUT_AB"
    elif (
        first_argmax_after_causal
        and rejection_match
        and rejection_prefix_equal
        and rejection_shape_equal
    ):
        source = "NEXT_VERIFY_REJECTION_ARGMAX_LOCALIZED"
        next_step = "LAYERWISE_NEXT_ITERATION_REJECTION_ROW_CP_AB"
    elif (
        first_argmax_after_causal
        and rejection_match
        and rejection_prefix_equal
    ):
        source = "NEXT_VERIFY_REJECTION_ARGMAX_WITH_SHAPE_CHANGE"
        next_step = "DRAFT_BLOCK_SHAPE_CAUSAL_AB"
    elif first_argmax is not None:
        source = "VERIFY_ARGMAX_DIVERGENCE_FOUND"
        next_step = "DRAFT_BLOCK_AND_ROW_SEMANTICS_AB"
    elif first_hash is not None:
        source = "VERIFY_NUMERICAL_DIVERGENCE_ONLY"
        next_step = "VERIFY_INPUT_EQUIVALENCE_AB"
    else:
        source = "VERIFY_LOGITS_EXACT_ON_ALIGNED_EVENTS"
        next_step = "NON_LOGIT_CORRECTION_STATE_AB"

    print(f"FAST_COMMIT_VERIFY_LOGITS_COMPARED_ROWS={compared_rows}")
    print(
        "FAST_COMMIT_VERIFY_LOGITS_CONTROL_FAILED="
        f"{','.join(failed) if failed else 'NONE'}"
    )
    print(
        "FAST_COMMIT_VERIFY_LOGITS_CONTROL_VERDICT="
        f"{'CONTROL_PASS' if controls_ok else 'CONTROL_FAIL'}"
    )
    print(f"FAST_COMMIT_VERIFY_LOGITS_SOURCE={source}")
    print(f"FAST_COMMIT_VERIFY_LOGITS_NEXT={next_step}")


if __name__ == "__main__":
    main()
