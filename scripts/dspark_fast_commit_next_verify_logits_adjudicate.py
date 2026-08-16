#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

ROW_RE = re.compile(
    r"^DS4_DSPARK_E9_LOGITS event_ordinal1=(\d+) verify_call_seq=(\d+) "
    r"start=(\d+) n_tokens=(\d+) row=(\d+) hash=([0-9a-fA-F]+) "
    r"verify_top_id=(-?\d+) top1_id=(\d+) .* status=(\S+)$"
)


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


def parse_rows(path: Path) -> list[tuple[int, int, int, int, int, str]]:
    out: list[tuple[int, int, int, int, int, str]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = ROW_RE.match(raw.strip())
        if not m:
            continue
        out.append(
            (
                int(m.group(1)),
                int(m.group(2)),
                int(m.group(5)),
                int(m.group(7)),
                int(m.group(8)),
                m.group(9),
            )
        )
    return out


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

    gates = {
        "baseline": field(baseline, "result") == "PASS",
        "oracle": field(oracle, "result") == "PASS",
        "trace_control": field(trace_control, "result") == "PASS",
        "alignment": field(trace, "event_alignment") == "PASS",
        "self_top": field(trace, "verify_top_self_mismatches") == "0",
        "causal_window": field(causal, "result") == "PASS",
    }

    failed = [name for name, ok in gates.items() if not ok]
    print(
        "FAST_COMMIT_VERIFY_LOGITS_CONTROL_DETAIL "
        + " ".join(f"{name}={'PASS' if ok else 'FAIL'}" for name, ok in gates.items())
    )
    print(
        "FAST_COMMIT_VERIFY_LOGITS_ALIGNMENT_DETAIL "
        f"fast_rows={field(trace, 'fast_rows') or 'MISSING'} "
        f"replay_rows={field(trace, 'replay_rows') or 'MISSING'} "
        f"fast_events={field(trace, 'fast_events') or 'MISSING'} "
        f"replay_events={field(trace, 'replay_events') or 'MISSING'} "
        f"fast_extra_verify_calls={field(trace, 'fast_extra_verify_calls') or 'MISSING'} "
        f"replay_extra_verify_calls={field(trace, 'replay_extra_verify_calls') or 'MISSING'} "
        f"verify_top_self_mismatches={field(trace, 'verify_top_self_mismatches') or 'MISSING'}"
    )

    fast_rows = parse_rows(fast_log)
    replay_rows = parse_rows(replay_log)
    bad_rows = []
    for arm, rows in (("FAST", fast_rows), ("REPLAY", replay_rows)):
        for event_ord, call_seq, row, verify_top, top1, status in rows:
            if status != "PASS" or (verify_top >= 0 and verify_top != top1):
                bad_rows.append((arm, event_ord, call_seq, row, verify_top, top1, status))

    for arm, event_ord, call_seq, row, verify_top, top1, status in bad_rows[:12]:
        print(
            "FAST_COMMIT_VERIFY_LOGITS_CONTROL_BAD_ROW "
            f"arm={arm} event_ordinal1={event_ord} verify_call_seq={call_seq} row={row} "
            f"verify_top_id={verify_top} readback_top1_id={top1} status={status}"
        )

    if not failed:
        verdict = "CONTROL_PASS"
        next_step = "USE_EXISTING_E9_NUMERICAL_RESULT"
    elif failed == ["self_top"]:
        verdict = "SELF_TOP_CONTRACT_MISMATCH_ONLY"
        next_step = "ADJUDICATE_ROW_TOP_SEMANTICS"
    elif failed == ["alignment"]:
        verdict = "EVENT_ALIGNMENT_ONLY"
        next_step = "FIX_EVENT_TO_VERIFY_MAPPING"
    elif failed == ["trace_control"]:
        verdict = "READBACK_PERTURBS_TRAJECTORY"
        next_step = "REDESIGN_NONPERTURBATIVE_CAPTURE"
    elif any(x in failed for x in ("baseline", "oracle")):
        verdict = "RUN_CONTROL_INVALID"
        next_step = "RERUN_BASELINE_OR_REPLAY"
    else:
        verdict = "MULTIPLE_CONTROL_FAILURES"
        next_step = "FIX_LISTED_GATES"

    print(f"FAST_COMMIT_VERIFY_LOGITS_CONTROL_FAILED={','.join(failed) if failed else 'NONE'}")
    print(f"FAST_COMMIT_VERIFY_LOGITS_CONTROL_VERDICT={verdict}")
    print(f"FAST_COMMIT_VERIFY_LOGITS_CONTROL_NEXT={next_step}")


if __name__ == "__main__":
    main()
