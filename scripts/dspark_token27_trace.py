#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

TRACE_PREFIX = "DS4_DSPARK_TRACE_RETURN"
TARGET_FN = "static int ds4_session_eval_dspark_speculative_argmax("
FAST_MARKER = "fast commit, no replay"


def die(msg: str) -> None:
    raise SystemExit(f"dspark_token27_trace: {msg}")


def fmt_diff(index0: int | None) -> str:
    return "NONE" if index0 is None else str(index0)


def fmt_ordinal(index0: int | None) -> str:
    return "NONE" if index0 is None else str(index0 + 1)


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
    die("could not find end of DSpark speculative function")


def trace_block(indent: str, path: str) -> str:
    if path == "fast_full":
        commit_type_expr = '"FULL_ACCEPT_FAST"'
    else:
        commit_type_expr = (
            '(commit_drafts == draft_n ? "FULL_ACCEPT_REPLAY" : '
            '(commit_drafts == 0 && n_accept > 0 ? "ZERO_ACCEPT_CORRECTION" : '
            '(n_accept > commit_drafts ? "PARTIAL_ACCEPT_WITH_CORRECTION" : '
            '"PARTIAL_ACCEPT")))'
        )
    return (
        f'{indent}{{\n'
        f'{indent}    const char *ds4_trace_env = getenv("DS4_DSPARK_TRACE_COMMITS");\n'
        f'{indent}    if (ds4_trace_env && ds4_trace_env[0] && strcmp(ds4_trace_env, "0") != 0) {{\n'
        f'{indent}        const char *ds4_trace_commit_type = {commit_type_expr};\n'
        f'{indent}        const int ds4_trace_correction = n_accept > commit_drafts ? 1 : 0;\n'
        f'{indent}        fprintf(stderr, "{TRACE_PREFIX} path={path} commit_type=%s drafted=%d accepted_drafts=%d returned=%d correction=%d ids=",\n'
        f'{indent}                ds4_trace_commit_type, draft_n, commit_drafts, n_accept, ds4_trace_correction);\n'
        f'{indent}        for (int ds4_trace_i = 0; ds4_trace_i < n_accept; ds4_trace_i++) {{\n'
        f'{indent}            fprintf(stderr, "%s%u", ds4_trace_i ? "," : "", (unsigned)accepted[ds4_trace_i]);\n'
        f'{indent}        }}\n'
        f'{indent}        fputc(\'\\n\', stderr);\n'
        f'{indent}    }}\n'
        f'{indent}}}\n'
        f'{indent}return n_accept;'
    )


def instrument(source: Path) -> None:
    text = source.read_text(encoding="utf-8")
    if TRACE_PREFIX in text:
        die(f"{source} is already instrumented")
    fn_start = text.find(TARGET_FN)
    if fn_start < 0:
        die(f"target function not found in {source}")
    open_pos = text.find("{", fn_start)
    if open_pos < 0:
        die("target function opening brace not found")
    fn_end = find_matching_brace(text, open_pos)
    fn = text[fn_start : fn_end + 1]

    decl = re.search(r"\bint\s+commit_drafts\b", fn)
    if not decl:
        die("commit_drafts declaration not found; source layout changed")
    fast_marker_pos = fn.find(FAST_MARKER)
    if fast_marker_pos < 0:
        die("fast-commit marker not found; wrong source branch or hook missing")

    returns = list(re.finditer(r"(?m)^(\s*)return n_accept;\s*$", fn))
    eligible = [m for m in returns if m.start() > decl.start()]
    if not eligible:
        die("no post-verification return n_accept sites found")
    fast_returns = [m for m in eligible if m.start() > fast_marker_pos]
    if not fast_returns:
        die("fast-commit return site not found")
    fast_return = fast_returns[0]

    pieces: list[str] = []
    cursor = 0
    labels: list[str] = []
    for m in eligible:
        path = "fast_full" if m.start() == fast_return.start() else "replay_or_partial"
        pieces.append(fn[cursor : m.start()])
        pieces.append(trace_block(m.group(1), path))
        labels.append(path)
        cursor = m.end()
    pieces.append(fn[cursor:])
    patched_fn = "".join(pieces)
    patched = text[:fn_start] + patched_fn + text[fn_end + 1 :]
    source.write_text(patched, encoding="utf-8")
    print(
        f"TOKEN27_INSTRUMENT status=PASS return_sites={len(eligible)} "
        f"fast_sites={labels.count('fast_full')} replay_sites={labels.count('replay_or_partial')}"
    )


def parse_token_ids(text: str) -> list[int]:
    lines = text.splitlines()
    for line in lines:
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
    p = subprocess.run(
        [
            str(ds4),
            "-m",
            str(model),
            "--raw-prompt",
            "--dump-tokens",
            "--prompt-file",
            str(text_file),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if p.returncode != 0:
        die(f"tokenizer failed for {text_file}: {p.stderr.strip()}")
    for stream in (p.stdout, p.stderr):
        ids = parse_token_ids(stream)
        if ids:
            return ids
    die(f"no token IDs parsed for {text_file}")


def first_divergence(a: Sequence[int], b: Sequence[int]) -> int | None:
    """Return zero-based generated-token index, or None through the horizon."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def extract_tps(log: Path) -> float | None:
    text = log.read_text(encoding="utf-8", errors="replace")
    vals = re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*t/s\b", text)
    return float(vals[-1]) if vals else None


@dataclass
class Event:
    iteration: int
    path: str
    commit_type: str
    drafted: int
    accepted_drafts: int
    returned: int
    correction: bool
    ids: list[int]
    token_start0: int | None = None
    token_end0: int | None = None


TRACE_RE = re.compile(
    rf"^{TRACE_PREFIX} path=(\S+) commit_type=(\S+) drafted=(\d+) "
    r"accepted_drafts=(\d+) returned=(\d+) correction=([01]) ids=(.*)$"
)


def parse_events(log: Path) -> list[Event]:
    events: list[Event] = []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = TRACE_RE.match(line.strip())
        if not m:
            continue
        ids = [int(x) for x in m.group(7).split(",") if x]
        returned = int(m.group(5))
        if len(ids) != returned:
            die(f"trace event returned={returned} but ids={len(ids)}")
        accepted_drafts = int(m.group(4))
        correction = m.group(6) == "1"
        if correction and returned <= accepted_drafts:
            die("trace event marks correction but has no correction return token")
        events.append(
            Event(
                iteration=len(events),
                path=m.group(1),
                commit_type=m.group(2),
                drafted=int(m.group(3)),
                accepted_drafts=accepted_drafts,
                returned=returned,
                correction=correction,
                ids=ids,
            )
        )
    if not events:
        die(f"no {TRACE_PREFIX} events found in {log}")
    return events


def find_contiguous(haystack: Sequence[int], needle: Sequence[int], start: int = 0) -> int | None:
    if not needle or len(needle) > len(haystack):
        return None
    first = needle[0]
    for i in range(start, len(haystack) - len(needle) + 1):
        if haystack[i] == first and list(haystack[i : i + len(needle)]) == list(needle):
            return i
    return None


def align_events(candidate: Sequence[int], events: list[Event]) -> str:
    flat = [tok for event in events for tok in event.ids]
    exact = find_contiguous(candidate, flat)
    if exact is not None:
        pos = exact
        for event in events:
            event.token_start0 = pos
            event.token_end0 = pos + event.returned - 1
            pos += event.returned
        return "EXACT_CONTIGUOUS"

    cursor = 0
    for event in events:
        hit = find_contiguous(candidate, event.ids, cursor)
        if hit is None:
            for e in events:
                e.token_start0 = None
                e.token_end0 = None
            return "UNRESOLVED"
        event.token_start0 = hit
        event.token_end0 = hit + event.returned - 1
        cursor = hit + event.returned
    return "ORDERED_WITH_GAPS"


def event_for_token(events: Sequence[Event], index0: int) -> Event | None:
    return next(
        (
            e
            for e in events
            if e.token_start0 is not None
            and e.token_end0 is not None
            and e.token_start0 <= index0 <= e.token_end0
        ),
        None,
    )


def token_source(event: Event, index0: int) -> tuple[str, int, int | None]:
    assert event.token_start0 is not None
    offset = index0 - event.token_start0
    if offset < event.accepted_drafts:
        return "DRAFT_ACCEPTED", offset, None
    if event.correction and offset == event.accepted_drafts:
        corr = event.ids[offset] if offset < len(event.ids) else None
        return "CORRECTION", offset, corr
    return "SEQUENTIAL_OR_OTHER", offset, None


def print_provenance(candidate_ids: Sequence[int], diff0: int | None, log: Path, label: str) -> tuple[list[Event], Event | None]:
    events = parse_events(log)
    mapping = align_events(candidate_ids, events)
    print(
        f"TOKEN27_TRACE arm={label} events={len(events)} "
        f"traced_tokens={sum(e.returned for e in events)} mapping={mapping}"
    )
    if diff0 is None or mapping == "UNRESOLVED":
        status = "NO_DIVERGENCE" if diff0 is None else "UNRESOLVED"
        print(f"TOKEN27_PROVENANCE arm={label} status={status}")
        return events, None

    hit = event_for_token(events, diff0)
    if hit is None:
        print(
            f"TOKEN27_PROVENANCE arm={label} status=UNTRACED "
            f"token_index0={diff0} token_ordinal1={diff0 + 1}"
        )
        return events, None

    source, offset, correction_id = token_source(hit, diff0)
    corr = "NONE" if correction_id is None else str(correction_id)
    print(
        f"TOKEN27_PROVENANCE arm={label} status=LOCALIZED "
        f"token_index0={diff0} token_ordinal1={diff0 + 1} "
        f"iteration={hit.iteration} block_offset={offset} source={source} "
        f"commit_type={hit.commit_type} drafted={hit.drafted} "
        f"accepted_drafts={hit.accepted_drafts} returned={hit.returned} "
        f"correction_token_id={corr}"
    )
    prior_fast = [
        e
        for e in events
        if e.commit_type == "FULL_ACCEPT_FAST"
        and e.token_end0 is not None
        and e.token_end0 < diff0
    ]
    if prior_fast:
        p = prior_fast[-1]
        print(
            f"TOKEN27_PRECEDING_FAST_FULL iteration={p.iteration} "
            f"token_start0={p.token_start0} token_end0={p.token_end0} "
            f"distance_tokens={diff0 - int(p.token_end0)}"
        )
    return events, hit


def pct_delta(new: float | None, old: float | None) -> str:
    if new is None or old is None or old == 0:
        return "NA"
    return f"{((new / old) - 1.0) * 100.0:.3f}"


def analyze_ab(
    ds4: Path,
    model: Path,
    oracle: Path,
    e0_out: Path,
    e0_log: Path,
    e1_out: Path,
    e1_log: Path,
    e2_out: Path,
    e2_log: Path,
    expected_ordinal1: int,
) -> None:
    oracle_ids = tokenize(ds4, model, oracle)
    e0_ids = tokenize(ds4, model, e0_out)
    e1_ids = tokenize(ds4, model, e1_out)
    e2_ids = tokenize(ds4, model, e2_out)
    d0 = first_divergence(oracle_ids, e0_ids)
    d1 = first_divergence(oracle_ids, e1_ids)
    d2 = first_divergence(oracle_ids, e2_ids)
    t0 = extract_tps(e0_log)
    t1 = extract_tps(e1_log)
    t2 = extract_tps(e2_log)

    print("TOKEN_INDEX_CONVENTION generated_zero_based prompt_excluded=1")
    reproduced = d0 is not None and d0 + 1 == expected_ordinal1
    print(
        f"TOKEN27_BASELINE first_diff_index0={fmt_diff(d0)} "
        f"first_diff_ordinal1={fmt_ordinal(d0)} expected_ordinal1={expected_ordinal1} "
        f"generation_tps={'NA' if t0 is None else f'{t0:.3f}'} "
        f"result={'PASS' if reproduced else 'FAIL'}"
    )

    trace_exact = e0_ids == e1_ids
    frontier_same = d0 == d1
    trace_ok = trace_exact and frontier_same
    print(
        f"TRACE_CONTROL token_output={'EXACT' if trace_exact else 'MISMATCH'} "
        f"frontier_same={int(frontier_same)} "
        f"e0_first_diff_index0={fmt_diff(d0)} e1_first_diff_index0={fmt_diff(d1)} "
        f"e0_tps={'NA' if t0 is None else f'{t0:.3f}'} "
        f"e1_tps={'NA' if t1 is None else f'{t1:.3f}'} "
        f"trace_tps_delta_pct={pct_delta(t1, t0)} result={'PASS' if trace_ok else 'FAIL'}"
    )

    _, hit = print_provenance(e1_ids, d1, e1_log, "E1_FAST_TRACE")
    print_provenance(e2_ids, d2, e2_log, "E2_FULL_ACCEPT_REPLAY")

    if not trace_ok or d1 is None:
        cause = "INCONCLUSIVE"
    elif d2 is None or d2 > d1:
        cause = "PROVEN"
    elif d2 == d1:
        cause = "REJECTED"
    else:
        cause = "INCONCLUSIVE"

    immediate = hit.commit_type if hit is not None else "UNRESOLVED"
    print(
        f"FULL_ACCEPT_CAUSAL_AB fast_first_diff_index0={fmt_diff(d1)} "
        f"fast_first_diff_ordinal1={fmt_ordinal(d1)} "
        f"replay_first_diff_index0={fmt_diff(d2)} "
        f"replay_first_diff_ordinal1={fmt_ordinal(d2)} "
        f"fast_generation_tps={'NA' if t1 is None else f'{t1:.3f}'} "
        f"replay_generation_tps={'NA' if t2 is None else f'{t2:.3f}'} "
        f"replay_vs_fast_tps_delta_pct={pct_delta(t2, t1)} "
        f"first_diff_commit_type={immediate}"
    )
    print(f"FAST_FULL_COMMIT_CAUSE={cause}")


def analyze_single(ds4: Path, model: Path, oracle: Path, candidate: Path, log: Path) -> None:
    oracle_ids = tokenize(ds4, model, oracle)
    candidate_ids = tokenize(ds4, model, candidate)
    diff0 = first_divergence(oracle_ids, candidate_ids)
    print("TOKEN_INDEX_CONVENTION generated_zero_based prompt_excluded=1")
    print(
        f"TOKEN27_FRONTIER first_diff_index0={fmt_diff(diff0)} "
        f"first_diff_ordinal1={fmt_ordinal(diff0)} "
        f"oracle_tokens={len(oracle_ids)} candidate_tokens={len(candidate_ids)}"
    )
    print_provenance(candidate_ids, diff0, log, "candidate")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_inst = sub.add_parser("instrument")
    p_inst.add_argument("--source", required=True, type=Path)

    p_an = sub.add_parser("analyze")
    p_an.add_argument("--ds4-bin", required=True, type=Path)
    p_an.add_argument("--model", required=True, type=Path)
    p_an.add_argument("--oracle", required=True, type=Path)
    p_an.add_argument("--candidate", required=True, type=Path)
    p_an.add_argument("--log", required=True, type=Path)

    p_ab = sub.add_parser("analyze-ab")
    p_ab.add_argument("--ds4-bin", required=True, type=Path)
    p_ab.add_argument("--model", required=True, type=Path)
    p_ab.add_argument("--oracle", required=True, type=Path)
    p_ab.add_argument("--e0-out", required=True, type=Path)
    p_ab.add_argument("--e0-log", required=True, type=Path)
    p_ab.add_argument("--e1-out", required=True, type=Path)
    p_ab.add_argument("--e1-log", required=True, type=Path)
    p_ab.add_argument("--e2-out", required=True, type=Path)
    p_ab.add_argument("--e2-log", required=True, type=Path)
    p_ab.add_argument("--expected-ordinal1", type=int, default=27)

    args = ap.parse_args()
    if args.cmd == "instrument":
        instrument(args.source)
    elif args.cmd == "analyze":
        analyze_single(
            args.ds4_bin.expanduser().resolve(),
            args.model.expanduser().resolve(),
            args.oracle.expanduser().resolve(),
            args.candidate.expanduser().resolve(),
            args.log.expanduser().resolve(),
        )
    else:
        analyze_ab(
            args.ds4_bin.expanduser().resolve(),
            args.model.expanduser().resolve(),
            args.oracle.expanduser().resolve(),
            args.e0_out.expanduser().resolve(),
            args.e0_log.expanduser().resolve(),
            args.e1_out.expanduser().resolve(),
            args.e1_log.expanduser().resolve(),
            args.e2_out.expanduser().resolve(),
            args.e2_log.expanduser().resolve(),
            args.expected_ordinal1,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
