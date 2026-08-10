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
    return (
        f'{indent}{{\n'
        f'{indent}    const char *ds4_trace_env = getenv("DS4_DSPARK_TRACE_COMMITS");\n'
        f'{indent}    if (ds4_trace_env && ds4_trace_env[0] && strcmp(ds4_trace_env, "0") != 0) {{\n'
        f'{indent}        fprintf(stderr, "{TRACE_PREFIX} path={path} drafted=%d verified=%d accepted=%d full=%d ids=",\n'
        f'{indent}                draft_n, commit_drafts, n_accept, commit_drafts == draft_n ? 1 : 0);\n'
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
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i + 1
    if len(a) != len(b):
        return min(len(a), len(b)) + 1
    return None


@dataclass
class Event:
    index: int
    path: str
    drafted: int
    verified: int
    accepted: int
    full: bool
    ids: list[int]
    token_start: int | None = None
    token_end: int | None = None


TRACE_RE = re.compile(
    rf"^{TRACE_PREFIX} path=(\S+) drafted=(\d+) verified=(\d+) "
    r"accepted=(\d+) full=([01]) ids=(.*)$"
)


def parse_events(log: Path) -> list[Event]:
    events: list[Event] = []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = TRACE_RE.match(line.strip())
        if not m:
            continue
        ids = [int(x) for x in m.group(6).split(",") if x]
        accepted = int(m.group(4))
        if len(ids) != accepted:
            die(f"trace event accepted={accepted} but ids={len(ids)}")
        events.append(
            Event(
                index=len(events) + 1,
                path=m.group(1),
                drafted=int(m.group(2)),
                verified=int(m.group(3)),
                accepted=accepted,
                full=m.group(5) == "1",
                ids=ids,
            )
        )
    if not events:
        die(f"no {TRACE_PREFIX} events found in {log}")
    return events


def find_contiguous(haystack: Sequence[int], needle: Sequence[int]) -> int | None:
    if not needle or len(needle) > len(haystack):
        return None
    first = needle[0]
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i] == first and list(haystack[i : i + len(needle)]) == list(needle):
            return i
    return None


def analyze(ds4: Path, model: Path, oracle: Path, candidate: Path, log: Path) -> None:
    oracle_ids = tokenize(ds4, model, oracle)
    candidate_ids = tokenize(ds4, model, candidate)
    diff = first_divergence(oracle_ids, candidate_ids)
    events = parse_events(log)
    trace_ids = [tok for event in events for tok in event.ids]
    offset0 = find_contiguous(candidate_ids, trace_ids)
    mapping = "EXACT_CONTIGUOUS" if offset0 is not None else "UNRESOLVED"

    if offset0 is not None:
        pos = offset0 + 1
        for event in events:
            event.token_start = pos
            event.token_end = pos + event.accepted - 1
            pos += event.accepted

    print(
        f"TOKEN27_FRONTIER first_diff_token={'NONE' if diff is None else diff} "
        f"oracle_tokens={len(oracle_ids)} candidate_tokens={len(candidate_ids)}"
    )
    print(
        f"TOKEN27_TRACE events={len(events)} traced_tokens={len(trace_ids)} mapping={mapping}"
        + (f" trace_start_token={offset0 + 1}" if offset0 is not None else "")
    )

    if diff is None:
        print("TOKEN27_BOUNDARY status=NO_DIVERGENCE")
        return
    if offset0 is None:
        print(
            "TOKEN27_BOUNDARY status=UNRESOLVED "
            "reason=trace_token_ids_do_not_align_with_retokenized_candidate"
        )
        return

    hit = next(
        (
            e
            for e in events
            if e.token_start is not None
            and e.token_end is not None
            and e.token_start <= diff <= e.token_end
        ),
        None,
    )
    if hit is None:
        if diff < offset0 + 1:
            print(f"TOKEN27_BOUNDARY status=UNTRACED_PREFIX first_trace_token={offset0 + 1}")
        else:
            print(f"TOKEN27_BOUNDARY status=UNTRACED_SUFFIX last_trace_token={events[-1].token_end}")
        return

    lo = max(0, hit.index - 2)
    hi = min(len(events), hit.index + 1)
    for e in events[lo:hi]:
        role = "HIT" if e.index == hit.index else "NEARBY"
        print(
            f"TOKEN27_EVENT role={role} iteration={e.index} "
            f"token_start={e.token_start} token_end={e.token_end} "
            f"path={e.path} full={int(e.full)} drafted={e.drafted} "
            f"verified={e.verified} accepted={e.accepted}"
        )
    print(
        f"TOKEN27_BOUNDARY status=LOCALIZED iteration={hit.index} "
        f"token_start={hit.token_start} token_end={hit.token_end} "
        f"path={hit.path} full={int(hit.full)} drafted={hit.drafted} "
        f"verified={hit.verified} accepted={hit.accepted}"
    )


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
    args = ap.parse_args()
    if args.cmd == "instrument":
        instrument(args.source)
    else:
        analyze(
            args.ds4_bin.expanduser().resolve(),
            args.model.expanduser().resolve(),
            args.oracle.expanduser().resolve(),
            args.candidate.expanduser().resolve(),
            args.log.expanduser().resolve(),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
