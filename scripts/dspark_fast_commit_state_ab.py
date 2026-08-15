#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

STATE_PREFIX = "DS4_DSPARK_STATE_RETURN"
LAYER_PREFIX = "DS4_DSPARK_STATE_LAYERS"
TARGET_FN = "static int ds4_session_eval_dspark_speculative_argmax("
FAST_MARKER = "fast commit, no replay"


def die(msg: str) -> None:
    raise SystemExit(f"dspark_fast_commit_state_ab: {msg}")


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


def state_block(indent: str, path: str) -> str:
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
        f'{indent}    const char *ds4_state_env = getenv("DS4_DSPARK_TRACE_STATE_AB");\n'
        f'{indent}    if (ds4_state_env && ds4_state_env[0] && strcmp(ds4_state_env, "0") != 0) {{\n'
        f'{indent}        const char *ds4_state_commit_type = {commit_type_expr};\n'
        f'{indent}        const int ds4_state_checkpoint_last = s->checkpoint.len > 0 ? s->checkpoint.v[s->checkpoint.len - 1] : -1;\n'
        f'{indent}        const int ds4_state_checkpoint_prev = s->checkpoint.len > 1 ? s->checkpoint.v[s->checkpoint.len - 2] : -1;\n'
        f'{indent}        fprintf(stderr, "{STATE_PREFIX} path={path} commit_type=%s drafted=%d accepted_drafts=%d returned=%d start=%u checkpoint_len=%d checkpoint_valid=%d checkpoint_last=%d checkpoint_prev=%d mtp_draft_valid=%d dspark_draft_len=%u dspark_draft_valid=%d mtp_n_raw=%u dspark_cache_start=%u dspark_cache_token_start=%u dspark_cache_len=%u sched_cycles=%u sched_accepted=%u sched_no_draft=%u sched_skip=%u sched_lifetime_accepted=%u ids=",\n'
        f'{indent}                ds4_state_commit_type, draft_n, commit_drafts, n_accept, (unsigned)start, s->checkpoint.len, (int)s->checkpoint_valid, ds4_state_checkpoint_last, ds4_state_checkpoint_prev, (int)s->mtp_draft_valid, s->dspark_draft_len, (int)s->dspark_draft_valid, s->graph.mtp_n_raw, s->graph.dspark_cache_start, s->graph.dspark_cache_token_start, s->graph.dspark_cache_len, s->dspark_sched_cycles, s->dspark_sched_accepted, s->dspark_sched_no_draft, s->dspark_sched_skip, s->dspark_sched_lifetime_accepted);\n'
        f'{indent}        for (int ds4_state_i = 0; ds4_state_i < n_accept; ds4_state_i++) {{\n'
        f'{indent}            fprintf(stderr, "%s%u", ds4_state_i ? "," : "", (unsigned)accepted[ds4_state_i]);\n'
        f'{indent}        }}\n'
        f'{indent}        fputc(\'\\n\', stderr);\n'
        f'{indent}        fprintf(stderr, "{LAYER_PREFIX} path={path} commit_type=%s n_comp=", ds4_state_commit_type);\n'
        f'{indent}        for (uint32_t ds4_state_il = 0; ds4_state_il < DS4_N_LAYER; ds4_state_il++) {{\n'
        f'{indent}            fprintf(stderr, "%s%u", ds4_state_il ? "," : "", s->graph.layer_n_comp[ds4_state_il]);\n'
        f'{indent}        }}\n'
        f'{indent}        fputs(" n_index=", stderr);\n'
        f'{indent}        for (uint32_t ds4_state_il = 0; ds4_state_il < DS4_N_LAYER; ds4_state_il++) {{\n'
        f'{indent}            fprintf(stderr, "%s%u", ds4_state_il ? "," : "", s->graph.layer_n_index_comp[ds4_state_il]);\n'
        f'{indent}        }}\n'
        f'{indent}        fputc(\'\\n\', stderr);\n'
        f'{indent}    }}\n'
        f'{indent}}}\n'
        f'{indent}return n_accept;'
    )


def instrument(source: Path) -> None:
    text = source.read_text(encoding="utf-8")
    if STATE_PREFIX in text:
        die(f"{source} is already state-instrumented")
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
    if re.search(r"\b(?:int|uint32_t|size_t)\s+start\b", fn) is None:
        die("start position variable not found; source layout changed")
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
        pieces.append(state_block(m.group(1), path))
        labels.append(path)
        cursor = m.end()
    pieces.append(fn[cursor:])
    patched_fn = "".join(pieces)
    patched = text[:fn_start] + patched_fn + text[fn_end + 1 :]
    source.write_text(patched, encoding="utf-8")
    print(
        f"FAST_COMMIT_STATE_INSTRUMENT status=PASS return_sites={len(eligible)} "
        f"fast_sites={labels.count('fast_full')} replay_sites={labels.count('replay_or_partial')}"
    )


@dataclass
class StateEvent:
    ordinal: int
    path: str
    commit_type: str
    drafted: int
    accepted_drafts: int
    returned: int
    fields: dict[str, int]
    ids: tuple[int, ...]
    n_comp: tuple[int, ...] = field(default_factory=tuple)
    n_index: tuple[int, ...] = field(default_factory=tuple)


STATE_RE = re.compile(
    rf"^{STATE_PREFIX} path=(\S+) commit_type=(\S+) drafted=(\d+) "
    r"accepted_drafts=(\d+) returned=(\d+) (.*?) ids=(.*)$"
)
LAYER_RE = re.compile(
    rf"^{LAYER_PREFIX} path=(\S+) commit_type=(\S+) n_comp=([^ ]*) n_index=(.*)$"
)


def parse_int_list(text: str) -> tuple[int, ...]:
    if not text:
        return ()
    return tuple(int(x) for x in text.split(",") if x)


def parse_events(log: Path) -> list[StateEvent]:
    events: list[StateEvent] = []
    pending_layer = 0
    for raw in log.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        m = STATE_RE.match(line)
        if m:
            fields: dict[str, int] = {}
            for key, value in re.findall(r"([A-Za-z0-9_]+)=(-?\d+)", m.group(6)):
                fields[key] = int(value)
            events.append(
                StateEvent(
                    ordinal=len(events),
                    path=m.group(1),
                    commit_type=m.group(2),
                    drafted=int(m.group(3)),
                    accepted_drafts=int(m.group(4)),
                    returned=int(m.group(5)),
                    fields=fields,
                    ids=parse_int_list(m.group(7)),
                )
            )
            continue
        lm = LAYER_RE.match(line)
        if lm:
            if pending_layer >= len(events):
                die(f"layer state without preceding scalar state in {log}")
            event = events[pending_layer]
            if event.path != lm.group(1) or event.commit_type != lm.group(2):
                die(f"layer/scalar state ordering mismatch in {log}")
            event.n_comp = parse_int_list(lm.group(3))
            event.n_index = parse_int_list(lm.group(4))
            pending_layer += 1
    if not events:
        die(f"no {STATE_PREFIX} events found in {log}")
    if pending_layer != len(events):
        die(f"missing layer state lines in {log}: scalar={len(events)} layer={pending_layer}")
    return events


def event_signature(e: StateEvent) -> tuple[int, int, tuple[int, ...]]:
    return e.drafted, e.accepted_drafts, e.ids


def match_full_accepts(fast: list[StateEvent], replay: list[StateEvent]) -> list[tuple[StateEvent, StateEvent]]:
    fast_full = [e for e in fast if e.commit_type == "FULL_ACCEPT_FAST"]
    replay_full = [e for e in replay if e.commit_type == "FULL_ACCEPT_REPLAY"]
    pairs: list[tuple[StateEvent, StateEvent]] = []
    cursor = 0
    for f in fast_full:
        sig = event_signature(f)
        hit = None
        for j in range(cursor, len(replay_full)):
            if event_signature(replay_full[j]) == sig:
                hit = j
                break
        if hit is None:
            continue
        pairs.append((f, replay_full[hit]))
        cursor = hit + 1
    return pairs


def first_array_diff(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int | None, int]:
    n = min(len(a), len(b))
    count = sum(1 for i in range(n) if a[i] != b[i]) + abs(len(a) - len(b))
    first = next((i for i in range(n) if a[i] != b[i]), None)
    if first is None and len(a) != len(b):
        first = n
    return first, count


def analyze(fast_log: Path, replay_log: Path) -> None:
    fast_events = parse_events(fast_log)
    replay_events = parse_events(replay_log)
    pairs = match_full_accepts(fast_events, replay_events)
    print(
        f"FAST_COMMIT_STATE_TRACE fast_events={len(fast_events)} replay_events={len(replay_events)} "
        f"matched_full_accepts={len(pairs)}"
    )
    if not pairs:
        print("FAST_COMMIT_METADATA_DIVERGENCE=INCONCLUSIVE reason=no_matching_full_accept")
        return

    first_div: tuple[int, str] | None = None
    for pair_idx, (fast, replay) in enumerate(pairs):
        scalar_diffs = [
            key
            for key in sorted(set(fast.fields) | set(replay.fields))
            if fast.fields.get(key) != replay.fields.get(key)
        ]
        comp_first, comp_count = first_array_diff(fast.n_comp, replay.n_comp)
        index_first, index_count = first_array_diff(fast.n_index, replay.n_index)
        mismatch_fields = list(scalar_diffs)
        if comp_count:
            mismatch_fields.append("layer_n_comp")
        if index_count:
            mismatch_fields.append("layer_n_index_comp")
        result = "MISMATCH" if mismatch_fields else "EXACT"
        print(
            f"FAST_COMMIT_STATE_AB pair={pair_idx} fast_event={fast.ordinal} replay_event={replay.ordinal} "
            f"drafted={fast.drafted} accepted_drafts={fast.accepted_drafts} result={result} "
            f"mismatch_fields={','.join(mismatch_fields) if mismatch_fields else 'NONE'}"
        )
        for key in scalar_diffs:
            print(
                f"FAST_COMMIT_STATE_DIFF pair={pair_idx} field={key} "
                f"fast={fast.fields.get(key, 'MISSING')} replay={replay.fields.get(key, 'MISSING')}"
            )
            if first_div is None:
                first_div = (pair_idx, key)
        if comp_count:
            fv = fast.n_comp[comp_first] if comp_first is not None and comp_first < len(fast.n_comp) else "MISSING"
            rv = replay.n_comp[comp_first] if comp_first is not None and comp_first < len(replay.n_comp) else "MISSING"
            print(
                f"FAST_COMMIT_STATE_DIFF pair={pair_idx} field=layer_n_comp "
                f"first_layer={comp_first} mismatch_count={comp_count} fast={fv} replay={rv}"
            )
            if first_div is None:
                first_div = (pair_idx, "layer_n_comp")
        if index_count:
            fv = fast.n_index[index_first] if index_first is not None and index_first < len(fast.n_index) else "MISSING"
            rv = replay.n_index[index_first] if index_first is not None and index_first < len(replay.n_index) else "MISSING"
            print(
                f"FAST_COMMIT_STATE_DIFF pair={pair_idx} field=layer_n_index_comp "
                f"first_layer={index_first} mismatch_count={index_count} fast={fv} replay={rv}"
            )
            if first_div is None:
                first_div = (pair_idx, "layer_n_index_comp")

    if first_div is None:
        print("FAST_COMMIT_FIRST_STATE_DIVERGENCE=NONE")
        print("FAST_COMMIT_METADATA_DIVERGENCE=NONE")
        print("FAST_COMMIT_STATE_NEXT=CACHE_CONTENT_AB")
    else:
        print(f"FAST_COMMIT_FIRST_STATE_DIVERGENCE pair={first_div[0]} field={first_div[1]}")
        print("FAST_COMMIT_METADATA_DIVERGENCE=FOUND")
        print("FAST_COMMIT_STATE_NEXT=MINIMAL_METADATA_REPAIR_AB")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_inst = sub.add_parser("instrument")
    p_inst.add_argument("--source", required=True, type=Path)
    p_an = sub.add_parser("analyze")
    p_an.add_argument("--fast-log", required=True, type=Path)
    p_an.add_argument("--replay-log", required=True, type=Path)
    args = ap.parse_args()
    if args.cmd == "instrument":
        instrument(args.source)
    else:
        analyze(args.fast_log.expanduser().resolve(), args.replay_log.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
