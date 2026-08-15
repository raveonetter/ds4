#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

EVENT_PREFIX = "DS4_DSPARK_CONTENT_EVENT"
HASH_PREFIX = "DS4_DSPARK_CONTENT_HASH"

EVENT_RE = re.compile(
    rf"^{EVENT_PREFIX} path=(\S+) commit_type=(\S+) drafted=(\d+) accepted_drafts=(\d+) "
    r"returned=(\d+) start=(\d+) checkpoint_len=(\d+) ids=(.*)$"
)
HASH_RE = re.compile(
    rf"^{HASH_PREFIX} path=(\S+) commit_type=(\S+) scope=(\S+) layer=(-?\d+) "
    r"tensor=(\S+) bytes=(\d+) hash=([0-9a-fA-F]+) status=(\S+)$"
)


def die(msg: str) -> None:
    raise SystemExit(f"dspark_fast_commit_kv_repair_ab: {msg}")


@dataclass(frozen=True)
class Event:
    path: str
    commit_type: str
    drafted: int
    accepted_drafts: int
    returned: int
    start: int
    checkpoint_len: int
    ids: tuple[int, ...]


@dataclass(frozen=True)
class Record:
    scope: str
    layer: int
    tensor: str
    bytes: int
    hash: str
    status: str


def parse_ids(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(",") if x)


def parse_content_log(path: Path) -> tuple[Event, dict[tuple[str, int, str], Record]]:
    event: Event | None = None
    records: dict[tuple[str, int, str], Record] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        em = EVENT_RE.match(line)
        if em and event is None:
            event = Event(
                path=em.group(1),
                commit_type=em.group(2),
                drafted=int(em.group(3)),
                accepted_drafts=int(em.group(4)),
                returned=int(em.group(5)),
                start=int(em.group(6)),
                checkpoint_len=int(em.group(7)),
                ids=parse_ids(em.group(8)),
            )
            continue
        hm = HASH_RE.match(line)
        if hm and event is not None:
            key = (hm.group(3), int(hm.group(4)), hm.group(5))
            if key not in records:
                records[key] = Record(
                    scope=hm.group(3),
                    layer=int(hm.group(4)),
                    tensor=hm.group(5),
                    bytes=int(hm.group(6)),
                    hash=hm.group(7).lower(),
                    status=hm.group(8),
                )
    if event is None:
        die(f"no first-full-accept {EVENT_PREFIX} event found in {path}")
    if not records:
        die(f"no {HASH_PREFIX} records found after first full accept in {path}")
    return event, records


def event_equivalent(a: Event, b: Event) -> bool:
    return (
        a.drafted == b.drafted
        and a.accepted_drafts == b.accepted_drafts
        and a.ids == b.ids
        and a.start == b.start
    )


def record_result(candidate: Record | None, replay: Record | None) -> str:
    if candidate is None or replay is None:
        return "MISSING"
    if candidate.status != "PASS" or replay.status != "PASS":
        return "ERROR"
    if candidate.bytes != replay.bytes:
        return "BYTES_MISMATCH"
    return "EXACT" if candidate.hash == replay.hash else "MISMATCH"


def parse_token_ids(text: str) -> list[int]:
    for line in text.splitlines():
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
    proc = subprocess.run(
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
    if proc.returncode != 0:
        die(f"tokenizer failed for {text_file}: {proc.stderr.strip()}")
    for stream in (proc.stdout, proc.stderr):
        ids = parse_token_ids(stream)
        if ids:
            return ids
    die(f"no token IDs parsed for {text_file}")


def first_divergence(a: Sequence[int], b: Sequence[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def fmt_index(v: int | None) -> str:
    return "NONE" if v is None else str(v)


def fmt_ordinal(v: int | None) -> str:
    return "NONE" if v is None else str(v + 1)


def extract_tps(log: Path) -> float | None:
    text = log.read_text(encoding="utf-8", errors="replace")
    vals = re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*t/s\b", text)
    return float(vals[-1]) if vals else None


def fmt_tps(v: float | None) -> str:
    return "NA" if v is None else f"{v:.3f}"


def pct_delta(new: float | None, old: float | None) -> str:
    if new is None or old is None or old == 0:
        return "NA"
    return f"{((new / old) - 1.0) * 100.0:.3f}"


def analyze(
    ds4: Path,
    model: Path,
    oracle: Path,
    replay_out: Path,
    replay_log: Path,
    fast_out: Path,
    fast_log: Path,
    kv_out: Path,
    kv_log: Path,
    expected_ordinal1: int,
) -> None:
    replay_event, replay_records = parse_content_log(replay_log)
    fast_event, fast_records = parse_content_log(fast_log)
    kv_event, kv_records = parse_content_log(kv_log)

    replay_match = event_equivalent(fast_event, replay_event)
    kv_match = event_equivalent(kv_event, replay_event)
    print(
        "FAST_COMMIT_KV_REPAIR_EVENT "
        f"fast_vs_replay={'EXACT' if replay_match else 'MISMATCH'} "
        f"kv_vs_replay={'EXACT' if kv_match else 'MISMATCH'} "
        f"drafted={replay_event.drafted} accepted_drafts={replay_event.accepted_drafts} "
        f"start={replay_event.start}"
    )

    raw_keys = sorted(
        key
        for key in set(replay_records) | set(fast_records) | set(kv_records)
        if key[0] == "layer" and key[2] == "raw_new"
    )
    before_exact = 0
    after_exact = 0
    total = len(raw_keys)
    layer0_before = "MISSING"
    layer0_after = "MISSING"
    first_before: int | None = None
    first_after: int | None = None

    for key in raw_keys:
        layer = key[1]
        before = record_result(fast_records.get(key), replay_records.get(key))
        after = record_result(kv_records.get(key), replay_records.get(key))
        if before == "EXACT":
            before_exact += 1
        elif first_before is None:
            first_before = layer
        if after == "EXACT":
            after_exact += 1
        elif first_after is None:
            first_after = layer
        if layer == 0:
            layer0_before = before
            layer0_after = after
            f = fast_records.get(key)
            k = kv_records.get(key)
            r = replay_records.get(key)
            print(
                "FAST_COMMIT_KV_REPAIR_CONTENT "
                f"scope=layer layer=0 tensor=raw_new "
                f"baseline={before} kv_repair={after} "
                f"fast_hash={f.hash if f else 'MISSING'} "
                f"kv_hash={k.hash if k else 'MISSING'} "
                f"replay_hash={r.hash if r else 'MISSING'}"
            )

    print(
        "FAST_COMMIT_KV_REPAIR_RAW_NEW "
        f"exact_layers_before={before_exact} exact_layers_after={after_exact} "
        f"total_layers={total} "
        f"first_mismatch_layer_before={fmt_index(first_before)} "
        f"first_mismatch_layer_after={fmt_index(first_after)}"
    )

    oracle_ids = tokenize(ds4, model, oracle)
    replay_ids = tokenize(ds4, model, replay_out)
    fast_ids = tokenize(ds4, model, fast_out)
    kv_ids = tokenize(ds4, model, kv_out)
    d_replay = first_divergence(oracle_ids, replay_ids)
    d_fast = first_divergence(oracle_ids, fast_ids)
    d_kv = first_divergence(oracle_ids, kv_ids)
    t_replay = extract_tps(replay_log)
    t_fast = extract_tps(fast_log)
    t_kv = extract_tps(kv_log)

    reproduced = d_fast is not None and d_fast + 1 == expected_ordinal1
    print(
        "FAST_COMMIT_KV_REPAIR_BASELINE "
        f"first_diff_index0={fmt_index(d_fast)} "
        f"first_diff_ordinal1={fmt_ordinal(d_fast)} "
        f"expected_ordinal1={expected_ordinal1} "
        f"generation_tps={fmt_tps(t_fast)} "
        f"result={'PASS' if reproduced else 'FAIL'}"
    )
    print(
        "FAST_COMMIT_KV_REPAIR_ORACLE_CONTROL "
        f"replay_first_diff_index0={fmt_index(d_replay)} "
        f"replay_first_diff_ordinal1={fmt_ordinal(d_replay)} "
        f"replay_tps={fmt_tps(t_replay)} "
        f"result={'PASS' if d_replay is None else 'FAIL'}"
    )
    print(
        "FAST_COMMIT_KV_REPAIR_FRONTIER "
        f"baseline_first_diff_index0={fmt_index(d_fast)} "
        f"baseline_first_diff_ordinal1={fmt_ordinal(d_fast)} "
        f"kv_first_diff_index0={fmt_index(d_kv)} "
        f"kv_first_diff_ordinal1={fmt_ordinal(d_kv)} "
        f"baseline_tps={fmt_tps(t_fast)} kv_tps={fmt_tps(t_kv)} "
        f"kv_vs_baseline_tps_delta_pct={pct_delta(t_kv, t_fast)}"
    )

    source_closed = (
        replay_match
        and kv_match
        and layer0_before == "MISMATCH"
        and layer0_after == "EXACT"
    )
    print(
        "FAST_COMMIT_KV_REPAIR_SOURCE="
        + ("FAMILY1_KV_PROJECTION" if source_closed else "NOT_CLOSED")
    )

    if not reproduced or not replay_match or not kv_match or d_replay is not None:
        effect = "INCONCLUSIVE"
    elif d_kv is None or (d_fast is not None and d_kv > d_fast):
        effect = "PROVEN"
    elif d_kv == d_fast:
        effect = "NO_FRONTIER_MOVE"
    else:
        effect = "INCONCLUSIVE"
    print(f"FAST_COMMIT_KV_REPAIR_TOKEN_EFFECT={effect}")

    if not source_closed:
        next_step = "RAW_KV_NUMERICAL_LOCALIZATION"
    elif effect == "PROVEN":
        next_step = "KV_REPAIR_EXPANSION_AND_BENCHMARK"
    elif effect == "NO_FRONTIER_MOVE":
        next_step = "COMPRESSOR_INDEXER_REPAIR_AB"
    else:
        next_step = "ADJUDICATE_CONTROL"
    print(f"FAST_COMMIT_KV_REPAIR_NEXT={next_step}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds4", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--replay-out", required=True, type=Path)
    parser.add_argument("--replay-log", required=True, type=Path)
    parser.add_argument("--fast-out", required=True, type=Path)
    parser.add_argument("--fast-log", required=True, type=Path)
    parser.add_argument("--kv-out", required=True, type=Path)
    parser.add_argument("--kv-log", required=True, type=Path)
    parser.add_argument("--expected-ordinal1", type=int, default=27)
    args = parser.parse_args()
    analyze(
        ds4=args.ds4,
        model=args.model,
        oracle=args.oracle,
        replay_out=args.replay_out,
        replay_log=args.replay_log,
        fast_out=args.fast_out,
        fast_log=args.fast_log,
        kv_out=args.kv_out,
        kv_log=args.kv_log,
        expected_ordinal1=args.expected_ordinal1,
    )


if __name__ == "__main__":
    main()
