#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import dspark_fast_commit_windowed_compressor_ab as e7

REFRESH_RE = re.compile(r"^DS4_DSPARK_E71_REFRESH_BATCH pos0=(\d+) n_tokens=(\d+)$")


def die(msg: str) -> None:
    raise SystemExit(f"dspark_fast_commit_windowed_compressor_e71: {msg}")


def instrument(source: Path) -> None:
    # Reuse the proven E7 projection/Family-7 instrumentation, then split the
    # refresh coordinate system from the outer projection batch coordinate.
    e7.instrument(source)
    text = source.read_text(encoding="utf-8")
    sentinel = "ds4_e71_window_match"
    if sentinel in text:
        die(f"source already contains {sentinel}")

    old_match = r'''static bool ds4_e7_window_match(uint32_t pos0, uint32_t n_tokens) {
    uint32_t target_pos0 = 0;
    uint32_t target_n = 0;
    return ds4_e7_parse_u32_env("DS4_DSPARK_E7_TARGET_POS0", &target_pos0) &&
           ds4_e7_parse_u32_env("DS4_DSPARK_E7_TARGET_N_TOKENS", &target_n) &&
           pos0 == target_pos0 && n_tokens == target_n;
}
'''
    new_match = r'''static bool ds4_e71_window_match(
        const char *part, uint32_t pos0, uint32_t n_tokens) {
    uint32_t target_pos0 = 0;
    uint32_t target_n = 0;
    const bool refresh = part && strcmp(part, "REFRESH") == 0;
    const char *pos_env = refresh ?
        "DS4_DSPARK_E71_REFRESH_TARGET_POS0" : "DS4_DSPARK_E7_TARGET_POS0";
    const char *n_env = refresh ?
        "DS4_DSPARK_E71_REFRESH_TARGET_N_TOKENS" : "DS4_DSPARK_E7_TARGET_N_TOKENS";
    return ds4_e7_parse_u32_env(pos_env, &target_pos0) &&
           ds4_e7_parse_u32_env(n_env, &target_n) &&
           pos0 == target_pos0 && n_tokens == target_n;
}
'''
    if old_match not in text:
        die("E7 window matcher source shape changed")
    text = text.replace(old_match, new_match, 1)

    scout_anchor = r'''static bool ds4_e7_window_part_enabled(
        const char *part, uint32_t pos0, uint32_t n_tokens) {
'''
    refresh_scout = r'''static void ds4_e71_scout_refresh(uint32_t pos0, uint32_t n_tokens) {
    static bool have_last = false;
    static uint32_t last_pos0 = 0;
    static uint32_t last_n = 0;
    if (!ds4_e7_trace_enabled()) return;
    if (have_last && last_pos0 == pos0 && last_n == n_tokens) return;
    have_last = true;
    last_pos0 = pos0;
    last_n = n_tokens;
    fprintf(stderr, "DS4_DSPARK_E71_REFRESH_BATCH pos0=%u n_tokens=%u\n",
            pos0, n_tokens);
}

'''
    if scout_anchor not in text:
        die("E7 part gate anchor missing")
    text = text.replace(scout_anchor, refresh_scout + scout_anchor, 1)

    old_body = r'''    if (!part) return false;
    if (strcmp(part, "PROJECTION") == 0) ds4_e7_scout_batch(pos0, n_tokens);
    if (!mode || !mode[0] || !ds4_e7_window_match(pos0, n_tokens)) return false;
    enabled = strcasecmp(mode, "BOTH") == 0 || strcasecmp(mode, part) == 0;
'''
    new_body = r'''    if (!part) return false;
    if (strcmp(part, "PROJECTION") == 0) ds4_e7_scout_batch(pos0, n_tokens);
    if (strcmp(part, "REFRESH") == 0) ds4_e71_scout_refresh(pos0, n_tokens);
    if (!mode || !mode[0] || !ds4_e71_window_match(part, pos0, n_tokens)) return false;
    enabled = strcasecmp(mode, "BOTH") == 0 || strcasecmp(mode, part) == 0;
'''
    if old_body not in text:
        die("E7 part gate body source shape changed")
    text = text.replace(old_body, new_body, 1)

    source.write_text(text, encoding="utf-8")
    print(
        "FAST_COMMIT_WINDOWED_COMPRESSOR_E71_INSTRUMENT "
        "projection_target=independent refresh_target=independent "
        "refresh_scout=1 result=PASS"
    )


def _parse_ids(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(",") if x)


def select_window(log: Path, output_json: Path) -> None:
    last_projection: tuple[int, int] | None = None
    last_refresh: tuple[int, int] | None = None
    projection_seen = 0
    refresh_seen = 0
    event: dict[str, object] | None = None

    for raw in log.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        pm = e7.BATCH_RE.match(line)
        if pm:
            last_projection = (int(pm.group(1)), int(pm.group(2)))
            projection_seen += 1
            continue
        fm = REFRESH_RE.match(line)
        if fm:
            last_refresh = (int(fm.group(1)), int(fm.group(2)))
            refresh_seen += 1
            continue
        em = e7.EVENT_RE.match(line)
        rm = e7.RETURN_RE.match(line)
        if em and em.group(2) == "FULL_ACCEPT_FAST":
            event = {
                "event_start": int(em.group(6)),
                "drafted": int(em.group(3)),
                "accepted_drafts": int(em.group(4)),
                "ids": list(_parse_ids(em.group(8))),
            }
            break
        if rm and rm.group(2) == "FULL_ACCEPT_FAST":
            event = {
                "event_start": -1,
                "drafted": int(rm.group(3)),
                "accepted_drafts": int(rm.group(4)),
                "ids": list(_parse_ids(rm.group(7))),
            }
            break

    if event is None:
        die("no FULL_ACCEPT_FAST event found in scout log")
    if last_projection is None:
        die("FULL_ACCEPT_FAST observed before any projection batch scout record")

    refresh_status = "PASS" if last_refresh is not None else "ABSENT"
    obj = {
        "pos0": last_projection[0],
        "n_tokens": last_projection[1],
        "projection_pos0": last_projection[0],
        "projection_n_tokens": last_projection[1],
        "refresh_status": refresh_status,
        "refresh_pos0": None if last_refresh is None else last_refresh[0],
        "refresh_n_tokens": None if last_refresh is None else last_refresh[1],
        "projection_batches_seen": projection_seen,
        "refresh_batches_seen": refresh_seen,
        **event,
    }
    output_json.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        "FAST_COMMIT_WINDOWED_COMPRESSOR_WINDOW "
        f"pos0={last_projection[0]} n_tokens={last_projection[1]} "
        f"event_start={'NA' if int(event['event_start']) < 0 else event['event_start']} "
        f"drafted={event['drafted']} accepted_drafts={event['accepted_drafts']} "
        f"batches_seen_before_event={projection_seen} result=PASS"
    )
    if last_refresh is None:
        print(
            "FAST_COMMIT_WINDOWED_COMPRESSOR_REFRESH_WINDOW "
            f"pos0=NONE n_tokens=NONE refreshes_seen_before_event={refresh_seen} "
            "result=ABSENT"
        )
    else:
        print(
            "FAST_COMMIT_WINDOWED_COMPRESSOR_REFRESH_WINDOW "
            f"pos0={last_refresh[0]} n_tokens={last_refresh[1]} "
            f"refreshes_seen_before_event={refresh_seen} result=PASS"
        )


def _load_spec(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _gate_hits(log: Path, spec: dict[str, object]) -> tuple[int, int, int]:
    p_target = (int(spec["projection_pos0"]), int(spec["projection_n_tokens"]))
    refresh_present = spec["refresh_status"] == "PASS"
    r_target = None
    if refresh_present:
        r_target = (int(spec["refresh_pos0"]), int(spec["refresh_n_tokens"]))

    projection = refresh = wrong = 0
    for raw in log.read_text(encoding="utf-8", errors="replace").splitlines():
        gm = e7.GATE_RE.match(raw.strip())
        if not gm:
            continue
        part = gm.group(1)
        coord = (int(gm.group(2)), int(gm.group(3)))
        if part == "PROJECTION":
            projection += 1
            if coord != p_target:
                wrong += 1
        else:
            refresh += 1
            if r_target is None or coord != r_target:
                wrong += 1
    return projection, refresh, wrong


def analyze(args: argparse.Namespace) -> None:
    spec = _load_spec(args.window_json)
    refresh_present = spec["refresh_status"] == "PASS"
    oracle_ids = e7.tokenize(args.ds4, args.model, args.oracle)
    outputs = {
        "replay": (args.replay_out, args.replay_log),
        "baseline": (args.fast_out, args.fast_log),
        "scout": (args.scout_out, args.scout_log),
        "projection": (args.projection_out, args.projection_log),
        "refresh": (args.refresh_out, args.refresh_log),
        "both": (args.both_out, args.both_log),
    }
    ids: dict[str, list[int]] = {}
    diffs: dict[str, int | None] = {}
    tps: dict[str, float | None] = {}
    for name, (out, log) in outputs.items():
        ids[name] = e7.tokenize(args.ds4, args.model, out)
        diffs[name] = e7.first_divergence(oracle_ids, ids[name])
        tps[name] = e7.extract_tps(log)

    baseline_ok = diffs["baseline"] is not None and diffs["baseline"] + 1 == args.expected_ordinal1
    replay_ok = diffs["replay"] is None
    trace_output_exact = ids["baseline"] == ids["scout"]
    trace_frontier_same = diffs["baseline"] == diffs["scout"]

    print(
        "FAST_COMMIT_WINDOWED_COMPRESSOR_BASELINE "
        f"first_diff_index0={e7.fmt_index(diffs['baseline'])} "
        f"first_diff_ordinal1={e7.fmt_ordinal(diffs['baseline'])} "
        f"expected_ordinal1={args.expected_ordinal1} generation_tps={e7.fmt_tps(tps['baseline'])} "
        f"result={'PASS' if baseline_ok else 'FAIL'}"
    )
    print(
        "FAST_COMMIT_WINDOWED_COMPRESSOR_ORACLE_CONTROL "
        f"replay_first_diff_index0={e7.fmt_index(diffs['replay'])} "
        f"replay_first_diff_ordinal1={e7.fmt_ordinal(diffs['replay'])} "
        f"replay_tps={e7.fmt_tps(tps['replay'])} result={'PASS' if replay_ok else 'FAIL'}"
    )
    print(
        "FAST_COMMIT_WINDOWED_COMPRESSOR_TRACE_CONTROL "
        f"token_output={'EXACT' if trace_output_exact else 'MISMATCH'} "
        f"frontier_same={1 if trace_frontier_same else 0} "
        f"baseline_tps={e7.fmt_tps(tps['baseline'])} scout_tps={e7.fmt_tps(tps['scout'])} "
        f"result={'PASS' if trace_output_exact and trace_frontier_same else 'FAIL'}"
    )
    print(
        "FAST_COMMIT_WINDOWED_COMPRESSOR_REFRESH_PARTICIPATION="
        + ("PRESENT" if refresh_present else "ABSENT")
    )

    gate_ok: dict[str, bool] = {}
    early_ok: dict[str, bool] = {}
    for name in ("projection", "refresh", "both"):
        p_hits, r_hits, wrong = _gate_hits(outputs[name][1], spec)
        if name == "projection":
            ok = p_hits > 0 and r_hits == 0 and wrong == 0
        elif name == "refresh":
            ok = p_hits == 0 and (r_hits > 0 if refresh_present else r_hits == 0) and wrong == 0
        else:
            ok = p_hits > 0 and (r_hits > 0 if refresh_present else r_hits == 0) and wrong == 0
        gate_ok[name] = ok
        vs_baseline = e7.first_divergence(ids["baseline"], ids[name])
        early_ok[name] = vs_baseline is None or (
            diffs["baseline"] is not None and vs_baseline >= diffs["baseline"]
        )
        print(
            "FAST_COMMIT_WINDOWED_COMPRESSOR_GATE "
            f"arm={name.upper()} projection_hits={p_hits} refresh_hits={r_hits} "
            f"wrong_window_hits={wrong} projection_target_pos0={spec['projection_pos0']} "
            f"projection_target_n_tokens={spec['projection_n_tokens']} "
            f"refresh_target_pos0={spec['refresh_pos0'] if refresh_present else 'NONE'} "
            f"refresh_target_n_tokens={spec['refresh_n_tokens'] if refresh_present else 'NONE'} "
            f"result={'PASS' if ok else 'FAIL'}"
        )
        print(
            "FAST_COMMIT_WINDOWED_COMPRESSOR_PRE27_CONTROL "
            f"arm={name.upper()} repair_vs_baseline_first_diff_index0={e7.fmt_index(vs_baseline)} "
            f"repair_vs_baseline_first_diff_ordinal1={e7.fmt_ordinal(vs_baseline)} "
            f"result={'PASS' if early_ok[name] else 'FAIL'}"
        )

    labels = {
        "projection": "FAST_COMMIT_WINDOWED_COMPRESSOR_PROJECTION_FRONTIER",
        "refresh": "FAST_COMMIT_WINDOWED_COMPRESSOR_REFRESH_FRONTIER",
        "both": "FAST_COMMIT_WINDOWED_COMPRESSOR_BOTH_FRONTIER",
    }
    for name in ("projection", "refresh", "both"):
        print(
            f"{labels[name]} baseline_first_diff_index0={e7.fmt_index(diffs['baseline'])} "
            f"baseline_first_diff_ordinal1={e7.fmt_ordinal(diffs['baseline'])} "
            f"repair_first_diff_index0={e7.fmt_index(diffs[name])} "
            f"repair_first_diff_ordinal1={e7.fmt_ordinal(diffs[name])} "
            f"baseline_tps={e7.fmt_tps(tps['baseline'])} repair_tps={e7.fmt_tps(tps[name])} "
            f"repair_vs_baseline_tps_delta_pct={e7.pct_delta(tps[name], tps['baseline'])}"
        )

    controls_ok = baseline_ok and replay_ok and trace_output_exact and trace_frontier_same \
        and all(gate_ok.values()) and all(early_ok.values())
    if not controls_ok:
        source = "INCONCLUSIVE"
        next_step = "ADJUDICATE_WINDOW_OR_CONTROL"
    else:
        p = e7.moved(diffs["projection"], diffs["baseline"])
        r = e7.moved(diffs["refresh"], diffs["baseline"]) if refresh_present else False
        b = e7.moved(diffs["both"], diffs["baseline"])
        if refresh_present:
            if (p or r) and not b:
                source = "INCONCLUSIVE"
                next_step = "ADJUDICATE_PROJECTION_REFRESH_COMPOSITION"
            elif p and not r:
                source = "PROJECTION"
                next_step = "PROJECTION_MINIMAL_PRODUCTION_REPAIR"
            elif r and not p:
                source = "REFRESH"
                next_step = "REFRESH_MINIMAL_PRODUCTION_REPAIR"
            elif p and r:
                source = "BOTH_INDEPENDENT"
                next_step = "BENCHMARK_MINIMAL_CHOICES"
            elif b:
                source = "INTERACTION"
                next_step = "PROJECTION_REFRESH_INTERACTION_SPLIT"
            else:
                source = "NOT_PROVEN"
                next_step = "DOWNSTREAM_FAST_COMMIT_CONTENT_AB"
        else:
            if p:
                source = "PROJECTION"
                next_step = "PROJECTION_MINIMAL_PRODUCTION_REPAIR"
            elif b:
                source = "INCONCLUSIVE"
                next_step = "ADJUDICATE_ABSENT_REFRESH_COMPOSITION"
            else:
                source = "NOT_PROVEN"
                next_step = "DOWNSTREAM_FAST_COMMIT_CONTENT_AB"

    print(f"FAST_COMMIT_WINDOWED_COMPRESSOR_SOURCE={source}")
    print(f"FAST_COMMIT_WINDOWED_COMPRESSOR_NEXT={next_step}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    ip = sub.add_parser("instrument")
    ip.add_argument("--source", type=Path, required=True)
    sp = sub.add_parser("select-window")
    sp.add_argument("--log", type=Path, required=True)
    sp.add_argument("--output-json", type=Path, required=True)
    ap = sub.add_parser("analyze")
    for name in ("ds4", "model", "oracle", "window-json", "replay-out", "replay-log", "fast-out", "fast-log",
                 "scout-out", "scout-log", "projection-out", "projection-log", "refresh-out", "refresh-log", "both-out", "both-log"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--expected-ordinal1", type=int, default=27)
    args = p.parse_args()
    if args.cmd == "instrument":
        instrument(args.source)
    elif args.cmd == "select-window":
        select_window(args.log, args.output_json)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
