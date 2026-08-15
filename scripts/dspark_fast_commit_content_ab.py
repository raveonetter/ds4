#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

EVENT_PREFIX = "DS4_DSPARK_CONTENT_EVENT"
HASH_PREFIX = "DS4_DSPARK_CONTENT_HASH"
TARGET_FN = "static int ds4_session_eval_dspark_speculative_argmax("
FAST_MARKER = "fast commit, no replay"
ENV_NAME = "DS4_DSPARK_TRACE_CONTENT_AB"


def die(msg: str) -> None:
    raise SystemExit(f"dspark_fast_commit_content_ab: {msg}")


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


HELPERS = r'''
/* E4 diagnostic only: hash persistent state after the first full accept.
 * Inserted into a temporary worktree by scripts/dspark_fast_commit_content_ab.py. */
static uint64_t ds4_e4_fnv1a_update(uint64_t h, const unsigned char *p, size_t n) {
    for (size_t i = 0; i < n; i++) {
        h ^= (uint64_t)p[i];
        h *= UINT64_C(1099511628211);
    }
    return h;
}

static uint64_t ds4_e4_hash_cpu_bytes(const void *ptr, size_t n) {
    return ds4_e4_fnv1a_update(UINT64_C(1469598103934665603),
                              (const unsigned char *)ptr, n);
}

static bool ds4_e4_hash_tensor_region(ds4_gpu_tensor *t,
                                      uint64_t offset,
                                      uint64_t bytes,
                                      uint64_t *out_hash) {
    if (!t || !out_hash) return false;
    const uint64_t total = ds4_gpu_tensor_bytes(t);
    if (offset > total || bytes > total - offset || bytes > (uint64_t)SIZE_MAX) {
        return false;
    }
    unsigned char *buf = xmalloc((size_t)(bytes ? bytes : 1u));
    bool ok = true;
    if (bytes != 0 && ds4_gpu_tensor_read(t, offset, buf, bytes) == 0) ok = false;
    if (ok) {
        *out_hash = ds4_e4_fnv1a_update(UINT64_C(1469598103934665603),
                                       buf, (size_t)bytes);
    }
    free(buf);
    return ok;
}

static bool ds4_e4_hash_tensor_ring(ds4_gpu_tensor *t,
                                    uint32_t cap,
                                    uint32_t start,
                                    uint32_t len,
                                    uint64_t row_bytes,
                                    uint64_t *out_hash,
                                    uint64_t *out_bytes) {
    if (!t || !out_hash || !out_bytes || cap == 0 || len > cap || start >= cap ||
        row_bytes == 0 || row_bytes > (uint64_t)SIZE_MAX) {
        return false;
    }
    unsigned char *buf = xmalloc((size_t)row_bytes);
    uint64_t h = UINT64_C(1469598103934665603);
    bool ok = true;
    for (uint32_t i = 0; ok && i < len; i++) {
        const uint32_t row = (start + i) % cap;
        const uint64_t off = (uint64_t)row * row_bytes;
        if (ds4_gpu_tensor_read(t, off, buf, row_bytes) == 0) {
            ok = false;
            break;
        }
        h = ds4_e4_fnv1a_update(h, buf, (size_t)row_bytes);
    }
    free(buf);
    if (!ok) return false;
    *out_hash = h;
    *out_bytes = (uint64_t)len * row_bytes;
    return true;
}

static void ds4_e4_emit_tensor_hash(const char *path,
                                    const char *commit_type,
                                    const char *scope,
                                    int layer,
                                    const char *name,
                                    ds4_gpu_tensor *t,
                                    uint64_t offset,
                                    uint64_t bytes) {
    uint64_t h = 0;
    const bool ok = ds4_e4_hash_tensor_region(t, offset, bytes, &h);
    fprintf(stderr,
            "DS4_DSPARK_CONTENT_HASH path=%s commit_type=%s scope=%s layer=%d tensor=%s bytes=%llu hash=%016llx status=%s\n",
            path, commit_type, scope, layer, name,
            (unsigned long long)bytes, (unsigned long long)h,
            ok ? "PASS" : "ERROR");
}

static void ds4_e4_emit_ring_hash(const char *path,
                                  const char *commit_type,
                                  const char *scope,
                                  int layer,
                                  const char *name,
                                  ds4_gpu_tensor *t,
                                  uint32_t cap,
                                  uint32_t start,
                                  uint32_t len,
                                  uint64_t row_bytes) {
    uint64_t h = 0;
    uint64_t bytes = 0;
    const bool ok = len == 0 ? true :
        ds4_e4_hash_tensor_ring(t, cap, start, len, row_bytes, &h, &bytes);
    if (len == 0) h = UINT64_C(1469598103934665603);
    fprintf(stderr,
            "DS4_DSPARK_CONTENT_HASH path=%s commit_type=%s scope=%s layer=%d tensor=%s bytes=%llu hash=%016llx status=%s\n",
            path, commit_type, scope, layer, name,
            (unsigned long long)bytes, (unsigned long long)h,
            ok ? "PASS" : "ERROR");
}
'''


def content_block(indent: str, path: str) -> str:
    if path == "fast_full":
        full_expr = "true"
        commit_type_expr = '"FULL_ACCEPT_FAST"'
    else:
        full_expr = "commit_drafts == draft_n"
        commit_type_expr = '"FULL_ACCEPT_REPLAY"'

    lines = [
        f"{indent}{{",
        f'{indent}    const char *ds4_e4_env = getenv("{ENV_NAME}");',
        f"{indent}    static int ds4_e4_dumped_full_accept = 0;",
        f"{indent}    const bool ds4_e4_is_full_accept = {full_expr};",
        f'{indent}    if (ds4_e4_env && ds4_e4_env[0] && strcmp(ds4_e4_env, "0") != 0 &&',
        f"{indent}        ds4_e4_is_full_accept && !ds4_e4_dumped_full_accept) {{",
        f"{indent}        ds4_e4_dumped_full_accept = 1;",
        f"{indent}        const char *ds4_e4_commit_type = {commit_type_expr};",
        f"{indent}        ds4_gpu_graph *ds4_e4_g = &s->graph;",
        f'{indent}        fprintf(stderr, "{EVENT_PREFIX} path={path} commit_type=%s drafted=%d accepted_drafts=%d returned=%d start=%u checkpoint_len=%d ids=",',
        f"{indent}                ds4_e4_commit_type, draft_n, commit_drafts, n_accept, (unsigned)start, s->checkpoint.len);",
        f"{indent}        for (int ds4_e4_i = 0; ds4_e4_i < n_accept; ds4_e4_i++) {{",
        f'{indent}            fprintf(stderr, "%s%u", ds4_e4_i ? "," : "", (unsigned)accepted[ds4_e4_i]);',
        f"{indent}        }}",
        f"{indent}        fputc('\\n', stderr);",
        f"{indent}        const uint64_t ds4_e4_logits_hash = ds4_e4_hash_cpu_bytes(s->logits, (size_t)DS4_N_VOCAB * sizeof(s->logits[0]));",
        f'{indent}        fprintf(stderr, "{HASH_PREFIX} path={path} commit_type=%s scope=session layer=-1 tensor=logits bytes=%llu hash=%016llx status=PASS\\n",',
        f"{indent}                ds4_e4_commit_type, (unsigned long long)((uint64_t)DS4_N_VOCAB * sizeof(s->logits[0])), (unsigned long long)ds4_e4_logits_hash);",
        f"{indent}        const uint64_t ds4_e4_raw_row_bytes = (uint64_t)DS4_N_HEAD_DIM * sizeof(float);",
        f"{indent}        const uint32_t ds4_e4_new_start = ds4_e4_g->raw_cap ? ((uint32_t)start % ds4_e4_g->raw_cap) : 0u;",
        f"{indent}        for (uint32_t ds4_e4_il = 0; ds4_e4_il < DS4_N_LAYER; ds4_e4_il++) {{",
        f"{indent}            ds4_e4_emit_ring_hash(\"{path}\", ds4_e4_commit_type, \"layer\", (int)ds4_e4_il, \"raw_new\",",
        f"{indent}                                  ds4_e4_g->layer_raw_cache[ds4_e4_il], ds4_e4_g->raw_cap, ds4_e4_new_start,",
        f"{indent}                                  (uint32_t)commit_drafts, ds4_e4_raw_row_bytes);",
        f"{indent}            const uint32_t ds4_e4_ratio = ds4_layer_compress_ratio(ds4_e4_il);",
        f"{indent}            if (ds4_e4_ratio != 0) {{",
        f"{indent}                ds4_e4_emit_tensor_hash(\"{path}\", ds4_e4_commit_type, \"layer\", (int)ds4_e4_il, \"attn_state_kv\",",
        f"{indent}                                        ds4_e4_g->layer_attn_state_kv[ds4_e4_il], 0,",
        f"{indent}                                        ds4_gpu_tensor_bytes(ds4_e4_g->layer_attn_state_kv[ds4_e4_il]));",
        f"{indent}                ds4_e4_emit_tensor_hash(\"{path}\", ds4_e4_commit_type, \"layer\", (int)ds4_e4_il, \"attn_state_score\",",
        f"{indent}                                        ds4_e4_g->layer_attn_state_score[ds4_e4_il], 0,",
        f"{indent}                                        ds4_gpu_tensor_bytes(ds4_e4_g->layer_attn_state_score[ds4_e4_il]));",
        f"{indent}                const uint64_t ds4_e4_comp_row_bytes = (uint64_t)DS4_N_HEAD_DIM *",
        f"{indent}                    (DS4_GPU_ATTN_COMP_CACHE_F16 ? sizeof(uint16_t) : sizeof(float));",
        f"{indent}                ds4_e4_emit_tensor_hash(\"{path}\", ds4_e4_commit_type, \"layer\", (int)ds4_e4_il, \"attn_comp_cache\",",
        f"{indent}                                        ds4_e4_g->layer_attn_comp_cache[ds4_e4_il], 0,",
        f"{indent}                                        (uint64_t)ds4_e4_g->layer_n_comp[ds4_e4_il] * ds4_e4_comp_row_bytes);",
        f"{indent}            }}",
        f"{indent}            if (ds4_e4_ratio == 4) {{",
        f"{indent}                ds4_e4_emit_tensor_hash(\"{path}\", ds4_e4_commit_type, \"layer\", (int)ds4_e4_il, \"index_state_kv\",",
        f"{indent}                                        ds4_e4_g->layer_index_state_kv[ds4_e4_il], 0,",
        f"{indent}                                        ds4_gpu_tensor_bytes(ds4_e4_g->layer_index_state_kv[ds4_e4_il]));",
        f"{indent}                ds4_e4_emit_tensor_hash(\"{path}\", ds4_e4_commit_type, \"layer\", (int)ds4_e4_il, \"index_state_score\",",
        f"{indent}                                        ds4_e4_g->layer_index_state_score[ds4_e4_il], 0,",
        f"{indent}                                        ds4_gpu_tensor_bytes(ds4_e4_g->layer_index_state_score[ds4_e4_il]));",
        f"{indent}                ds4_e4_emit_tensor_hash(\"{path}\", ds4_e4_commit_type, \"layer\", (int)ds4_e4_il, \"index_comp_cache\",",
        f"{indent}                                        ds4_e4_g->layer_index_comp_cache[ds4_e4_il], 0,",
        f"{indent}                                        (uint64_t)ds4_e4_g->layer_n_index_comp[ds4_e4_il] *",
        f"{indent}                                            DS4_N_INDEXER_HEAD_DIM * sizeof(float));",
        f"{indent}            }}",
        f"{indent}        }}",
        f"{indent}        if (ds4_e4_g->mtp_raw_cache && ds4_e4_g->raw_cap != 0) {{",
        f"{indent}            const uint32_t ds4_e4_mtp_len = ds4_e4_g->mtp_n_raw > ds4_e4_g->raw_cap ? ds4_e4_g->raw_cap : ds4_e4_g->mtp_n_raw;",
        f"{indent}            const uint32_t ds4_e4_ckpt_len = s->checkpoint.len > 0 ? (uint32_t)s->checkpoint.len : 0u;",
        f"{indent}            const uint32_t ds4_e4_mtp_abs_start = ds4_e4_ckpt_len >= ds4_e4_mtp_len ? ds4_e4_ckpt_len - ds4_e4_mtp_len : 0u;",
        f"{indent}            ds4_e4_emit_ring_hash(\"{path}\", ds4_e4_commit_type, \"mtp\", -1, \"mtp_raw_active\",",
        f"{indent}                                  ds4_e4_g->mtp_raw_cache, ds4_e4_g->raw_cap,",
        f"{indent}                                  ds4_e4_mtp_abs_start % ds4_e4_g->raw_cap, ds4_e4_mtp_len, ds4_e4_raw_row_bytes);",
        f"{indent}        }}",
        f"{indent}        if (ds4_e4_g->mtp_state_hc) {{",
        f"{indent}            ds4_e4_emit_tensor_hash(\"{path}\", ds4_e4_commit_type, \"mtp\", -1, \"mtp_state_hc\",",
        f"{indent}                                    ds4_e4_g->mtp_state_hc, 0, ds4_gpu_tensor_bytes(ds4_e4_g->mtp_state_hc));",
        f"{indent}        }}",
        f"{indent}        if (ds4_e4_g->mtp_next_hc) {{",
        f"{indent}            ds4_e4_emit_tensor_hash(\"{path}\", ds4_e4_commit_type, \"mtp\", -1, \"mtp_next_hc\",",
        f"{indent}                                    ds4_e4_g->mtp_next_hc, 0, ds4_gpu_tensor_bytes(ds4_e4_g->mtp_next_hc));",
        f"{indent}        }}",
        f"{indent}        if (ds4_e4_g->dspark_cache_cap != 0 && ds4_e4_g->dspark_cache_len <= ds4_e4_g->dspark_cache_cap) {{",
        f"{indent}            for (uint32_t ds4_e4_stage = 0; ds4_e4_stage < DS4_DSPARK_MAX_STAGES; ds4_e4_stage++) {{",
        f"{indent}                if (!ds4_e4_g->dspark_raw_cache[ds4_e4_stage]) continue;",
        f"{indent}                char ds4_e4_name[48];",
        f'{indent}                snprintf(ds4_e4_name, sizeof(ds4_e4_name), "dspark_raw_stage%u", ds4_e4_stage);',
        f"{indent}                ds4_e4_emit_ring_hash(\"{path}\", ds4_e4_commit_type, \"dspark\", -1, ds4_e4_name,",
        f"{indent}                                      ds4_e4_g->dspark_raw_cache[ds4_e4_stage], ds4_e4_g->dspark_cache_cap,",
        f"{indent}                                      ds4_e4_g->dspark_cache_start, ds4_e4_g->dspark_cache_len, ds4_e4_raw_row_bytes);",
        f"{indent}            }}",
        f"{indent}        }}",
        f"{indent}        fflush(stderr);",
        f"{indent}    }}",
        f"{indent}}}",
        f"{indent}return n_accept;",
    ]
    return "\n".join(lines)


def instrument(source: Path) -> None:
    text = source.read_text(encoding="utf-8")
    if EVENT_PREFIX in text or ENV_NAME in text:
        die(f"{source} is already content-instrumented")
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
        pieces.append(content_block(m.group(1), path))
        labels.append(path)
        cursor = m.end()
    pieces.append(fn[cursor:])
    patched_fn = "".join(pieces)
    patched = text[:fn_start] + HELPERS + "\n" + patched_fn + text[fn_end + 1 :]
    source.write_text(patched, encoding="utf-8")
    print(
        f"FAST_COMMIT_CONTENT_INSTRUMENT status=PASS return_sites={len(eligible)} "
        f"fast_sites={labels.count('fast_full')} replay_sites={labels.count('replay_or_partial')}"
    )


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
    hash_hex: str
    status: str


EVENT_RE = re.compile(
    rf"^{EVENT_PREFIX} path=(\S+) commit_type=(\S+) drafted=(\d+) accepted_drafts=(\d+) "
    r"returned=(\d+) start=(\d+) checkpoint_len=(\d+) ids=(.*)$"
)
HASH_RE = re.compile(
    rf"^{HASH_PREFIX} path=(\S+) commit_type=(\S+) scope=(\S+) layer=(-?\d+) "
    r"tensor=(\S+) bytes=(\d+) hash=([0-9a-fA-F]+) status=(\S+)$"
)


def parse_ids(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(",") if x)


def parse_log(path: Path) -> tuple[Event, dict[tuple[str, int, str], Record]]:
    event: Event | None = None
    records: dict[tuple[str, int, str], Record] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        em = EVENT_RE.match(line)
        if em:
            if event is None:
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
        if hm:
            key = (hm.group(3), int(hm.group(4)), hm.group(5))
            if key not in records:
                records[key] = Record(
                    scope=hm.group(3),
                    layer=int(hm.group(4)),
                    tensor=hm.group(5),
                    bytes=int(hm.group(6)),
                    hash_hex=hm.group(7).lower(),
                    status=hm.group(8),
                )
    if event is None:
        die(f"no {EVENT_PREFIX} event found in {path}")
    if not records:
        die(f"no {HASH_PREFIX} records found in {path}")
    return event, records


def classify(key: tuple[str, int, str]) -> str:
    scope, _layer, tensor = key
    if scope == "session" and tensor == "logits":
        return "SESSION_LOGITS"
    if tensor == "raw_new":
        return "TARGET_RAW_KV"
    if tensor.startswith("attn_state_"):
        return "ATTN_COMPRESSOR_FRONTIER"
    if tensor == "attn_comp_cache":
        return "ATTN_COMPRESSED_CACHE"
    if tensor.startswith("index_state_"):
        return "INDEX_COMPRESSOR_FRONTIER"
    if tensor == "index_comp_cache":
        return "INDEX_COMPRESSED_CACHE"
    if scope == "mtp":
        return "MTP_STATE"
    if scope == "dspark":
        return "DSPARK_SUPPORT_CACHE"
    return "OTHER"


def analyze(fast_log: Path, replay_log: Path) -> None:
    fast_event, fast = parse_log(fast_log)
    replay_event, replay = parse_log(replay_log)
    event_match = (
        fast_event.drafted == replay_event.drafted
        and fast_event.accepted_drafts == replay_event.accepted_drafts
        and fast_event.ids == replay_event.ids
    )
    print(
        f"FAST_COMMIT_CONTENT_TRACE fast_records={len(fast)} replay_records={len(replay)} "
        f"event_match={1 if event_match else 0} drafted_fast={fast_event.drafted} drafted_replay={replay_event.drafted}"
    )
    if not event_match:
        print("FAST_COMMIT_CONTENT_DIVERGENCE=INCONCLUSIVE reason=full_accept_event_mismatch")
        return

    keys = sorted(set(fast) | set(replay), key=lambda k: (k[0], k[1], k[2]))
    mismatches: list[tuple[tuple[str, int, str], str]] = []
    errors = 0
    for key in keys:
        a = fast.get(key)
        b = replay.get(key)
        if a is None or b is None:
            mismatches.append((key, "MISSING"))
            continue
        if a.status != "PASS" or b.status != "PASS":
            errors += 1
            mismatches.append((key, "ERROR"))
            continue
        if a.bytes != b.bytes:
            mismatches.append((key, "SIZE"))
            continue
        if a.hash_hex != b.hash_hex:
            mismatches.append((key, "HASH"))

    print(
        f"FAST_COMMIT_CONTENT_AB compared={len(keys)} mismatches={len(mismatches)} "
        f"read_errors={errors} result={'MISMATCH' if mismatches else 'EXACT'}"
    )
    if not mismatches:
        print("FAST_COMMIT_FIRST_CONTENT_DIVERGENCE=NONE")
        print("FAST_COMMIT_CONTENT_DIVERGENCE=NONE")
        print("FAST_COMMIT_CONTENT_NEXT=NON_CACHE_STATE_OR_VERIFY_LOGITS")
        return

    classes: list[str] = []
    for idx, (key, reason) in enumerate(mismatches):
        scope, layer, tensor = key
        cls = classify(key)
        if cls not in classes:
            classes.append(cls)
        a = fast.get(key)
        b = replay.get(key)
        if idx < 64:
            print(
                f"FAST_COMMIT_CONTENT_DIFF scope={scope} layer={layer} tensor={tensor} class={cls} reason={reason} "
                f"fast_hash={a.hash_hex if a else 'MISSING'} replay_hash={b.hash_hex if b else 'MISSING'} "
                f"fast_bytes={a.bytes if a else 'MISSING'} replay_bytes={b.bytes if b else 'MISSING'}"
            )
    first_key, _ = mismatches[0]
    print(
        f"FAST_COMMIT_FIRST_CONTENT_DIVERGENCE scope={first_key[0]} layer={first_key[1]} "
        f"tensor={first_key[2]} class={classify(first_key)}"
    )
    print(f"FAST_COMMIT_CONTENT_CLASSES={','.join(classes)}")
    if errors:
        print("FAST_COMMIT_CONTENT_DIVERGENCE=INCONCLUSIVE reason=tensor_read_error")
    else:
        print("FAST_COMMIT_CONTENT_DIVERGENCE=FOUND")
        print("FAST_COMMIT_CONTENT_NEXT=NUMERICAL_LOCALIZATION_AND_CAUSAL_SUBSTITUTION")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_inst = sub.add_parser("instrument")
    p_inst.add_argument("--source", type=Path, required=True)
    p_an = sub.add_parser("analyze")
    p_an.add_argument("--fast-log", type=Path, required=True)
    p_an.add_argument("--replay-log", type=Path, required=True)
    args = parser.parse_args()
    if args.cmd == "instrument":
        instrument(args.source)
    else:
        analyze(args.fast_log, args.replay_log)


if __name__ == "__main__":
    main()
