# Codex C4/C5 Implementation Note

## Gate and scope

- C2b runtime gate supplied by Qwen/M4 Max:
  - `C2B_CONTROL A0_vs_A1 PASS`
  - `C2B_PROBE A0_vs_A2 PASS`
  - `C2B_RESULT PASS`
- Runtime target: single-device Apple Silicon Metal, non-streaming model load.
- Diagnostic switch: `DS4_FIRST_DIVERGENCE=1`; default execution is unchanged.
- `CP3-P UNAVAILABLE` remains frozen. Canonical sequential fusion is unchanged.

## C4 execution

The diagnostic copies the Pass-A proposal IDs into immutable local storage,
preallocates both checkpoint sets, snapshots the raw-KV physical rows, and then:

1. runs the generic DSpark batched verifier with the forced proposal sequence;
2. resets `checkpoint.len` to `start`;
3. calls `spec_frontier_restore()`;
4. restores every Pass-A-overwritable raw-KV row independently at
   `(start + row) % raw_cap`;
5. calls canonical `metal_graph_eval_token_raw_swa(...)` once per forced token,
   in proposal order, and appends that same token only after its decode succeeds.

Pass B never samples or regenerates a token. The sequential checkpoint hook is
NULL outside this process-ending diagnostic and only adds same-encoder inline
copies after a canonical layer encode and before its HC pointer swap.

## C5 comparison

CPU comparison runs only after both passes and GPU synchronization. It traverses:

```text
row ascending
  layer ascending
    CP1
    CP2-Q
    CP2-KV-P
    CP2-KV-R
    CP3-F subobjects
    CP4
    CP5
```

Each F32 object uses `ds4_float_compare_exact()`. Native Metal F16 attention
compressed-cache rows are compared bitwise and expanded only for diagnostic
metrics. The log emits a full exact/mismatch map and exactly one terminal
first-divergence result:

```text
FIRST_DIVERGENCE row=... layer=... checkpoint=... index=... actual_bits=... expected_bits=... mismatch_count=... max_abs=... max_rel=... max_ulp=...
```

or:

```text
FIRST_DIVERGENCE NONE
```

Here `actual` is generic Pass A and `expected` is canonical sequential Pass B.

## Build

```sh
make -B ds4
```

## M4 Max execution

```sh
DS4_FIRST_DIVERGENCE=1 \
DS4_DSPARK_SCHEDULER=0 \
./ds4 --dspark --dspark-confidence 0 \
  -m "$PWD/ds4flash.gguf" \
  --mtp "$PWD/gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf" \
  --tokens 16 --temp 0 --nothink \
  -p 'Explain Redis in one sentence.'
```

The process exits immediately after the first available proposal block. A
successful experiment run exits zero whether the paths are exact or diverged;
setup, capture, restore, synchronization, and readback errors exit nonzero.

## Local validation

The implementation host is not an M4 Max. Completed checks:

```sh
make -B ds4.o ds4_float_compare.o
make -B ds4_cpu.o
make test-float-compare
git diff --check
```

No C5 runtime result is claimed by this commit.
