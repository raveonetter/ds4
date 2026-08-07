# Codex C2b Implementation Note

## Scope

- Source baseline: `34f858fce638dff779c883ada3354339cd07a95d`.
- Frozen design documents are inherited from `codex/p2-first-divergence`.
- Runtime target: single-device Apple Silicon Metal.
- Diagnostic switch: `DS4_DSPARK_C2B=1`; default execution is unchanged.
- This commit implements C2b only. It does not implement or execute C4/C5.

## Checkpoint source verification

| Checkpoint | Generic source and capture point | Layout | Status |
|---|---|---|---|
| CP1 | `metal_graph_batch_attn_norm(g)`, after layer attention encode | `n_tokens × DS4_N_EMBD`, contiguous F32 | enabled |
| CP2-Q | `metal_graph_batch_qr(g)`, after its Q-A producer and before the next layer reuses it | `n_tokens × layer->attn_q_a->dim[1]`, contiguous F32 | enabled |
| CP2-KV-P | `metal_graph_batch_kv_raw(g)`, after KV projection | `n_tokens × DS4_N_HEAD_DIM`, contiguous F32 | enabled |
| CP2-KV-R | `layer_raw_cache[il]` after batch store | per-row F32 at `(start+r) % raw_cap` | enabled; copied per physical row |
| CP3-P | no canonical sequential semantic-equivalent materialized object without changing fusion | — | `CP3-P UNAVAILABLE` |
| CP3-F | existing prefix frontier slots for rows before the last; live frontier for the last row; newly live append rows | contiguous F32 frontier; attention append rows retain native F16 on Metal | enabled |
| CP4 | `metal_graph_batch_after_attn_hc(g)` after attention | `n_tokens × DS4_N_HC × DS4_N_EMBD`, contiguous F32 | enabled |
| CP5 | `metal_graph_batch_next_hc(g)` after FFN and before pointer swap | `n_tokens × DS4_N_HC × DS4_N_EMBD`, contiguous F32 | enabled |

All eligible checkpoint transport uses
`ds4_gpu_tensor_copy_f32_inline()` in the active compute encoder. There is no
checkpoint blit fallback and no layer-loop CPU readback.

## Three-run gate

The diagnostic performs one isolated process-ending gate:

1. snapshot the original compressor frontier and every affected raw-KV row;
2. A0 with the checkpoint hook completely detached;
3. allocate checkpoint buffers, restore S0, run A1 with hooks active but no copies;
4. restore S0, run A2 with full checkpoint copies;
5. restore S0, then compare A0/A1 and A0/A2 on CPU.

Compared post-Pass-A observables include verifier logits/tops, session logits,
all affected raw-KV physical rows, attention/index frontier tensors and
counters, newly live compressed-cache rows, checkpoint metadata, and restored
support-cache window counters. F32 comparisons use `ds4_float_compare_exact()`.
Metal F16 attention-cache rows are compared by their native 16-bit storage;
their numerical diagnostics are computed after F16-to-F32 expansion.

Ring wrap is handled with independent per-row copies. A proposal span larger
than `raw_cap`, or larger than the frozen prefix checkpoint capacity, fails the
diagnostic instead of aliasing silently.

## Files and functions

- `ds4.c`
  - C2b checkpoint storage and same-encoder hooks.
  - raw-KV S0 snapshot/restore.
  - post-A observable capture and exact comparison.
  - `DS4_DSPARK_C2B` diagnostic entry point.
- `Makefile`
  - links the verified CPU comparator into GPU executables.
- `skill/CODEX-C2B-IMPLEMENTATION-NOTE.md`
  - frozen execution handoff.

## Build

```sh
make -B ds4
```

## M4 Max execution

```sh
DS4_DSPARK_C2B=1 \
DS4_DSPARK_SCHEDULER=0 \
./ds4 --dspark --dspark-confidence 0 \
  -m "$PWD/ds4flash.gguf" \
  --mtp "$PWD/gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf" \
  --tokens 16 --temp 0 --nothink \
  -p 'Explain Redis in one sentence.'
```

The process exits immediately after the first available C2b proposal block.
Success requires all three lines:

```text
C2B_CONTROL A0_vs_A1 PASS
C2B_PROBE A0_vs_A2 PASS
C2B_RESULT PASS
```

Any setup, verifier, restoration, or comparison failure emits
`C2B_RESULT FAIL` and exits nonzero.

## Local validation

The implementation host is not an M4 Max. Completed checks:

```sh
make -B ds4.o ds4_float_compare.o
make -B ds4_cpu.o
make test-float-compare
git diff --check
```

No M4 Max runtime result is claimed by this commit.
