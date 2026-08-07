# P2 Dual-Pass Exactness Probe — Status

## 1. Four Critical Findings (from user review)

### Finding 1: Dual-pass direction is correct
The architecture `generic verifier → restore S0 → ordinary sequential decode` with exact N=2 as auxiliary oracle only is the right approach. This preserves P2's original intent while correctly separating reference (ordinary sequential) from measurement (batched verifier).

### Finding 2: Old P2 docs are contradictory — abandon them
- `PROBE-P2-DESIGN.md` has multiple errors:
  - Says to read tensors during layer loop execution (CPU sync in inner loop)
  - Claims Q/KV projections come from `g->batch_cur_hc` — **wrong**, that's HC (hidden state), not Q/KV projection intermediate tensors. Q/KV are at `g->qr[g]` and `g->kv_raw[g]` during attention encoder, visible only within command buffer scope.
  - Recommends "现场重算 canonical" (recompute per-row on GPU) — unnecessary and would change timing/GPU behavior
- `PROBE-P2-AUDIT.md` partially corrected these but the architecture was already revised before writing

### Finding 3: backup/qwen-p2-draft is UNSAFE as implementation base
The backup branch contains `DS4_DSPARK_EXACT_NOREPLAY` which bypasses replay and commits verifier frontier directly via `spec_frontier_commit_prefix1()`. This **re-opens** the greedy identity risk that upstream commit **7fb2830** explicitly fixed:

```
The partial-accept shortcut retained the verifier batch compressor frontier
and could change later greedy tokens. Restore the snapshot and replay partial
accepts just like full accepts. Closes #658 #659
```

Root cause: `ds4_gpu_matmul_f16_pair_tensor` (generic verifier) vs `ds4_gpu_matmul_f16_pair_compressor_store_tensor` (canonical single-token decode) produce different floating-point results for the same input, so committing batch frontier directly causes state drift that flips argmax ties on longer generations.

**Verdict**: backup/qwen-p2-draft is tagged as取证/参考 only — NOT implementation base.

### Finding 4: Probe code has "written too fast" issues
- `ds4_exactness_probe.c` claims "pure C helper" but includes `#include "ds4_gpu.h"` and calls `ds4_gpu_tensor_read()` — not GPU-independent
- Sentinel reset logic incomplete (doesn't clear all probe_state fields)
- Exact-match counting has off-by-one edge cases
- Compile-time gate (`DS4_PROBE_RUNTIME_ENABLED`) was dead code — env var at runtime is the only real control

## 2. P2 Architecture (approved, not yet implemented)

### Execution model
```
=== Pass A: generic verifier ===
1. spec_frontier_snapshot(&frontier, s)    → captures all persistent state
2. push drafts [d0..dN] to checkpoint
3. metal_graph_verify_suffix_tops()        → batched verify
   └─ probe: GPU→GPU copy each layer's 5 checkpoints to snapshot buffers
4. spec_frontier_restore(&frontier, s)     → restores persistent state

=== Pass B: canonical sequential reference ===
5. Snapshot raw KV cache at spec positions [start..start+N-1] (missing from spec_frontier!)
6. metal_graph_eval_token_raw_swa() x N    → each draft token one-at-a-time
   └─ probe: GPU→GPU copy same 5 checkpoints to snapshot buffers
7. Restore raw KV cache

=== Comparison ===
8. ds4_gpu_end_commands() + batched CPU readback
9. byte-for-byte compare batch[cp] vs ref[cp]
10. Report first divergence
```

### Sanity check: ordinary decode == exact N=2 on rows 0-1
Before interpreting any verifier divergence, verify that `metal_graph_eval_token_raw_swa(d0)` produces identical results to `metal_graph_verify_decode2_exact(d0, d1)`, row-by-row. If not, it means the matmul kernel itself is sensitive to batch width (tiling artifact), and ALL comparisons are invalid.

## 3. Unresolved Items Before Implementation

### Raw KV cache gap in spec_frontier
`spec_frontier_snapshot/restore()` does NOT capture raw KV cache at spec positions. This must be added as an additional GPU tensor snapshot before Pass A → after restore, before Pass B. The fix is straightforward (tensor copy per layer), but it's the critical missing piece for identical S0.

### Q/KV checkpoint tensor accuracy
Must use actual Q/KV projection tensors (`g->qr[g]`, `g->kv_raw[g]` via `metal_graph_tensor_row_view()` from attention encoder internals), NOT `batch_cur_hc`. The latter is HC workspace, not Q/KV intermediate state.

### Probe buffer allocation
Need to verify that dedicated GPU snapshot storage for 5 checkpoints × n_tokens rows per layer can be allocated during P2 setup without conflicting with existing tensor layout. Existing candidates: `spec_prefix1_attn_state_kv[il]` and `dspark_target_hidden_batch`.

## 4. What NOT to do
- No exact-row kernel design yet (wait for probe data)
- No new API headers or code generation
- No assumption that compressor projection is the first divergence point
- No sync CPU readback inside layer execution loop

## 5. Recommendation: GO + raw KV snapshot fix

The dual-pass experiment is technically valid per audits A-D. The only blocker before implementation is the raw KV cache gap in `spec_frontier_restore()`.
