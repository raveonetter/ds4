# Qwen CP3 Compressor State-Machine Audit

## Source baseline
`b0309611041655f4e45671cfd9c9886aff161406` (main branch, upstream/ds4f-mxfp4)

## 1. Generic batch path: CP2 → CP3

### metal_graph_encode_layer_attention_batch (line 27172, decode sub-path)
For DSpark verifier with n_tokens draft tokens, the compressor stage starts at line 22216:

```
Line 22217-22219: ratio = ds4_layer_compress_ratio(il), coff = (ratio==4 ? 2 : 1), comp_width = coff * DS4_N_HEAD_DIM
Line 22220: emit = ((pos + 1) % ratio) == 0

/* Branch A: fused path */
Line 22236: comp_state_already_stored = false
Lines 22237-22258: ds4_gpu_matmul_f16_pair_compressor_store_tensor()
    → If returns > 0 (M3/M5 default): writes KV+score to state tensors in-place, sets comp_state_already_stored = true
    → input: metal_graph_attn_norm(g)[token] + compressor weights
    → output: out_kv/out_score (projected KV/score) + direct write to layer_attn_state_kv/sc[il]

/* Branch B: separate matmul path */
Line 22259-22269: ds4_gpu_matmul_f16_pair_tensor()
    → Only reached if fused returns ≤ 0
    → Writes projected KV/score to metal_graph_comp_kv_cur/sc_cur(g)

/* Compressor update */
Line 22282: comp_row = g->layer_n_comp[il]   /* READ counter for target row */
Lines 22283-22307: ds4_gpu_compressor_update_tensor(..., comp_state_already_stored)
    → If !state_already_stored: writes to state_kv/sc accumulators
    → If state_already_stored: skips redundant state writes
    → Writes compressed KV row to comp_cache[comp_row] (if emit == true)
Line 22309-22318: FP8 quantize of comp_cache[comp_row] (if emit)
Line 22321: metal_graph_commit_attn_comp_stage(g, il, comp_row, 1) (if emit)
Line 22324: if (ok && emit) g->layer_n_comp[il]++   /* INCREMENT counter */
```

### Generic batch key characteristic
- In `metal_graph_encode_layer_attention_batch`, the generic verifier path processes **n_tokens tokens in ONE call** to this function.
- Each token calls fused store → one write per row into state_kv/sc at positions `(pos % ratio)`.
- The function is called with pos0 (absolute start position), and within the loop, each draft token gets its own `pos = pos0 + i`.

## 2. Ordinary sequential path: CP2 → CP3

### metal_graph_encode_decode_layer_phase (line 21788)
Called once per token per layer. One call at a time:

```
Line 22217-22219: ratio, coff, comp_width (same as generic)
Line 22220: emit = ((pos + 1) % ratio) == 0
Line 22236: comp_state_already_stored = false   /* Reset EVERY call */

Lines 22237-22258: ds4_gpu_matmul_f16_pair_compressor_store_tensor()
    → On M3/M5 hardware without env override: always returns >0
    → Sets comp_state_already_stored = true (line 22258)
    → Writes projected KV to state_kv[row] = kv[gid]   /* Assignment, not accumulation */
    → Writes projected score to state_score[row] = sc[gid] + ape[gid]

Lines 22282-2307: ds4_gpu_compressor_update_tensor(metal_graph_comp_kv_cur(g), ..., comp_state_already_stored)
    → Skips state store (already done by fused kernel)
    → For emit tokens: pool+norm+RoPE on accumulated rows, write to comp_cache[comp_row]

Line 22324: if (ok && emit) g->layer_n_comp[il]++   /* INCREMENT */
```

## 3. Four objects — separated and timeline'd

### Object 1: Projection result for current token
| Path | Tensor | When valid |
|------|--------|-----------|
| Generic batch | out_kv/out_score from fused kernel | Immediately after fused matmul within the same layer call |
| Sequential | Same tensor | After each separate encode_decode_layer_phase call |

**Implementation difference**: Generic writes all n_tokens in one kernel dispatch (row-parallel). Sequential writes one row at a time. The MATMUL kernels have different tiling/accumulation orders — this is the **earliest source-level implementation difference**, but it's BEFORE CP3 boundary (at the matmul level, not state storage).

### Object 2: Recurrent/state frontier `state_kv` / `state_score`
| Path | How accumulated | Write pattern | Tensor size |
|------|----------------|--------------|------------|
| Generic batch | Within ONE layer call, all n_tokens rows written sequentially to different positions in state tensor | `state_kv[row] = projected_kv;` (assignment per row) | `coff * DS4_N_HEAD_DIM × coff * ratio` |
| Sequential | Across separate calls for successive tokens | Same assignment pattern | Same |

**Key insight**: With fused store, the write is **assignment** (`state_kv[dst] = kv[gid]`), not accumulation. This is per-row: row `pos % ratio`. After 4 consecutive ordinary decode calls (ratio=4) or after the n_tokens draft tokens in generic batch, ALL rows of the state tensor are filled with fresh projected values — no accumulation across tokens happens.

### Object 3: Compressed cache emitted row
| Path | Written when | Kernel |
|------|-------------|--------|
| Generic batch | On emit boundaries (every `ratio`-th token) | `ds4_gpu_dsv4_fp8_kv_quantize_tensor` |
| Sequential | Same (every `ratio`-th token) | Same kernel |

**Implementation difference**: Both paths write to the SAME physical tensor `layer_attn_comp_cache[il]`. The row index is determined by `comp_row = g->layer_n_comp[il]`, which is shared mutable state.

### Object 4: Compressed-row counter
| Path | Read/write pattern |
|------|-------------------|
| Generic batch | READ: comp_row = layer_n_comp[il]; WRITE: if emit, increment |
| Sequential | Same pattern |

## 4. `comp_state_already_stored` — resolved

### Where it becomes true
| Location | Line | Condition |
|----------|------|-----------|
| Initialized to false | 22236 (generic) / 21788 path (sequential) | Always at entry |
| Set to true | 22258 | Only when fused store returns >0 |

### Conditions for fused store → true
- NOT in quality mode
- Device contains "M3" or "M5" (Apple Silicon >= M3)
- `DS4_METAL_ENABLE_COMPRESSOR_PAIR_STATE_STORE` or env override not set to 0
- `in_dim == 4096`, width in {256, 512, 1024}, ratio in {4, 128}
- All buffers valid and properly sized
- On M3/M5 hardware in normal operation: **always true**

### What operations are skipped when true
1. No separate `ds4_gpu_matmul_f16_pair_tensor` call needed (bypassed)
2. `ds4_gpu_compressor_update_tensor` skips its state_store step (`if (!state_already_stored)` at line 22005 of ds4_metal.m)

### Relevant to which CP?
**Both CP3 and S0 restoration argument.** The fused store path means the state boundary seen by compressor_update is different — it sees already-written state tensors instead of needing to write them. This affects both:
- What CP3 actually captures (state was written BEFORE compressor_update runs)
- Whether `spec_frontier_restore()` needs to also restore state tensor contents (it does, because the counter + tensor content together define the frontier)

## 5. Earliest source-level implementation difference between generic and ordinary (CP2→CP3)

### Timeline of differences:

| # | Point | Generic batch | Ordinary sequential | PROVEN? |
|---|-------|--------------|-------------------|---------|
| D1 | Fused store path selection | Always fused on M3/M5 | Always fused on M3/M5 | No difference (same hardware) |
| D2 | Matmul kernel dispatch order | n_tokens rows dispatched in ONE kernel launch | One row per dispatch, one kernel per call | PROVEN BY SOURCE (line 27172 vs line 21788) |
| D3 | Fused store write pattern within state tensor | Within single kernel, all n_tokens writes to different rows in same pass | Each call writes ONE row at pos%ratio position | PROVEN (kernel_dsv4_compressor_store_one, dsv4_kv.metal:343) |
| D4 | State accumulation across tokens | Within one layer call, each token OVERWRITES state_row[pos%ratio] — no inter-token accumulation | Across calls, same overwrite pattern | Same logic, different timing |
| D5 | Compressor update call frequency | Called ONCE per layer with all rows pre-written by fused store | Called ONCE per token-layer pair | PROVEN (line 22283 vs line 21788) |
| D6 | Pool+norm+RoPE timing | After ALL n_tokens tokens processed (if emit == true within that batch) | After each individual emit token | PROVEN |
| D7 | Counter increment pattern | `if (ok && emit) g->layer_n_comp[il]++` — potentially multiple increments across one call for the same layer if multiple draft positions hit emit boundaries | Same formula, called less frequently | PROVEN |

### D5 is THE key difference affecting CP3:
- **Generic batch**: Fused store writes ALL rows for all draft tokens in one kernel pass → then ONE compressor_update call processes them
- **Sequential**: Each token call triggers its own fused_store (one row) + compressor_update (pool+norm+RoPE if emit)

For CP3 comparison: the question is whether the state tensor contents after D5's write differ between paths, AND whether the pool+norm+RoPE results differ.

## 6. Ratio-4 Indexer path

### Structural comparison:
| Aspect | Attention compressor | Indexer compressor |
|--------|---------------------|-------------------|
| State tensors | `layer_attn_state_kv/sc[il]` | `layer_index_state_kv/sc[il]` |
| Cache tensor | `layer_attn_comp_cache[il]` | `layer_index_comp_cache[il]` |
| Counter | `layer_n_comp[il]` | `layer_n_index_comp[il]` |
| Fused store function | Same (same fused kernel, different offsets) | Same pattern |
| Update call line | 22283 | 22390 |
| Quantize kernel | `ds4_gpu_dsv4_fp8_kv_quantize_tensor` | `ds4_gpu_dsv4_indexer_qat_tensor` (QAT format, not FP8) |
| Top-K selection | N/A | Additional: scores compressed rows, selects top-K candidates |

### Key structural difference for ratio-4:
The indexer path triggers sparse attention when BOTH `layer_n_comp > decode_sparse_threshold` AND `layer_index_comp > DS4_N_INDEXER_TOP_K`. This creates a **second compressor state machine** running in parallel with the first, with its own frontiers but sharing some input tensors.

## 7. CP3 definition challenge

Codex defines CP3 as "compressor projection + persistent frontier/update state". Challenge:

### Problem: Fused store makes "projection result" and "frontier update" inseparable
With `comp_state_already_stored = true`:
- The projected KV/score for a token is WRITTEN directly into the state tensor in the SAME kernel that computes it
- There is no materialized "intermediate projection result" tensor accessible at CP3 boundary
- The only observable boundaries are: (a) BEFORE fused store writes to state, or (b) AFTER fused store + after compressor_update pool

### Recommendation for CP3 instrumentation:
CP3 should be split into two observables for the fused-store case:
1. **CP3a**: Projected KV/score values IMMEDIATELY after matmul but BEFORE state write (requires instrumenting within the fused kernel)
2. **CP3b**: Persistent state frontier AFTER compressor_update pool+norm+RoPE completes

For the separate-matmul fallback path, CP3 can be a single observable: the content of `metal_graph_comp_kv_cur/sc_cur(g)` after matmul but before update.

## 8. Earliest source-level implementation difference summary

| Difference | Classification | Impact on experiment |
|-----------|---------------|---------------------|
| D2: Matmul dispatch order | EARLIEST SOURCE-LEVEL DIFFERENCE (BEFORE CP3) | Batch kernel vs sequential kernel may produce different projected values for the same input — this is a **CP2→CP3** issue, not CP3 itself |
| D5: Single call vs multiple calls to compressor_update | EARLIEST SOURCE-LEVEL DIFFERENCE (AT/IN CP3) | Different number of pool+norm+RoPE applications may cause different intermediate states |
| D6: Pool timing | EARLIEST SOURCE-LEVEL DIFFERENCE (IN CP3) | Sequential applies pool after each emit; generic applies pool only at end of batch — **but both should see the same final state_kv/sc content if row writes are identical** |

### Important distinction:
D2 is BEFORE CP3 — it's about whether projected values differ between paths (that's a CP2→CP3 issue). D5 and D6 are IN CP3 — they're about whether the compressor state machine processes rows differently, regardless of input.

## 9. Verdict on generic vs sequential equivalence at CP3

- **If `comp_state_already_stored = false` (separate matmul path)**: CP3 comparison is clean — projected values in comp_kv/sc_cur tensors are materialized and comparable before update begins.
- **If `comp_state_already_stored = true` (fused store, default on M3/M5)**: CP3 boundary is blurred because projection + state write happen simultaneously. The only truly observable boundaries are pre-fused-store and post-compressor-update.

### Strongest remaining risk
The fused-store path means that **there may be no clean CP3 boundary** that exists identically in both paths. When `comp_state_already_stored = true`, the "projection result" is never a standalone tensor — it's embedded within the state write operation. This doesn't invalidate the experiment but requires rethinking CP3 instrumentation: either instrument inside the fused kernel (difficult without changing code) or compare only pre/post boundaries.
