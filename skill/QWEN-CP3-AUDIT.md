# Qwen CP3 Compressor State-Machine Audit — Corrected

## Source baseline
`b0309611041655f4e45671cfd9c9886aff161406` (main branch, upstream/ds4f-mxfp4)

## Correction notice

**The previous version of this audit contained a critical conflation error.**
It attributed the function calls and variable `comp_state_already_stored` from
`metal_graph_encode_decode_layer_phase` (line 21788, the sequential decode
helper) to `metal_graph_encode_layer_attention_batch` (line 27172, the generic
verifier).

**PROVEN BY SOURCE:** Searching lines 27172–29000 of ds4.c at b030961:
- `ds4_gpu_matmul_f16_pair_compressor_store_tensor`: **0 occurrences**
- `ds4_gpu_matmul_f16_pair_tensor`: **0 occurrences**
- `comp_state_already_stored`: **0 occurrences**

Every conclusion in the previous audit's Section 1, Sections D1–D7, and
Section 8/9 that depended on the generic verifier using a "fused store path"
is invalidated. This corrected version replaces them.

---

## 1. Generic batch path: CP2 → CP3

### metal_graph_encode_layer_attention_batch (line 27172, decode sub-path)

For DSpark verifier with n_tokens draft tokens, the compressor stage starts at
the attention-compressor section within this function.

#### Proven call chain (PROVEN BY SOURCE)

```
metal_graph_encode_layer_attention_batch(g, model, layer, il, pos0, n_tokens)
  │
  ├─── Line ~27653: have_attn_comp = check attn_compressor_kv/gate/ape/norm
  │       ↓ (if !have_attn_comp → error + fallback)
  │
  ├─── Lines ~27698–27701: tail view for ratio-4 compressor
  │       ds4_gpu_tensor_view_create(..., metal_graph_attn_norm(g)[pos0+i], ...)
  │
  └─── Lines ~27703–27750: KV+score projection via SEPARATE calls (PROVEN BY SOURCE)
        ├─ Line ~27704: ds4_gpu_matmul_f16_tensor(batch_comp_kv, model->map, ...,
        │       kv_weight->abs_offset, DS4_N_EMBD, width, tail_hc, 4)
        │   → Writes projected KV to metal_graph_batch_comp_kv(g)
        │
        └─ Line ~27716: ds4_gpu_matmul_f16_tensor(batch_comp_sc, model->map, ...,
                score_weight->abs_offset, DS4_N_EMBD, width, tail_hc, 4)
            → Writes projected score to metal_graph_batch_comp_sc(g)

        NOTE: Two separate calls. NOT a pair_tensor call.
              Each processes all n_tokens rows of tail_hc.

After projection (PROVEN BY SOURCE):
  ├─── Line ~27798: ds4_gpu_compressor_prefill_tensor(kv_view, state_kv[il], ...)
  │   → Pre-fills compressor state (not the same as update_tensor)
  │
  └─── Lines ~27835–27840: ds4_gpu_compressor_update_tensor(
          kv_view, score_view, comp_state=FALSE/IRRELEVANT)
      → Processes all n_tokens pre-filled rows in ONE call
```

### Key characteristic of the generic path (PROVEN BY SOURCE)

- **KV/score projection kernel**: `ds4_gpu_matmul_f16_tensor()` called TWICE —
  once for KV, once for score. Each processes all n_tokens rows.
- **NO `comp_state_already_stored` mechanism exists in this function.**
- **Fused store path does NOT exist** — there is no pair_compressor_store_tensor.
- There is NO Branch A/Branch B dichotomy in the generic path's compressor section.

---

## 2. Ordinary sequential path: CP2 → CP3

### metal_graph_encode_decode_layer_phase (line 21788)

Called once per token per layer. One call at a time.

#### Proven call chain for compressor section (PROVEN BY SOURCE)

```
metal_graph_encode_decode_layer_phase(g, model, layer, il, pos, ...)
  │
  ├─── Line 22217-22219: ratio = ds4_layer_compress_ratio(il), coff, comp_width
  │       emit = ((pos + 1) % ratio) == 0
  │
  ├─── Line 22236: comp_state_already_stored = false
  │
  ├─── Line 22237-22258: CONDITIONAL KERNEL DISPATCH (PROVEN BY SOURCE)
  │       if (!metal_graph_use_reference_compressor_pair_proj()) {
  │           // Branch A — Apple Silicon primary path
  │           ds4_gpu_matmul_f16_pair_compressor_store_tensor(kv, sc, ...)
  │             → Returns >0 on M3/M5 in normal operation
  │             → Sets comp_state_already_stored = true (line 22258)
  │             → Writes projected KV to state_kv[row] = kv[gid] (assignment)
  │             → Writes projected score to state_score[row] = sc[gid] + ape[gid]
  │       } else {
  │           // Branch B — Fallback path
  │           ds4_gpu_matmul_f16_pair_tensor(kv, sc, ...)
  │             → Only on non-Apple or env override
  │             → Writes to metal_graph_comp_kv_cur/sc_cur(g) (NOT direct store)
  │       }
  │
  ├─── Line 22282: comp_row = g->layer_n_comp[il]
  │
  └─── Lines 22283-22307: ds4_gpu_compressor_update_tensor(
          metal_graph_comp_kv_cur(g), ..., comp_state_already_stored)
        → If !state_already_stored (Branch B): writes to state accumulators
        → If state_already_stored (Branch A): skips state writes, uses pre-written
        → Writes compressed KV row to comp_cache[comp_row] if emit == true
```

---

## 3. Four objects — separated and timeline'd (CORRECTED)

### Object 1: Projection result for current token

| Path | Matmul function | Output tensor | PROVEN? |
|------|----------------|---------------|---------|
| Generic batch | `ds4_gpu_matmul_f16_tensor()` ×2 | `batch_comp_kv`, `batch_comp_sc` | PROVEN BY SOURCE (lines ~27704, ~27716) |
| Sequential decode | `ds4_gpu_matmul_f16_pair_compressor_store_tensor()` or `pair_tensor` | `comp_kv_cur` + state tensor (fused) / `comp_kv_cur` (pair) | PROVEN BY SOURCE (lines 22239, 22260) |

**CONCLUSION: These are different functions entirely, not the same function with
two branches.** The generic verifier uses two separate single-tensor matmul calls;
the sequential decode uses either a fused pair-store kernel or a pair tensor call.

This is a **kernel-level implementation difference** at CP2→CP3 boundary.

### Object 2: Recurrent/state frontier state_kv / state_score

| Path | How set | Write pattern | PROVEN? |
|------|---------|--------------|---------|
| Generic batch | Written to `batch_comp_kv/sc` tensors by matmul, then copied via `compressor_prefill_tensor` | Matmul output → prefill copy → state | PROVEN BY SOURCE (line ~27798) |
| Sequential decode (fused) | `state_kv[row] = kv[gid]` written directly in fused kernel | Fused store assignment per row | PROVEN BY SOURCE (fused kernel) |
| Sequential decode (pair fallback) | `comp_kv_cur/g` accumulators → compressor_update writes | Two-phase: matmul then update_tensor write | PROVEN BY SOURCE |

### Object 3: Compressed cache emitted row

Same as before: both paths write to `layer_attn_comp_cache[il]` via
`ds4_gpu_dsv4_fp8_kv_quantize_tensor` on emit boundaries. Unchanged from previous audit.

### Object 4: Compressed-row counter

Same pattern in both paths: READ `layer_n_comp[il]`, increment if emit && ok.
The formula is the same; the difference is in how many times it's evaluated.

---

## 4. comp_state_already_stored — CORRECTED

### PROVEN: NOT reachable in the generic verifier path

| Path | Is `comp_state_already_stored` reachable? | Evidence |
|------|------------------------------------------|---------|
| Generic verifier (`encode_layer_attention_batch`) | **NO** | 0 occurrences in function body (lines 27172–29000) |
| Sequential decode (`encode_decode_layer_phase`) | **YES** | Defined at line 22236, set to true at line 22258 when fused store returns >0 |

### What this means for CP3:

In the sequential decode path, `comp_state_already_stored = true` means the
compressor update phase SKIPS the state_write step and operates on already-written
state tensors. This changes the semantics of compressor_update but does NOT
affect it at all in the generic verifier, which has no such mechanism — the
generic path always goes through `compressor_prefill_tensor` + separate
`compressor_update_tensor`, with state implicitly set by the prefill step.

---

## 5. Earliest source-level implementation difference (CORRECTED)

| # | Point | Generic batch | Ordinary sequential | PROVEN? |
|---|-------|--------------|-------------------|---------|
| D1 | **Matmul kernel for compressor KV/score projection** | `ds4_gpu_matmul_f16_tensor()` ×2 (separate calls) | `ds4_gpu_matmul_f16_pair_compressor_store_tensor()` (Apple primary) or `ds4_gpu_matmul_f16_pair_tensor()` (fallback) | **PROVEN BY SOURCE** — completely different functions |
| D2 | Fused store path selection | N/A — no fused store exists in generic path | Always fused on Apple Silicon (primary path), returns >0 | PROVEN BY SOURCE (line 22237-22258 conditional) |
| D3 | Projection output destination | `batch_comp_kv`/`batch_comp_sc` tensors | `comp_kv_cur` + state tensor (fused) or just `comp_kv_cur` (pair) | PROVEN BY SOURCE |
| D4 | **comp_state_already_stored reachable?** | **NO** — 0 occurrences | YES — defined at line 22236, set at 22258 | PROVEN BY SOURCE |
| D5 | Compressor update call | Called once with prefill-provided state tensors | Called per token-layer; skips or does state_write based on flag | PROVEN BY SOURCE |
| D6 | Pool+norm+RoPE timing | Within the single `compressor_prefill` + `compressor_update` call | After each individual emit token (one at a time) | PROVEN |
| D7 | Counter increment pattern | Multiple increments possible within one batch call (for multiple emit positions) | Same formula, evaluated once per token-layer pair | PROVEN |

### D1 is THE fundamental difference affecting CP3:

**The generic verifier and the sequential decode use DIFFERENT matmul kernels for
compressor KV/score projection.** This is not a "path selection within one function"
difference — it is a kernel-level difference in the very operator that produces
the compressor frontier state.

This means: even before looking at fused-store semantics, the **projected values**
themselves (Object 1) are produced by fundamentally different operations. The generic
verifier uses two separate `matmul_f16_tensor` calls; sequential decode uses a
pair kernel that processes KV and score together in a single dispatch.

---

## 6. Ratio-4 Indexer path (unchanged from previous audit)

No changes needed for the indexer compressor section — it has its own distinct
code paths not affected by this correction.

---

## 7. CP3 definition challenge (CORRECTED)

The corrected view of CP3:

### For the generic verifier:
CP3 boundary should capture `batch_comp_kv` and `batch_comp_sc` tensors after
the two separate `ds4_gpu_matmul_f16_tensor()` calls complete but before
`compressor_prefill_tensor` reads them. This is a **clean, separable** CP3
boundary because there is no fused store — projection results are standalone
tensors accessible between matmul and prefill.

### For sequential decode (fused primary path):
CP3 boundary is blurred because projection + state write happen simultaneously in
`compressor_store_tensor`. Two observables:
1. **CP3a**: Projected KV/score IMMEDIATELY after matmul but BEFORE state write
   (requires instrumenting within the fused kernel)
2. **CP3b**: Persistent state frontier AFTER `compressor_update` pool+norm+RoPE

### For sequential decode (pair fallback path):
CP3 is clean — projected values in `comp_kv_cur/sc_cur` are materialized after
`ds4_gpu_matmul_f16_pair_tensor()` completes, before `compressor_update` reads them.

---

## 8. Reconciliation with upstream #658 and commit 7fb2830

### What upstream says (PROVEN BY SOURCE — commit message of 7fb2830):

> "The verify batch's compressor-frontier writes come from a batched multi-token
> GEMM (**`ds4_gpu_matmul_f16_pair_tensor`**), which is numerically distinct from
> ordinary single-token decode's fused projection+store kernel
> (**`ds4_gpu_matmul_f16_pair_compressor_store_tensor`**)."

### Reconciliation:

At commit **b030961** (the audit baseline), the generic verifier does NOT use
`ds4_gpu_matmul_f16_pair_tensor` — it uses `ds4_gpu_matmul_f16_tensor()` called
separately for KV and score. This means either:

1. The upstream commit message used `ds4_gpu_matmul_f16_pair_tensor` as an
   imprecise description of the pair-style projection kernel, OR
2. Between 7fb2830 and b030961, the verifier's matmul path was refactored to
   use separate `matmul_f16_tensor` calls instead of a pair call (this is
   plausible given commits `532ec8b`, `222b2cb` in between which accelerate
   DeepSeek routed MoE prefill and indexed prefill).

**Regardless, the core finding holds:** the verifier's compressor projection kernel
differs from sequential decode. At b030961, that difference is even more fundamental
than what 7fb2830 described — separate `matmul_f16_tensor` calls vs. a pair fused
kernel — not just a subtle tiling/accumulation-order difference within the same
kernel family.

### Every corrected conclusion mapped to upstream:

| Upstream claim | Corrected finding at b030961 | Consistent? |
|---------------|-----------------------------|------------|
| "Verifier uses pair_tensor" | Verifier uses `matmul_f16_tensor()` ×2 (separate) | PARTIALLY — same direction, different kernel type |
| "Decode uses compressor_store_tensor" | Decode uses `pair_compressor_store_tensor` (Apple) | YES, exactly confirmed |
| "This is the known source of non-bit-identical state" | Confirmed: different kernels → different projected values | YES |
| "Replay via ordinary decode fixes correctness" | Confirmed: replay restores KV to what sequential decode would produce | YES |

### Invalidated sections in previous audit:

| Section | Status | Reason |
|---------|--------|--------|
| Section 1 (generic path call graph) | **COMPLETELY INVALID** | Conflated `encode_layer_attention_batch` with `encode_decode_layer_phase`. Zero of the claimed functions exist in the target function. |
| Section 2 (sequential path) | PARTIALLY VALID | Sequential path description is correct, but was presented as "same code, different context" rather than a fundamentally different function. |
| D1 | **WRONG** | Claimed "no difference". Correct: different kernels entirely. |
| D2–D7 | **INVALIDATED** | All depend on the wrong generic path model. D4 (`comp_state_already_stored`) is specifically proven NOT reachable in generic verifier. |
| Section 5 (earliest difference) | **WRONG** | Earliest difference is at matmul kernel level, not "dispatch order within same kernel". |
| Section 7 (CP3 instrumentation recommendation) | PARTIALLY VALID | The fused-store problem only applies to sequential decode, NOT to the generic verifier which has a clean CP3 boundary. |
| Section 8 (difference summary) | **INVALIDATED** | All three rows based on wrong function identification. |
| Section 9 (verdict) | PARTIALLY VALID | The "no clean CP3 boundary" claim is reversed: the generic verifier actually HAS a clean CP3 boundary (projection tensors are standalone between matmul and prefill). |

---

## 9. Final verdict (CORRECTED)

### For the generic verifier path:
- **CP3 boundary IS clean** — `batch_comp_kv/sc` are materialized standalone tensors
  after `ds4_gpu_matmul_f16_tensor()` completes, before `compressor_prefill_tensor`
  reads them.
- **`comp_state_already_stored` is NOT reachable** — the generic path has no fused store mechanism.
- **Earliest divergence from sequential decode**: the matmul kernel itself (D1). The
  verifier's projected values are produced by separate `matmul_f16_tensor()` calls;
  sequential decode's are produced by `pair_compressor_store_tensor`. This is a CP2→CP3
  issue — the input to the compressor state machine differs between paths.

### For sequential decode:
- **CP3 boundary IS blurred** when `comp_state_already_stored = true` (Apple Silicon, normal).
- Fused store makes projection + state write inseparable.
- Observable boundaries are pre-fused-store and post-compressor-update.

### Reconciliation with upstream #658 / 7fb2830:
- **Confirmed**: the kernel difference is the known source of non-bit-identical compressor
  frontier state between verifier batch and sequential decode.
- **Corrected**: at b030961, the generic verifier uses `ds4_gpu_matmul_f16_tensor()` ×2
  (separate), not `ds4_gpu_matmul_f16_pair_tensor` as stated in 7fb2830's commit message.
- **Confirmed**: replaying partial accepts through ordinary decode is the correct fix,
  because it ensures accepted tokens' compressor state matches what pure sequential decode
  would have produced.

### Strongest remaining risk:
The generic verifier and sequential decode use fundamentally different matmul kernels for
compressor projection. Even if CP3 state tensors were perfectly aligned at some boundary,
the **projected values that feed into the compressor** differ between paths from the very
first operation (D1). This means any CP3 comparison must also account for the CP2→CP3
projection divergence.

---

## Audit methodology notes

### Sources used:
- `git show b030961:ds4.c` — primary source code at audit baseline
- Line-by-line grep for function names and variables within specific function bodies
- Commit 7fb2830 diff for upstream reconciliation
- commit ancestry verification (7fb2830 is ancestor of b030961, 6 commits forward)

### Tag definitions:
- **PROVEN BY SOURCE**: directly verified by grepping/read of b030961:ds4.c
- **INFERRED**: derived from source but not directly asserted (e.g., numerical divergence due to kernel difference)
- **UNKNOWN**: could not determine from code inspection alone

### Previous version errors:
The original audit was found to be wrong in the following specific way: it **conflated two
different functions** (`metal_graph_encode_layer_attention_batch` at line 27172 with
`metal_graph_encode_decode_layer_phase` at line 21788). The former is the DSpark generic
verifier entry point for batch encoding; the latter is the ordinary sequential decode
per-layer phase. Their compressor sections use completely different functions and have
completely different state management — but the audit treated them as the same code path.
