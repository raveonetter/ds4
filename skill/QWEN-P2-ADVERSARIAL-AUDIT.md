# Qwen P2 Adversarial Audit — Two Critical Claims Challenge

## Scope

Challenge two claims in Codex's C0 audit (`CODEX-P2-C0-AUDIT.md`):

1. Append-only compressor/indexer rows invisible after counter restore
2. Probe blit copy does not perturb numerical execution

No code was written or modified. Source traced independently from upstream ds4.c.

---

## Question 1: Append-Only Cache Rows — Counter Restore Completeness

### Codex Claim

> "future rows are invisible after counter restore and an emitting sequential token overwrites its current append row before use"

### Source Trace

| Fact | Line in ds4.c |
|------|---------------|
| Compressor write pattern: `comp_row = g->layer_n_comp[il]`, then write, then increment | 22282, 22283-22307, 22324 |
| No ring-wrap in compressed cache — pure append-only until ctx_size full | 22332-22334 (overflow error) |
| Attention reads ALL rows [0, n_comp) from compressed cache | ds4_gpu_attention_decode_heads_tensor (n_comp passed as row count) |
| Restore resets counter + state tensors | 50012, 50017-50026 |

### Counter-Example: Fused Store Bypass

**Lines 22237-22258**: `ds4_gpu_matmul_f16_pair_compressor_store_tensor` has a `comp_state_already_stored = true` path that writes compressed rows directly to comp_cache, bypassing state_kv/state_score tensors.

If this branch triggers during Pass A:
```
Pass A fused store → new comp_cache rows at [saved_n_comp, saved_n_comp + Δ)
Restore → copies state_kv/score back, resets n_comp, DOES NOT touch comp_cache contents
Post-restore attention at layer L+1 → reads comp_cache[0..n_comp), skips stale data
Subsequent decode overwrites stale data when n_comp catches up
```

The counter-example **does not invalidate** the claim because attention kernel respects `n_comp` as row count, and subsequent writes overwrite stale data before it's ever read. But this is verified only for the default path — **the `comp_state_already_stored` fused-store branch must be confirmed as inactive during Pass A**.

### Verdict: WEAKENED but not invalidated

The appendix-only cache argument holds in its current form. The strongest remaining risk is the fused store bypass: if `comp_state_already_stored = true` during generic verifier, the state accumulation path diverges from single-token decode's canonical path before the comp_cache write. This means even with correct counter restore, the **state_kv/score tensors** may not match between Pass A and Pass B at the point where compressor_update reads them — but this is a **Pass B correctness issue**, not an append-only visibility issue.

---

## Question 2: Probe Blit Copy Perturbation

### Codex Claim

> "preserves command order within the command buffer; performs no CPU tensor readback"
> → `ds4_gpu_tensor_copy()` suitable for snapshotting without perturbation

### Source Trace

| Fact | Line in ds4_metal.m / ds4.c |
|------|----------------------------|
| `ds4_gpu_tensor_copy` calls `ds4_gpu_close_batch_encoder()` (ends compute encoder) | 8086 |
| Creates MTLBlitCommandEncoder within SAME g_batch_cb | 8087-8095 |
| One command buffer for entire pass in default mode | ds4.c line 34496 begin, 34562 end |
| All layers contribute to single `g_batch_enc` compute encoder | ds4.c lines 29595-29616 |
| No implicit fence between encoders — resource dependency only | Metal spec + source trace |

### 唯一 proven fact

**Encoder boundary disruption (proven by C source):**
`ds4_gpu_close_batch_encoder()` terminates the current compute encoder before blit, and a new compute encoder is created after. Default mode: 40 layers → ONE encoder. With probe: N encoders + N blits. **This is the ONLY claim fully supported by static code analysis.**

### Unproven speculation (independent claims — no causal chain)

The following are **independent hypotheses**, each requiring GPU microarchitecture verification. None can be derived from C source alone, and none causally imply the others:

| Hypothesis | Status | Why unproven |
|---|---|---|
| Cache lines flushed at encoder boundary | No evidence | Encoder boundary guarantees dependency ordering, not cache invalidation |
| Kernel fusion prevented by multi-encoder topology | No evidence | Metal fusion rules are driver internals; opaque from C source |
| GPU scheduling priorities shift with encoder count | No evidence | Apple Silicon scheduling heuristics are not public nor in our codebase |
| FP accumulation order changes across boundaries | No evidence | Same mathematical formula ≠ same floating-point result; requires GPU verification |

**Bottom line:** Only the command encoding structure change is proven. Everything about its effect on GPU execution semantics is speculation. The only way to verify any of these hypotheses is runtime comparison (probe-on vs probe-off).

### Verdict: WEAKENED

Codex's dependency correctness argument is solid. But the stronger claim "probe does not perturb what we're measuring" is **not proven by static analysis**. The most serious implication: if blit perturbation changes floating-point accumulation order, the differential experiment measures probe-induced artifacts rather than genuine DSpark bugs.

---

## Summary Table

| Codex claim | Strongest counterexample | Source evidence | Verdict |
|---|---|---|---|
| App-only rows invisible after counter restore | Fused store bypass: `comp_state_already_stored=true` skips state tensor path | ds4.c 22237-22258 | WEAKENED (but not invalidated) |
| Probe blit copy doesn't perturb numerical execution | Only proven: encoder split (1→N). GPU semantics effect is speculation, not disproven by source. | ds4_metal.m 8086 | WEAKENED (only encoding structure proven) |

---

## Strongest Remaining Risk

**Blit copies change command encoding structure (proven).** Whether this changes the *result* of the generic verifier computation (vs merely changing *how* it's encoded) is unproven by static analysis and requires runtime verification. The differential experiment measures the gap between Pass A and Pass B — if encoder split alters the FP accumulation in Pass A, the measured "divergence" may include probe artifacts.

### What Would Falsify the GO Decision

Runtime observation: **probe-on vs probe-off outputs for Pass A differ** (even without running Pass B). This demonstrates that the blit copies perturb the computation we're trying to measure, making the differential inconclusive.

### Recommendation

**GO with additional prerequisite:** Before comparing generic vs ordinary outputs, validate that `ds4_verify_suffix_tops()` with probe enabled produces bit-identical results to probe disabled (layer-by-layer tensor comparison on a known-good path). If this validation passes, the differential experiment is credible. If it fails, the probe architecture needs redesign to minimize encoder boundary impact (e.g., batch all blits at layer end instead of after each stage, or use compute-shader-based copies that don't require encoder termination).

---

## What NOT to do

- Do not propose code changes
- Do not modify any files
- Do not commit or push
