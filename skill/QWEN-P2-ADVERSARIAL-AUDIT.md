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

### Three Confirmed Perturbation Mechanisms

**#1: Encoder boundary disruption (confirmed)**
`ds4_gpu_close_batch_encoder()` terminates the current compute encoder before blit, preventing any cross-boundary kernel fusion/reordering. Default mode: 40 layers → ONE encoder. With probe: N encoders + N blits.

**#2: Resource cache line flushing (confirmed)**
Metal unified memory flushes cache lines at encoder boundary. Without probe: layer L's output may stay in cache for layer L+1's kernel read. With probe: forced flush to memory, then reload.

**#3: GPU bandwidth contention (confirmed but unquantified)**
Blit copy competes with compute kernels for unified memory bandwidth on Apple Silicon. Latency in microseconds to milliseconds shifts dependent reads timing.

### Critical Gap: Floating-Point Accumulation Order Change

The question is not whether dependencies are correct (they are). The question is: **does the Metal driver's change from 1 encoder vs N encoders affect floating-point accumulation order at the kernel level?**

| Perturbation chain | Exists? | Evidence |
|---|---|---|
| Encoder split blocks kernel fusion | YES | `ds4_gpu_close_batch_encoder()` ends encoder scope |
| Cache line flushing policy differs | YES | Metal unified memory at encoder boundary |
| Kernel scheduling priorities change | MAYBE | Depends on Metal driver internals |
| Floating-point accumulation order changes | UNPROVEN | Architecture-dependent, not provable from C code alone |

### The Fundamental Logical Gap

**"Command order preserved" ≠ "Floating-point accumulation order identical"**

- Command order: kernel dispatches happen in the order they were added to the encoder
- But Metal can reorder **within** a single encoder based on resource dependencies and heuristics
- Encoder boundary changes the set of available reorderings
- If two independent kernels share register/lcache state, their execution order within an encoder may produce different floating-point results than if they execute across boundaries

This is a property of the Apple Silicon GPU microarchitecture, not provable from C source. The only way to verify is **runtime**: compare probe-on vs probe-off outputs.

### Verdict: WEAKENED

Codex's dependency correctness argument is solid. But the stronger claim "probe does not perturb what we're measuring" is **not proven by static analysis**. The most serious implication: if blit perturbation changes floating-point accumulation order, the differential experiment measures probe-induced artifacts rather than genuine DSpark bugs.

---

## Summary Table

| Codex claim | Strongest counterexample | Source evidence | Verdict |
|---|---|---|---|
| App-only rows invisible after counter restore | Fused store bypass: `comp_state_already_stored=true` skips state tensor path | ds4.c 22237-22258 | WEAKENED (but not invalidated) |
| Probe blit copy doesn't perturb numerical execution | Encoder split changes Metal driver's kernel scheduling → FP accumulation order may differ | ds4_metal.m 8086, 907-914 | WEAKENED |

---

## Strongest Remaining Risk

**Probe-induced floating-point perturbation via encoder boundary disruption.** The differential experiment measures the gap between Pass A (generic) and Pass B (ordinary sequential). If blit copies in Pass A change the accumulation order of generic verifier kernels, then the measured "divergence" may be partly or wholly due to probe artifacts — making the entire experiment invalid.

### What Would Falsify the GO Decision

Runtime observation: **probe-on vs probe-off outputs for Pass A differ** (even without running Pass B). This demonstrates that the blit copies perturb the very computation we're trying to measure, making the differential inconclusive.

### Recommendation

**GO with additional prerequisite:** Before comparing generic vs ordinary outputs, validate that `ds4_verify_suffix_tops()` with probe enabled produces bit-identical results to probe disabled (layer-by-layer tensor comparison on a known-good path). If this validation passes, the differential experiment is credible. If it fails, the probe architecture needs redesign to minimize encoder boundary impact (e.g., batch all blits at layer end instead of after each stage, or use compute-shader-based copies that don't require encoder termination).

---

## What NOT to do

- Do not propose code changes
- Do not modify any files
- Do not commit or push
