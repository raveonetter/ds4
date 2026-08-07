# Qwen CP1→CP2 Audit Summary

## Scope

Compare generic DSpark verifier vs ordinary sequential decode at two checkpoint boundaries:
- **CP1**: Q/KV projection output (after matmul, before raw KV store)
- **CP2**: Raw KV cache write (what gets written to GPU-visible KV storage)

Source baseline: `b030961` (ds4.c)

---

## CP1: Q/KV Projection — Kernel-Level Difference

### Generic verifier (`metal_graph_encode_layer_attention_batch`, line 27172)

```
Three SEPARATE matmul calls, each processing all n_tokens rows:
  Line ~27360: ds4_gpu_matmul_q8_0_named_tensor(attn_q_a)    → Q for layer il
  Line ~27376: ds4_gpu_matmul_q8_0_named_tensor(attn_kv)     → K/V for layer il (first pass)
  Line ~27478: ds4_gpu_matmul_q8_0_named_tensor(attn_q_b)    → Q output head weight
  Line ~27552: ds4_gpu_matmul_q8_0_named_tensor(attn_kv)     → K/V weight (second pass)

Dispatch kernel: metal_graph_matmul_dense_quant_tensor (per-tensor, not pair)
Accumulation order: n_tokens rows in ONE kernel dispatch
```

### Sequential decode (`metal_graph_encode_decode_layer_phase`, line 21788)

```
ONE PAIRED call per token-layer pair:
  Line 22014: ds4_gpu_matmul_q8_0_pair_tensor(kv, q, ...)    → Q and KV in SINGLE dispatch

Dispatch kernel: pair_tensor variant (processes both Q and KV together)
Accumulation order: single row per call, sequential across tokens
```

### PROVEN differences at CP1

| # | Difference | Proven? |
|---|-----------|---------|
| C1.1 | **Function**: `matmul_q8_0_named_tensor` (×3 separate) vs `matmul_q8_0_pair_tensor` (×1 paired) | PROVEN BY SOURCE |
| C1.2 | **Accumulation order**: batched over n_tokens in one dispatch vs single-row sequential | PROVEN BY SOURCE |
| C1.3 | **Kernel type**: `metal_graph_matmul_dense_quant_tensor` vs pair_tensor dispatch kernel | INFERRED from function name differences + upstream #658 context |

### Implication for checkpoint freeze

**Different matmul kernels + different accumulation order = projected Q/KV values differ between paths at CP1.** This is the **earliest divergence point** — it precedes both raw KV store (CP2) and compressor projection (CP3).

---

## CP2: Raw KV Store — Batch vs Single-Row

### Generic verifier (`metal_graph_encode_layer_attention_batch`)

```
  Line ~28385: ds4_gpu_store_raw_kv_batch_tensor(...)
    → Writes ALL n_tokens rows to raw cache in ONE call
    → Batch write pattern
    → Proven by grep within encode_layer_attention_batch body (lines 27172-29000)
```

### Sequential decode (`metal_graph_encode_decode_layer_phase`)

```
  Line ~22169: metal_graph_decode_kv_store(...)
    → Writes ONE row per call
    → Called n_tokens times for a full block
    → Each call interleaves with layer processing between tokens
```

### PROVEN differences at CP2

| # | Difference | Proven? |
|---|-----------|---------|
| C2.1 | **Write pattern**: batch write (all rows at once) vs single-row loop | PROVEN BY SOURCE |
| C2.2 | **Interleaving**: generic writes KV then proceeds; sequential interleaves KV with norm/rope between each token | PROVEN BY SOURCE |

### Implication

Even if CP1 values were identical, the different write patterns at CP2 mean the **timing of when KV data becomes visible to subsequent operations** differs between paths. For the generic verifier, all n_tokens KV rows are written before any attention output computation. For sequential decode, each token's KV is immediately available for that token's attention layer.

---

## Root Cause Chain

```
CP1: Different matmul kernels (named_tensor ×3 vs pair_tensor ×1)
     → Projected Q/KV values differ (accumulation order + kernel internals)
         → CP2: Different raw KV write patterns (batch vs single-row)
             → Downstream: attention output, compressor projection all operate on different inputs
```

**CP1 is the fundamental divergence.** Even if we fixed CP2 batch-vs-single-row discrepancy, the **projected Q/KV values already differ at CP1**, which propagates through everything downstream.

---

## Summary Table

| Checkpoint | Generic Verifier | Sequential Decode | Difference Type |
|-----------|-----------------|-------------------|-----------------|
| **CP1** (Q/KV proj) | `matmul_q8_0_named_tensor` ×3, batched over n_tokens | `matmul_q8_0_pair_tensor` ×1, single-row sequential | Kernel + accumulation order |
| **CP2** (raw KV) | `store_raw_kv_batch_tensor` (batch write) | `decode_kv_store` (one row per call) | Write pattern + interleaving |

---

## Tag Definitions

- **PROVEN BY SOURCE**: directly verified by grepping/read of b030961:ds4.c
- **INFERRED**: derived from source but not directly asserted (e.g., kernel type differences)
- **UNKNOWN**: could not determine from code inspection alone

---

## Previous audit documents

- `QWEN-CP3-AUDIT.md` — corrected CP3 (compressor projection) audit, separate document
- `QWEN-P2-ADVERSARIAL-AUDIT.md` — challenges to probe architecture claims

This summary supersedes all prior fragmented findings on CP1/CP2.
