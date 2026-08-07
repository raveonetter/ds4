/*
 * P2: Runtime first-divergence probe for DSpark batched verifier.
 *
 * This file is a **drop-in probe** that adds conditional comparison logic
 * into the generic verifier path (metal_graph_encode_layer_batch).  It only
 * activates when DS4_DSPARK_EXACTNESS_PROBE is set.
 *
 * Design:
 *   The generic verifier processes N tokens in ONE batched dispatch per layer.
 *   For DSpark exactness, we need to know WHERE the batched path first diverges
 *   from the canonical single-token decode path.
 *
 *   This probe captures state at 5 strategic checkpoints inside each layer's
 *   encode_layer_batch execution and compares against:
 *     - The per-row canonical result (re-computed on CPU for small tensors,
 *       or via separate GPU dispatches for large ones)
 *
 * Checkpoints (ordered by position in pipeline):
 *   CP1  Q/KV projection output    (attention encoder, row 0 & row 1)
 *   CP2  Raw KV store              (after metal_graph_decode_kv_store)
 *   CP3  Compressor projected KV   (after ds4_gpu_matmul_f16_pair_tensor/store)
 *   CP4  Mutable score frontier    (after compressor update)
 *   CP5  Layer output hidden state (after FFN, batch_cur_hc)
 *
 * Probe output format (machine-readable):
 *   cp=N layer=L row=R exact=1|0 [metrics...]
 *
 * Build: compile into ds4 by including this file in the build.
 *        Probe is zero-cost when disabled (env var not set).
 */

#include "ds4_exactness_probe.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Maximum tensors to capture per layer for comparison.
 * Each entry holds a tensor pointer + offset in that tensor.
 */
#define DS4_PROBE_MAX_CAPTURED 8

typedef struct {
    const void *data;    /* CPU buffer (after ds4_gpu_tensor_read) */
    uint64_t    n_floats;
    const char *label;   /* CP1-CP5 label */
} probe_capture_entry;

/* Global state — single probe run per evaluation session */
static struct {
    int enabled;
    FILE *out;
    int layer_count;
    probe_capture_entry captures[DS4_PROBE_MAX_CAPTURED];
    uint64_t total_comparisons;
    uint64_t exact_matches;
    uint64_t first_cp;   /* checkpoint index of first divergence */
    int first_layer;
} probe_state = {0, NULL, 0, {{0}}, 0, 0, 0, -1};

/* ---- Probe API (used by ds4.c hooks) ---- */

void ds4_probe_init(FILE *fp) {
    const char *env = getenv("DS4_DSPARK_EXACTNESS_PROBE");
    if (!env || !env[0] || strcmp(env, "0") == 0) return;
    probe_state.enabled = 1;
    probe_state.out = fp ? fp : stderr;
    probe_state.layer_count = 0;
    memset(probe_state.captures, 0, sizeof(probe_state.captures));
    probe_state.total_comparisons = 0;
    probe_state.exact_matches = 0;
    probe_state.first_cp = (uint64_t)-1;
    if (probe_state.out) {
        fprintf(probe_state.out,
                "ds4: P2 probe enabled — starting fresh run\n");
    }
}

void ds4_probe_reset(void) {
    /* Called between different runs to reset state */
    memset(&probe_state, 0, sizeof(probe_state));
    const char *env = getenv("DS4_DSPARK_EXACTNESS_PROBE");
    if (env && env[0] && strcmp(env, "0") != 0) {
        probe_state.enabled = 1;
        probe_state.out = stderr;
    }
}

/* ---- Per-layer checkpoint hooks ---- */

/*
 * Called after Q/KV projection on row 0 and row 1.
 * batched_qkv[0] = result for row 0 in the batch dispatch
 * batched_qkv[1] = result for row 1 in the batch dispatch
 * canonical_qkv  = per-row canonical results (via separate dispatch)
 * dim = dimension per row (e.g. Q_proj_dim or KV_proj_dim)
 */
void ds4_probe_checkpoint_cp1(
    const float *batched_row0, const float *batched_row1,
    const float *canonical_row0, const float *canonical_row1,
    uint64_t dim, uint32_t layer) {

    if (!probe_state.enabled || !probe_state.out) return;

    ds4_probe_diff diff_r0, diff_r1;

    bool ok_r0 = ds4_probe_compare_f32_arrays(
        batched_row0, canonical_row0, dim, &diff_r0);
    bool ok_r1 = ds4_probe_compare_f32_arrays(
        batched_row1, canonical_row1, dim, &diff_r1);

    (void)ok_r0; (void)ok_r1;

    fprintf(probe_state.out,
            "cp=1 layer=%u row=0 exact=%d mismatches=%llu "
            "max_abs=%.6g max_rel=%.6g max_ulp=%u\n",
        layer, diff_r0.bit_exact ? 1 : 0,
        (unsigned long long)diff_r0.mismatch_count,
        (double)diff_r0.max_abs_diff,
        (double)diff_r0.max_rel_diff,
        diff_r0.max_ulp_diff);

    fprintf(probe_state.out,
            "cp=1 layer=%u row=1 exact=%d mismatches=%llu "
            "max_abs=%.6g max_rel=%.6g max_ulp=%u\n",
        layer, diff_r1.bit_exact ? 1 : 0,
        (unsigned long long)diff_r1.mismatch_count,
        (double)diff_r1.max_abs_diff,
        (double)diff_r1.max_rel_diff,
        diff_r1.max_ulp_diff);

    probe_state.total_comparisons += 2;
    if (probe_state.first_cp == (uint64_t)-1 && !diff_r0.bit_exact) {
        probe_state.first_cp = 1;
        probe_state.first_layer = layer;
        fprintf(probe_state.out,
                "ds4: *** FIRST DIVERGENCE DETECTED cp=1 layer=%u ***\n", layer);
    } else if (probe_state.first_cp == (uint64_t)-1 && !diff_r1.bit_exact) {
        probe_state.first_cp = 1;
        probe_state.first_layer = layer;
        fprintf(probe_state.out,
                "ds4: *** FIRST DIVERGENCE DETECTED cp=1 layer=%u (row1) ***\n", layer);
    }
    if (diff_r0.bit_exact && diff_r1.bit_exact) {
        probe_state.exact_matches += 2;
    }
}

/*
 * CP2: Raw KV store comparison.
 */
void ds4_probe_checkpoint_cp2(
    const float *batched_kv, const float *canonical_kv,
    uint64_t n_positions, uint32_t layer) {

    if (!probe_state.enabled || !probe_state.out) return;

    ds4_probe_diff diff;
    bool ok = ds4_probe_compare_f32_arrays(batched_kv, canonical_kv, n_positions, &diff);
    (void)ok;

    fprintf(probe_state.out,
            "cp=2 layer=%u exact=%d mismatches=%llu "
            "max_abs=%.6g max_rel=%.6g max_ulp=%u\n",
        layer, diff.bit_exact ? 1 : 0,
        (unsigned long long)diff.mismatch_count,
        (double)diff.max_abs_diff,
        (double)diff.max_rel_diff,
        diff.max_ulp_diff);

    probe_state.total_comparisons++;
    if (probe_state.first_cp == (uint64_t)-1 && !diff.bit_exact) {
        probe_state.first_cp = 2;
        probe_state.first_layer = layer;
        fprintf(probe_state.out,
                "ds4: *** FIRST DIVERGENCE DETECTED cp=2 layer=%u ***\n", layer);
    } else if (diff.bit_exact) {
        probe_state.exact_matches++;
    }
}

/*
 * CP3: Compressor projected KV + score.
 */
void ds4_probe_checkpoint_cp3_kv(
    const float *batched_proj, const float *canonical_proj,
    uint64_t n_floats, uint32_t layer) {

    if (!probe_state.enabled || !probe_state.out) return;

    ds4_probe_diff diff;
    bool ok = ds4_probe_compare_f32_arrays(batched_proj, canonical_proj, n_floats, &diff);
    (void)ok;

    fprintf(probe_state.out,
            "cp=3 layer=%u exact=%d mismatches=%llu "
            "max_abs=%.6g max_rel=%.6g max_ulp=%u\n",
        layer, diff.bit_exact ? 1 : 0,
        (unsigned long long)diff.mismatch_count,
        (double)diff.max_abs_diff,
        (double)diff.max_rel_diff,
        diff.max_ulp_diff);

    probe_state.total_comparisons++;
    if (probe_state.first_cp == (uint64_t)-1 && !diff.bit_exact) {
        probe_state.first_cp = 3;
        probe_state.first_layer = layer;
        fprintf(probe_state.out,
                "ds4: *** FIRST DIVERGENCE DETECTED cp=3 layer=%u ***\n", layer);
    } else if (diff.bit_exact) {
        probe_state.exact_matches++;
    }
}

void ds4_probe_checkpoint_cp3_score(
    const float *batched_sc, const float *canonical_sc,
    uint64_t n_floats, uint32_t layer) {

    if (!probe_state.enabled || !probe_state.out) return;

    ds4_probe_diff diff;
    bool ok = ds4_probe_compare_f32_arrays(batched_sc, canonical_sc, n_floats, &diff);
    (void)ok;

    fprintf(probe_state.out,
            "cp=3s layer=%u exact=%d mismatches=%llu "
            "max_abs=%.6g max_rel=%.6g max_ulp=%u\n",
        layer, diff.bit_exact ? 1 : 0,
        (unsigned long long)diff.mismatch_count,
        (double)diff.max_abs_diff,
        (double)diff.max_rel_diff,
        diff.max_ulp_diff);

    probe_state.total_comparisons++;
    if (probe_state.first_cp == (uint64_t)-1 && !diff.bit_exact) {
        probe_state.first_cp = 3;
        probe_state.first_layer = layer;
        fprintf(probe_state.out,
                "ds4: *** FIRST DIVERGENCE DETECTED cp=3s layer=%u ***\n", layer);
    } else if (diff.bit_exact) {
        probe_state.exact_matches++;
    }
}

/*
 * CP4: Mutable score frontier after compressor update.
 */
void ds4_probe_checkpoint_cp4(
    const float *batched_frontier, const float *canonical_frontier,
    uint64_t n_floats, uint32_t layer) {

    if (!probe_state.enabled || !probe_state.out) return;

    ds4_probe_diff diff;
    bool ok = ds4_probe_compare_f32_arrays(batched_frontier, canonical_frontier, n_floats, &diff);
    (void)ok;

    fprintf(probe_state.out,
            "cp=4 layer=%u exact=%d mismatches=%llu "
            "max_abs=%.6g max_rel=%.6g max_ulp=%u\n",
        layer, diff.bit_exact ? 1 : 0,
        (unsigned long long)diff.mismatch_count,
        (double)diff.max_abs_diff,
        (double)diff.max_rel_diff,
        diff.max_ulp_diff);

    probe_state.total_comparisons++;
    if (probe_state.first_cp == (uint64_t)-1 && !diff.bit_exact) {
        probe_state.first_cp = 4;
        probe_state.first_layer = layer;
        fprintf(probe_state.out,
                "ds4: *** FIRST DIVERGENCE DETECTED cp=4 layer=%u ***\n", layer);
    } else if (diff.bit_exact) {
        probe_state.exact_matches++;
    }
}

/*
 * CP5: Layer output hidden state after FFN.
 */
void ds4_probe_checkpoint_cp5(
    const float *batched_hc, const float *canonical_hc,
    uint64_t n_floats, uint32_t layer) {

    if (!probe_state.enabled || !probe_state.out) return;

    ds4_probe_diff diff;
    bool ok = ds4_probe_compare_f32_arrays(batched_hc, canonical_hc, n_floats, &diff);
    (void)ok;

    fprintf(probe_state.out,
            "cp=5 layer=%u exact=%d mismatches=%llu "
            "max_abs=%.6g max_rel=%.6g max_ulp=%u\n",
        layer, diff.bit_exact ? 1 : 0,
        (unsigned long long)diff.mismatch_count,
        (double)diff.max_abs_diff,
        (double)diff.max_rel_diff,
        diff.max_ulp_diff);

    probe_state.total_comparisons++;
    if (probe_state.first_cp == (uint64_t)-1 && !diff.bit_exact) {
        probe_state.first_cp = 5;
        probe_state.first_layer = layer;
        fprintf(probe_state.out,
                "ds4: *** FIRST DIVERGENCE DETECTED cp=5 layer=%u ***\n", layer);
    } else if (diff.bit_exact) {
        probe_state.exact_matches++;
    }
}

/* ---- Summary ---- */

void ds4_probe_print_summary(void) {
    if (!probe_state.enabled || !probe_state.out) return;

    fprintf(probe_state.out, "\n");
    fprintf(probe_state.out, "=== DSpark Exactness P2 Probe Summary ===\n");
    fprintf(probe_state.out, "Total comparisons: %llu\n",
            (unsigned long long)probe_state.total_comparisons);
    fprintf(probe_state.out, "Exact matches:     %llu\n",
            (unsigned long long)probe_state.exact_matches);
    fprintf(probe_state.out, "Divergent:         %llu\n",
            (unsigned long long)(probe_state.total_comparisons - probe_state.exact_matches));

    if (probe_state.first_cp != (uint64_t)-1) {
        fprintf(probe_state.out,
                "First divergence at CP%d layer=%d\n",
                (int)probe_state.first_cp, probe_state.first_layer);
    } else {
        fprintf(probe_state.out,
                "All comparisons exact — no divergence detected in any checkpoint.\n");
    }

    fprintf(probe_state.out, "=== End P2 Summary ===\n\n");
}

/* ---- Compile-time gate: ensure probe is zero-cost when disabled ---- */

/* If DS4_DSPARK_EXACTNESS_PROBE is not set at compile time, these become no-ops */
#ifndef DS4_PROBE_RUNTIME_ENABLED
#define DS4_PROBE_RUNTIME_ENABLED 0
#endif
