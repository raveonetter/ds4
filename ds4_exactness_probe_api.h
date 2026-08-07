/*
 * P2 probe API for DSpark exactness investigation.
 *
 * These functions are called from ds4.c at 5 checkpoint positions during
 * the generic verifier path (metal_graph_encode_layer_batch).
 *
 * The probe is model-independent: it only needs tensor pointers and dimensions.
 * All comparison logic lives in ds4_exactness_probe.c.
 *
 * This header is for inclusion inside ds4.c ONLY (not a public API).
 */

#include "ds4_exactness_probe.h"

/* ---- Probe enable check — zero cost when disabled ---- */

static int ds4_probe_is_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("DS4_DSPARK_EXACTNESS_PROBE");
        cached = (env && env[0] && strcmp(env, "0") != 0);
    }
    return cached;
}

/* ---- P2 probe session state (static local per call) ---- */

typedef struct {
    int enabled;
    FILE *fp;
    uint64_t total_comparisons;
    uint64_t exact_matches;
    uint64_t first_cp;      /* first divergent checkpoint (1-5) */
    int first_layer;        /* layer of first divergence */
} ds4_p2_probe_session;

static void ds4_probe_session_init(ds4_p2_probe_session *s, FILE *fp) {
    memset(s, 0, sizeof(*s));
    s->fp = fp ? fp : stderr;
    s->enabled = ds4_probe_is_enabled();
    s->total_comparisons = 0;
    s->exact_matches = 0;
    s->first_cp = (uint64_t)-1;
    s->first_layer = -1;
    if (s->enabled) {
        fprintf(s->fp, "ds4: P2 exactness probe — session started\n");
    }
}

static void ds4_probe_session_record(ds4_p2_probe_session *s, int cp, uint64_t layer,
                                      ds4_probe_diff *diff) {
    if (!s->enabled || !s->fp) return;

    s->total_comparisons++;
    fprintf(s->fp, "cp=%d layer=%lu exact=%d mismatches=%llu "
            "max_abs=%.6g max_rel=%.6g max_ulp=%u\n",
            cp, (unsigned long)layer, diff->bit_exact ? 1 : 0,
            (unsigned long long)diff->mismatch_count,
            (double)diff->max_abs_diff,
            (double)diff->max_rel_diff,
            diff->max_ulp_diff);

    if (!s->first_cp && !diff->bit_exact) {
        s->first_cp = cp;
        s->first_layer = (int)layer;
        fprintf(s->fp, "ds4: *** FIRST DIVERGENCE DETECTED at cp=%d layer=%lu ***\n",
                (int)cp, (unsigned long)layer);
    }
    if (diff->bit_exact) {
        s->exact_matches++;
    }
}

/* ---- Helper: read GPU tensor to CPU buffer in-place ---- */

static bool ds4_probe_read_f32_from_gpu(
        const ds4_gpu_tensor *tensor,
        uint64_t offset_bytes,
        float *dst,
        uint64_t n_floats) {
    return ds4_gpu_tensor_read(tensor, offset_bytes, dst, n_floats * sizeof(float)) != 0;
}

/* ---- Checkpoint functions (called from ds4.c at strategic points) ---- */

/* CP1: Q/KV projection — compare row 0 and row 1 individually */
static void ds4_probe_cp1(ds4_p2_probe_session *s, uint64_t layer,
                          const float *batched_row0, const float *canonical_row0,
                          const float *batched_row1, const float *canonical_row1,
                          uint64_t dim) {
    if (!s->enabled) return;

    ds4_probe_diff d0, d1;
    ds4_probe_compare_f32_arrays(batched_row0, canonical_row0, dim, &d0);
    ds4_probe_compare_f32_arrays(batched_row1, canonical_row1, dim, &d1);

    /* Print row 0 */
    fprintf(s->fp, "cp=1 layer=%lu row=0 exact=%d mismatches=%llu "
            "max_abs=%.6g max_rel=%.6g max_ulp=%u\n",
            (unsigned long)layer, d0.bit_exact ? 1 : 0,
            (unsigned long long)d0.mismatch_count,
            (double)d0.max_abs_diff,
            (double)d0.max_rel_diff,
            d0.max_ulp_diff);

    /* Print row 1 */
    fprintf(s->fp, "cp=1 layer=%lu row=1 exact=%d mismatches=%llu "
            "max_abs=%.6g max_rel=%.6g max_ulp=%u\n",
            (unsigned long)layer, d1.bit_exact ? 1 : 0,
            (unsigned long long)d1.mismatch_count,
            (double)d1.max_abs_diff,
            (double)d1.max_rel_diff,
            d1.max_ulp_diff);

    s->total_comparisons += 2;
    if (!s->first_cp && !d0.bit_exact) {
        s->first_cp = 1;
        s->first_layer = (int)layer;
        fprintf(s->fp, "ds4: *** FIRST DIVERGENCE DETECTED cp=1 layer=%lu (row0) ***\n",
                (unsigned long)layer);
    }
    if (!s->first_cp && !d1.bit_exact) {
        s->first_cp = 1;
        s->first_layer = (int)layer;
        fprintf(s->fp, "ds4: *** FIRST DIVERGENCE DETECTED cp=1 layer=%lu (row1) ***\n",
                (unsigned long)layer);
    }
    if (d0.bit_exact && d1.bit_exact) {
        s->exact_matches += 2;
    }
}

/* CP3: Compressor projected KV+score — combined for one tensor pair */
static void ds4_probe_cp3(ds4_p2_probe_session *s, uint64_t layer,
                          const char *label,
                          const float *batched, const float *canonical,
                          uint64_t n_floats) {
    if (!s->enabled || !n_floats) return;

    ds4_probe_diff d;
    ds4_probe_compare_f32_arrays(batched, canonical, n_floats, &d);

    fprintf(s->fp, "cp=3 label=%s layer=%lu exact=%d mismatches=%llu "
            "max_abs=%.6g max_rel=%.6g max_ulp=%u\n",
            label, (unsigned long)layer, d.bit_exact ? 1 : 0,
            (unsigned long long)d.mismatch_count,
            (double)d.max_abs_diff,
            (double)d.max_rel_diff,
            d.max_ulp_diff);

    s->total_comparisons++;
    if (!s->first_cp && !d.bit_exact) {
        s->first_cp = 3;
        s->first_layer = (int)layer;
        fprintf(s->fp, "ds4: *** FIRST DIVERGENCE DETECTED cp=3 label=%s layer=%lu ***\n",
                label, (unsigned long)layer);
    }
    if (d.bit_exact) {
        s->exact_matches++;
    }
}

/* CP5: Layer output hidden state */
static void ds4_probe_cp5(ds4_p2_probe_session *s, uint64_t layer,
                          const float *batched_hc, const float *canonical_hc,
                          uint64_t n_floats) {
    if (!s->enabled || !n_floats) return;

    ds4_probe_diff d;
    ds4_probe_compare_f32_arrays(batched_hc, canonical_hc, n_floats, &d);

    fprintf(s->fp, "cp=5 layer=%lu exact=%d mismatches=%llu "
            "max_abs=%.6g max_rel=%.6g max_ulp=%u\n",
            (unsigned long)layer, d.bit_exact ? 1 : 0,
            (unsigned long long)d.mismatch_count,
            (double)d.max_abs_diff,
            (double)d.max_rel_diff,
            d.max_ulp_diff);

    s->total_comparisons++;
    if (!s->first_cp && !d.bit_exact) {
        s->first_cp = 5;
        s->first_layer = (int)layer;
        fprintf(s->fp, "ds4: *** FIRST DIVERGENCE DETECTED cp=5 layer=%lu ***\n",
                (unsigned long)layer);
    }
    if (d.bit_exact) {
        s->exact_matches++;
    }
}

/* ---- Summary output ---- */

static void ds4_probe_print_summary(ds4_p2_probe_session *s) {
    if (!s->enabled || !s->fp) return;

    fprintf(s->fp, "\n=== DSpark Exactness P2 Probe Summary ===\n");
    fprintf(s->fp, "Total comparisons: %llu\n", (unsigned long long)s->total_comparisons);
    fprintf(s->fp, "Exact matches:     %llu\n", (unsigned long long)s->exact_matches);
    fprintf(s->fp, "Divergent:         %llu\n",
            (unsigned long long)(s->total_comparisons - s->exact_matches));

    if (s->first_cp != (uint64_t)-1) {
        fprintf(s->fp, "First divergence at CP%d layer=%d\n",
                (int)s->first_cp, s->first_layer);
    } else {
        fprintf(s->fp, "All comparisons exact — no divergence detected.\n");
    }

    fprintf(s->fp, "=== End P2 Summary ===\n\n");
}

/* ---- Static scratch buffers for GPU read operations (avoid per-call malloc) ---- */

#define DS4_PROBE_SCRATCH_FLOATS 65536u /* 256KB — enough for Q/KV projection reads */
static float ds4_probe_scratch[DS4_PROBE_SCRATCH_FLOATS];
