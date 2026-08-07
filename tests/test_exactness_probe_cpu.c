/*
 * CPU-only unit tests for ds4_exactness_probe helper functions.
 *
 * Tests only the array comparison path (ds4_probe_compare_f32_arrays)
 * which has no GPU runtime dependency.  The tensor comparison path
 * (ds4_probe_compare_f32_tensors) is tested at runtime via P2 probe.
 *
 * Build:  cc -O2 -Wall -Wextra -std=c99 -I. \
 *          -o tests/test_exactness_probe_cpu \
 *          tests/test_exactness_probe_cpu.c ds4_exactness_probe.o -lm
 * Run:    ./tests/test_exactness_probe_cpu
 */

#include "ds4_exactness_probe.h"

/*
 * Stub for ds4_gpu_tensor_read — only needed because probe.c contains
 * ds4_probe_compare_f32_tensors which references this symbol.  The CPU
 * test never invokes the tensor path; stub returns -1 (failure) so any
 * accidental call is detected but does not crash the linker.
 */
#include "ds4_gpu.h"

__attribute__((weak))
int ds4_gpu_tensor_read(const ds4_gpu_tensor *tensor, uint64_t offset, void *data, uint64_t bytes) {
    (void)tensor; (void)offset; (void)data; (void)bytes;
    return -1; /* CPU test only — tensor path not exercised */
}
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <stdint.h>

#define EPSILON_F32 1e-6f

/* Track test results */
static int tests_run = 0;
static int tests_failed = 0;

static void check_true(int cond, const char *desc) {
    tests_run++;
    if (!cond) {
        fprintf(stderr, "FAIL [%s]: %s\n", __func__, desc);
        tests_failed++;
    }
}

static void check_approx(float a, float b, const char *desc) {
    tests_run++;
    float diff = fabsf(a - b);
    int ok = (diff < EPSILON_F32) ||
             (fabsf(a) < EPSILON_F32 && fabsf(b) < EPSILON_F32);
    if (!ok) {
        fprintf(stderr, "FAIL [%s]: %s: got %g, expected ~%g\n",
                __func__, desc, (double)a, (double)b);
        tests_failed++;
    }
}

static void check_u32(uint32_t a, uint32_t b, const char *desc) {
    tests_run++;
    if (a != b) {
        fprintf(stderr, "FAIL [%s]: %s: got 0x%08x, expected 0x%08x\n",
                __func__, desc, a, b);
        tests_failed++;
    }
}

static void check_u64(uint64_t a, uint64_t b, const char *desc) {
    tests_run++;
    if (a != b) {
        fprintf(stderr, "FAIL [%s]: %s: got 0x%016lx, expected 0x%016lx\n",
                __func__, desc, (unsigned long)a, (unsigned long)b);
        tests_failed++;
    }
}

/* ---- Test cases ---- */

static void test_null_safety(void) {
    ds4_probe_diff diff;
    memset(&diff, 0, sizeof(diff));
    float buf[4] = {1.0f, 2.0f, 3.0f, 4.0f};

    check_true(!ds4_probe_compare_f32_arrays(NULL, buf, 4, &diff), "NULL actual rejected");
    check_true(!ds4_probe_compare_f32_arrays(buf, NULL, 4, &diff), "NULL expected rejected");
    check_true(!ds4_probe_compare_f32_arrays(buf, buf, 4, NULL), "NULL result rejected");
}

static void test_bit_exact_match(void) {
    ds4_probe_diff diff;
    float a[8] = {0.0f, 1.0f, -1.0f, 0.5f, INFINITY, -INFINITY, NAN, 3.14159265f};
    float b[8] = {0.0f, 1.0f, -1.0f, 0.5f, INFINITY, -INFINITY, NAN, 3.14159265f};

    check_true(ds4_probe_compare_f32_arrays(a, b, 8, &diff), "compare returns true");
    check_true(diff.bit_exact, "bit_exact should be true for identical arrays");
    check_u64(diff.first_mismatch, (uint64_t)-1, "first_mismatch should be all-ones");
    check_u32(diff.mismatch_count, 0u, "mismatch_count should be zero");
    check_approx(diff.max_abs_diff, 0.0f, "max_abs_diff for exact match");
    check_approx(diff.max_rel_diff, 0.0f, "max_rel_diff for exact match");
    check_u32(diff.max_ulp_diff, 0u, "max_ulp_diff for exact match");
}

static void test_single_mismatch(void) {
    ds4_probe_diff diff;
    float a[8];
    float b[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

    for (int i = 0; i < 8; i++) a[i] = b[i];
    a[3] = 4.1f; /* change index 3 */

    check_true(ds4_probe_compare_f32_arrays(a, b, 8, &diff), "compare returns true");
    check_true(!diff.bit_exact, "bit_exact should be false for mismatching arrays");
    check_u64(diff.first_mismatch, 3u, "first_mismatch at index 3");
    check_u32(diff.mismatch_count, 1u, "exactly 1 mismatch");
}

static void test_all_mismatches(void) {
    ds4_probe_diff diff;
    float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[4] = {5.0f, 6.0f, 7.0f, 8.0f};

    check_true(ds4_probe_compare_f32_arrays(a, b, 4, &diff), "compare returns true");
    check_true(!diff.bit_exact, "bit_exact should be false");
    check_u32(diff.mismatch_count, 4u, "all 4 mismatches detected");
}

static void test_signed_zero(void) {
    /* IEEE 754: +0.0f and -0.0f have different bit patterns */
    ds4_probe_diff diff;
    float a[2] = {0.0f, 1.0f};
    float b[2] = {-0.0f, 1.0f};

    check_true(ds4_probe_compare_f32_arrays(a, b, 2, &diff), "compare +0 vs -0");
    check_true(!diff.bit_exact, "+0 and -0 should differ in bit pattern");
    check_u64(diff.first_mismatch, 0u, "first mismatch at index 0");
}

static void test_nan_handling(void) {
    ds4_probe_diff diff;
    float a[4] = {1.0f, NAN, 3.0f, 4.0f};
    float b[4] = {1.0f, 2.0f, 3.0f, 4.0f};

    check_true(ds4_probe_compare_f32_arrays(a, b, 4, &diff), "compare with NaN");
    check_true(!diff.bit_exact, "NaN should cause mismatch");
    check_u64(diff.first_mismatch, 1u, "first mismatch at index 1 (NAN)");
    check_u32(diff.max_ulp_diff, UINT32_MAX, "ULP distance for NaN should be UINT32_MAX");
}

static void test_infinity_handling(void) {
    ds4_probe_diff diff;
    float a[2] = {INFINITY, 1.0f};
    float b[2] = {-INFINITY, 1.0f};

    check_true(ds4_probe_compare_f32_arrays(a, b, 2, &diff), "compare +inf vs -inf");
    check_true(!diff.bit_exact, "+inf and -inf should differ");
    check_u64(diff.first_mismatch, 0u, "first mismatch at index 0");
    check_true(isinf(diff.max_abs_diff), "max_abs_diff for +inf vs -inf should be inf");
}

static void test_zero_length(void) {
    ds4_probe_diff diff;

    /* Empty arrays with valid pointers should be trivially exact.
     * NULL actual/expected are rejected by the API regardless of count. */
    float dummy[2];
    check_true(ds4_probe_compare_f32_arrays(dummy, dummy, 0, &diff), "compare zero-length arrays returns true");
    check_true(diff.bit_exact, "zero-length arrays should be bit_exact");
    check_u32(diff.mismatch_count, 0u, "zero mismatches for zero-length");
}

static void test_ulps_nearby_values(void) {
    ds4_probe_diff diff;
    float a = 1.0f;
    /* Next representable float after 1.0 */
    uint32_t bits_b = 0x3F800001u;
    memcpy(&a, &bits_b, sizeof(a));

    check_true(ds4_probe_compare_f32_arrays(&a, &(float){1.0f}, 1, &diff), "ULP neighbors");
    check_true(!diff.bit_exact, "nearby floats should differ");
    check_u32(diff.max_ulp_diff, 1u, "ULP distance between consecutive floats should be 1");
}

static void test_large_array(void) {
    ds4_probe_diff diff;
    const uint64_t n = 100000u;
    float *a = malloc(n * sizeof(float));
    float *b = malloc(n * sizeof(float));

    check_true(a && b, "large array allocation");
    if (a && b) {
        for (uint64_t i = 0; i < n; i++) {
            a[i] = (float)i;
            b[i] = (float)i;
        }
        /* Modify every 1000th element */
        for (uint64_t i = 500u; i < n; i += 1000u) {
            b[i] += 1.0f;
        }

        check_true(ds4_probe_compare_f32_arrays(a, b, n, &diff), "large array compare");
        check_true(!diff.bit_exact, "should detect mismatches in large array");
        check_u32(diff.mismatch_count, 100u, "every 1000th element modified (i=500..99500 = 100 iterations)");
        check_u64(diff.first_mismatch, 500u, "first mismatch at index 500");

        free(a);
        free(b);
    } else {
        free(a);
        free(b);
    }
}

int main(void) {
    printf("Running ds4_exactness_probe CPU unit tests...\n\n");

    test_null_safety();
    test_bit_exact_match();
    test_single_mismatch();
    test_all_mismatches();
    test_signed_zero();
    test_nan_handling();
    test_infinity_handling();
    test_zero_length();
    test_ulps_nearby_values();
    test_large_array();

    printf("\nResults: %d/%d tests passed (%d failed)\n",
           tests_run - tests_failed, tests_run, tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
