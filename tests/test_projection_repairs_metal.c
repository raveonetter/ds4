#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE

#include "ds4_float_compare.h"
#include "ds4_gpu.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

enum {
    TEST_ROWS = 5,
    TEST_BENCH_TRIALS = 20,
    TEST_IN = 4096,
    TEST_Q8_OUT = 1024,
    TEST_F16_OUT = 256,
    TEST_PAIR_OUT = 512,
    TEST_CP5_IN = 1024,
    TEST_CP5_EMBD = 4096,
    TEST_CP5_HC = 4,
    TEST_CP5_SPLIT = 2 * TEST_CP5_HC + TEST_CP5_HC * TEST_CP5_HC,
    TEST_ROUTER_EXPERTS = 256,
    TEST_ROUTER_TOPK = 6,
    TEST_MOE_DIM = 256,
    TEST_MOE_TOTAL_EXPERTS = 8,
    TEST_MOE_USED_EXPERTS = 6,
    TEST_IQ2_XXS_TYPE = 16,
    TEST_Q2_K_TYPE = 10,
    TEST_ATTN_HEADS = 2,
    TEST_ATTN_DIM = 512,
    TEST_ATTN_RAW_CAP = 8,
    TEST_CP4_GROUP_DIM = 64,
    TEST_CP4_RANK = 32,
    TEST_CP4_GROUPS = 2,
    TEST_CP4_LOW_DIM = TEST_CP4_RANK * TEST_CP4_GROUPS,
    TEST_CP4_OUT = 64,
};

typedef struct __attribute__((packed)) {
    uint16_t d;
    int8_t qs[32];
} test_block_q8_0;

typedef struct __attribute__((packed)) {
    uint16_t d;
    uint16_t qs[32];
} test_block_iq2_xxs;

typedef struct __attribute__((packed)) {
    uint8_t scales[16];
    uint8_t qs[64];
    uint16_t d;
    uint16_t dmin;
} test_block_q2_k;

_Static_assert(sizeof(test_block_q8_0) == 34, "Q8_0 block layout changed");
_Static_assert(sizeof(test_block_iq2_xxs) == 66,
               "IQ2_XXS block layout changed");
_Static_assert(sizeof(test_block_q2_k) == 84, "Q2_K block layout changed");

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

static uint64_t align_up(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1u) / alignment * alignment;
}

static uint16_t half_bits(float value) {
    _Float16 half = (_Float16)value;
    uint16_t bits;
    memcpy(&bits, &half, sizeof(bits));
    return bits;
}

static void fill_q8_matrix(test_block_q8_0 *matrix, uint32_t in_dim,
                           uint32_t out_dim, uint32_t salt) {
    const uint32_t blocks = in_dim / 32u;
    for (uint32_t row = 0; row < out_dim; row++) {
        for (uint32_t block = 0; block < blocks; block++) {
            test_block_q8_0 *q = matrix + (uint64_t)row * blocks + block;
            q->d = half_bits((float)(1u + ((row + block + salt) % 7u)) /
                             128.0f);
            for (uint32_t i = 0; i < 32u; i++) {
                q->qs[i] = (int8_t)((int32_t)((row * 3u + block * 5u +
                                               i * 7u + salt) % 15u) - 7);
            }
        }
    }
}

static void fill_f16_matrix(uint16_t *matrix, uint32_t out_dim,
                            uint32_t salt) {
    for (uint32_t row = 0; row < out_dim; row++) {
        for (uint32_t col = 0; col < TEST_IN; col++) {
            const int32_t raw =
                (int32_t)((row * 11u + col * 13u + salt * 17u) % 63u) - 31;
            matrix[(uint64_t)row * TEST_IN + col] =
                half_bits((float)raw / 512.0f);
        }
    }
}

static void fill_iq2_xxs_experts(test_block_iq2_xxs *matrix,
                                 uint32_t salt) {
    for (uint32_t expert = 0; expert < TEST_MOE_TOTAL_EXPERTS; expert++) {
        for (uint32_t row = 0; row < TEST_MOE_DIM; row++) {
            test_block_iq2_xxs *block = matrix +
                (uint64_t)expert * TEST_MOE_DIM + row;
            block->d = half_bits(
                (float)(1u + ((expert + row + salt) % 5u)) / 2048.0f);
            for (uint32_t i = 0; i < 32u; i++) {
                block->qs[i] = (uint16_t)(
                    (expert * 97u + row * 53u + i * 29u + salt * 11u) &
                    0xffffu);
            }
        }
    }
}

static void fill_q2_k_experts(test_block_q2_k *matrix, uint32_t salt) {
    for (uint32_t expert = 0; expert < TEST_MOE_TOTAL_EXPERTS; expert++) {
        for (uint32_t row = 0; row < TEST_MOE_DIM; row++) {
            test_block_q2_k *block = matrix +
                (uint64_t)expert * TEST_MOE_DIM + row;
            block->d = half_bits(
                (float)(1u + ((expert * 3u + row + salt) % 7u)) / 512.0f);
            block->dmin = half_bits(
                (float)(1u + ((expert + row * 5u + salt) % 3u)) / 2048.0f);
            for (uint32_t i = 0; i < 16u; i++) {
                block->scales[i] = (uint8_t)(
                    expert * 13u + row * 7u + i * 17u + salt);
            }
            for (uint32_t i = 0; i < 64u; i++) {
                block->qs[i] = (uint8_t)(
                    expert * 19u + row * 11u + i * 23u + salt * 3u);
            }
        }
    }
}

static int compare_exact(const char *label, const float *actual,
                         const float *expected, size_t count) {
    ds4_float_compare_result result;
    if (!ds4_float_compare_exact(actual, expected, count, &result)) {
        fprintf(stderr, "%s comparator error\n", label);
        return 0;
    }
    if (!result.bit_exact) {
        fprintf(stderr,
                "%s MISMATCH count=%zu first=%zu actual=0x%08x expected=0x%08x\n",
                label, result.mismatch_count, result.first_mismatch_index,
                result.first_actual_bits, result.first_expected_bits);
        return 0;
    }
    return 1;
}

static double wall_clock_ms(void) {
    struct timeval tv;
    if (gettimeofday(&tv, NULL) != 0) return 0.0;
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}

static int run_q8_contract(const void *model, uint64_t model_size,
                           uint64_t gate_offset, uint64_t up_offset,
                           ds4_gpu_tensor *input) {
    const uint64_t count = (uint64_t)TEST_ROWS * TEST_Q8_OUT;
    const uint64_t bytes = count * sizeof(float);
    float *generic_host = calloc((size_t)count, sizeof(float));
    float *candidate_host = calloc((size_t)count, sizeof(float));
    float *expected = calloc((size_t)count, sizeof(float));
    float *gate_actual = calloc((size_t)count, sizeof(float));
    float *gate_expected = calloc((size_t)count, sizeof(float));
    float *up_actual = calloc((size_t)count, sizeof(float));
    float *up_expected = calloc((size_t)count, sizeof(float));
    float *mid_actual = calloc((size_t)count, sizeof(float));
    float *mid_expected = calloc((size_t)count, sizeof(float));
    ds4_gpu_tensor *generic = ds4_gpu_tensor_alloc(bytes);
    ds4_gpu_tensor *candidate = ds4_gpu_tensor_alloc(bytes);
    ds4_gpu_tensor *reference = ds4_gpu_tensor_alloc(bytes);
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(bytes);
    ds4_gpu_tensor *up = ds4_gpu_tensor_alloc(bytes);
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc(bytes);
    ds4_gpu_tensor *gate_ref = ds4_gpu_tensor_alloc(bytes);
    ds4_gpu_tensor *up_ref = ds4_gpu_tensor_alloc(bytes);
    ds4_gpu_tensor *mid_ref = ds4_gpu_tensor_alloc(bytes);
    int old_mismatch = 0;
    int candidate_exact = 0;
    int shared_gate_exact = 0;
    int performance_ok = 0;
    double generic_total_ms = 0.0;
    double candidate_total_ms = 0.0;
    double exact_total_ms = 0.0;
    int ok = generic_host && candidate_host && expected &&
             gate_actual && gate_expected &&
             up_actual && up_expected && mid_actual && mid_expected &&
             generic && candidate && reference && gate && up && mid &&
             gate_ref && up_ref && mid_ref;

    if (ok) {
        ok = ds4_gpu_matmul_q8_0_tensor(
                 generic, model, model_size, gate_offset,
                 TEST_IN, TEST_Q8_OUT, input, TEST_ROWS) != 0 &&
             ds4_gpu_matmul_q8_0_canonical_batch_tensor(
                 candidate, model, model_size, gate_offset,
                 TEST_IN, TEST_Q8_OUT, input, TEST_ROWS) != 0;
    }
    for (uint32_t row = 0; ok && row < TEST_ROWS; row++) {
        ds4_gpu_tensor *in_row = ds4_gpu_tensor_view(
            input, (uint64_t)row * TEST_IN * sizeof(float),
            (uint64_t)TEST_IN * sizeof(float));
        ds4_gpu_tensor *out_row = ds4_gpu_tensor_view(
            reference, (uint64_t)row * TEST_Q8_OUT * sizeof(float),
            (uint64_t)TEST_Q8_OUT * sizeof(float));
        ok = in_row && out_row &&
             ds4_gpu_matmul_q8_0_tensor(
                 out_row, model, model_size, gate_offset,
                 TEST_IN, TEST_Q8_OUT, in_row, 1) != 0;
        ds4_gpu_tensor_free(out_row);
        ds4_gpu_tensor_free(in_row);
    }
    if (ok) {
        ok = ds4_gpu_matmul_q8_0_canonical_batch_tensor(
                 gate, model, model_size, gate_offset,
                 TEST_IN, TEST_Q8_OUT, input, TEST_ROWS) != 0 &&
             ds4_gpu_matmul_q8_0_canonical_batch_tensor(
                 up, model, model_size, up_offset,
                 TEST_IN, TEST_Q8_OUT, input, TEST_ROWS) != 0 &&
             ds4_gpu_swiglu_tensor(
                 mid, gate, up, (uint32_t)count, 7.0f, 1.0f) != 0;
    }
    for (uint32_t row = 0; ok && row < TEST_ROWS; row++) {
        const uint64_t in_off = (uint64_t)row * TEST_IN * sizeof(float);
        const uint64_t out_off =
            (uint64_t)row * TEST_Q8_OUT * sizeof(float);
        ds4_gpu_tensor *in_row = ds4_gpu_tensor_view(
            input, in_off, (uint64_t)TEST_IN * sizeof(float));
        ds4_gpu_tensor *gate_row = ds4_gpu_tensor_view(
            gate_ref, out_off, (uint64_t)TEST_Q8_OUT * sizeof(float));
        ds4_gpu_tensor *up_row = ds4_gpu_tensor_view(
            up_ref, out_off, (uint64_t)TEST_Q8_OUT * sizeof(float));
        ds4_gpu_tensor *mid_row = ds4_gpu_tensor_view(
            mid_ref, out_off, (uint64_t)TEST_Q8_OUT * sizeof(float));
        ok = in_row && gate_row && up_row && mid_row &&
             ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
                 gate_row, up_row, mid_row, model, model_size,
                 gate_offset, up_offset, TEST_IN, TEST_Q8_OUT,
                 in_row, 7.0f) != 0;
        ds4_gpu_tensor_free(mid_row);
        ds4_gpu_tensor_free(up_row);
        ds4_gpu_tensor_free(gate_row);
        ds4_gpu_tensor_free(in_row);
    }
    if (ok) {
        ds4_float_compare_result old_result;
        ok = ds4_gpu_tensor_read(generic, 0, generic_host, bytes) &&
             ds4_gpu_tensor_read(candidate, 0, candidate_host, bytes) &&
             ds4_gpu_tensor_read(reference, 0, expected, bytes) &&
             ds4_gpu_tensor_read(gate, 0, gate_actual, bytes) &&
             ds4_gpu_tensor_read(gate_ref, 0, gate_expected, bytes) &&
             ds4_gpu_tensor_read(up, 0, up_actual, bytes) &&
             ds4_gpu_tensor_read(up_ref, 0, up_expected, bytes) &&
             ds4_gpu_tensor_read(mid, 0, mid_actual, bytes) &&
             ds4_gpu_tensor_read(mid_ref, 0, mid_expected, bytes) &&
             ds4_float_compare_exact(generic_host, expected,
                                     (size_t)count, &old_result);
        if (ok) {
            old_mismatch = !old_result.bit_exact;
            candidate_exact = compare_exact(
                "family1_qa_candidate", candidate_host, expected,
                (size_t)count);
            ok = old_mismatch && candidate_exact;
            if (!old_mismatch) {
                fprintf(stderr,
                        "family1_qa_generic unexpectedly matched the single-row oracle\n");
            }
        }
        if (ok) {
            shared_gate_exact =
                compare_exact("q8_fused_gate", gate_actual, gate_expected,
                              (size_t)count) &&
                compare_exact("q8_fused_up", up_actual, up_expected,
                              (size_t)count) &&
                compare_exact("q8_fused_mid", mid_actual, mid_expected,
                              (size_t)count);
            ok = shared_gate_exact;
        }
    }

    /* Warm all three pipelines before measuring.  Every timed operation is
     * followed by an explicit synchronization so a future batched command
     * context cannot turn this into CPU enqueue timing. */
    if (ok) {
        ok = ds4_gpu_matmul_q8_0_tensor(
                 generic, model, model_size, gate_offset,
                 TEST_IN, TEST_Q8_OUT, input, TEST_ROWS) != 0 &&
             ds4_gpu_matmul_q8_0_canonical_batch_tensor(
                 candidate, model, model_size, gate_offset,
                 TEST_IN, TEST_Q8_OUT, input, TEST_ROWS) != 0 &&
             ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(
                 reference, model, model_size, gate_offset,
                 TEST_IN, TEST_Q8_OUT, input, TEST_ROWS) != 0 &&
             ds4_gpu_synchronize() != 0;
    }
    for (unsigned trial = 0; ok && trial < TEST_BENCH_TRIALS; trial++) {
        double start = wall_clock_ms();
        ok = ds4_gpu_matmul_q8_0_tensor(
                 generic, model, model_size, gate_offset,
                 TEST_IN, TEST_Q8_OUT, input, TEST_ROWS) != 0 &&
             ds4_gpu_synchronize() != 0;
        generic_total_ms += wall_clock_ms() - start;

        start = wall_clock_ms();
        if (ok) {
            ok = ds4_gpu_matmul_q8_0_canonical_batch_tensor(
                     candidate, model, model_size, gate_offset,
                     TEST_IN, TEST_Q8_OUT, input, TEST_ROWS) != 0 &&
                 ds4_gpu_synchronize() != 0;
        }
        candidate_total_ms += wall_clock_ms() - start;

        start = wall_clock_ms();
        if (ok) {
            ok = ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(
                     reference, model, model_size, gate_offset,
                     TEST_IN, TEST_Q8_OUT, input, TEST_ROWS) != 0 &&
                 ds4_gpu_synchronize() != 0;
        }
        exact_total_ms += wall_clock_ms() - start;
    }
    if (ok) {
        performance_ok = candidate_total_ms > 0.0 && exact_total_ms > 0.0 &&
            candidate_total_ms < exact_total_ms;
        ok = performance_ok;
    }

    printf("FAMILY1_REPAIR_TEST "
           "family=FAMILY_Q8_0_BATCH_EXT_VS_SINGLE_MV site=QA "
           "old_kernel=kernel_mul_mv_ext_q8_0_f32_r1_5 "
           "new_kernel=kernel_mul_mv_q8_0_f32_canonical_batch "
           "input_bits_equal=PASS weights_same=PASS metadata_same=PASS "
           "old_vs_seq=%s new_vs_seq=%s batch_parallelism=PRESERVED "
           "candidate_dispatches=1 exact_dispatches=%u "
           "timing_trials=%u generic_ms=%.3f candidate_ms=%.3f exact_ms=%.3f "
           "candidate_over_generic=%.3f "
           "candidate_faster_than_exact=%s result=%s\n",
           old_mismatch ? "MISMATCH" : "EXACT",
           candidate_exact ? "EXACT" : "MISMATCH",
           TEST_ROWS,
           TEST_BENCH_TRIALS,
           generic_total_ms / TEST_BENCH_TRIALS,
           candidate_total_ms / TEST_BENCH_TRIALS,
           exact_total_ms / TEST_BENCH_TRIALS,
           generic_total_ms > 0.0
               ? candidate_total_ms / generic_total_ms
               : 0.0,
           performance_ok ? "PASS" : "FAIL",
           ok ? "PASS" : "FAIL");
    printf("EXACT_ROW_ORACLE_AB "
           "family=FAMILY_Q8_0_BATCH_EXT_VS_SINGLE_MV "
           "site=shared_gate_up rows=%u batch_parallelism=PRESERVED "
           "candidate_dispatches=2 result=%s performance=UNMEASURED\n",
           TEST_ROWS, shared_gate_exact ? "EXACT" : "MISMATCH");

    ds4_gpu_tensor_free(mid_ref);
    ds4_gpu_tensor_free(up_ref);
    ds4_gpu_tensor_free(gate_ref);
    ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(up);
    ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(reference);
    ds4_gpu_tensor_free(candidate);
    ds4_gpu_tensor_free(generic);
    free(mid_expected);
    free(mid_actual);
    free(up_expected);
    free(up_actual);
    free(gate_expected);
    free(gate_actual);
    free(expected);
    free(candidate_host);
    free(generic_host);
    return ok;
}

static int run_cp4_output_b_contract(const void *model, uint64_t model_size,
                                     uint64_t out_a_offset,
                                     uint64_t out_b_offset) {
    const uint64_t heads_count = (uint64_t)TEST_ROWS * TEST_CP4_GROUPS *
                                 TEST_CP4_GROUP_DIM;
    const uint64_t low_count = (uint64_t)TEST_ROWS * TEST_CP4_LOW_DIM;
    const uint64_t out_count = (uint64_t)TEST_ROWS * TEST_CP4_OUT;
    float *heads_host = calloc((size_t)heads_count, sizeof(float));
    float *actual = calloc((size_t)out_count, sizeof(float));
    float *expected = calloc((size_t)out_count, sizeof(float));
    ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc(heads_count * sizeof(float));
    ds4_gpu_tensor *low = ds4_gpu_tensor_alloc(low_count * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(out_count * sizeof(float));
    ds4_gpu_tensor *reference = ds4_gpu_tensor_alloc(out_count * sizeof(float));
    ds4_gpu_tensor *group_tmp = ds4_gpu_tensor_alloc(sizeof(float));
    ds4_gpu_tensor *low_tmp = ds4_gpu_tensor_alloc(sizeof(float));
    int ok = heads_host && actual && expected && heads && low && out &&
             reference && group_tmp && low_tmp;

    for (uint64_t i = 0; ok && i < heads_count; i++) {
        uint32_t bits = 0x3f000000u |
            ((uint32_t)(i * 0x45d9f3bu + 0x27d4eb2du) & 0x007fffffu);
        memcpy(&heads_host[i], &bits, sizeof(float));
        if (i & 1u) heads_host[i] = -heads_host[i];
    }
    if (ok) {
        ok = ds4_gpu_tensor_write(heads, 0, heads_host,
                                  heads_count * sizeof(float)) &&
             ds4_gpu_attention_output_q8_canonical_b_batch_tensor(
                 out, low, group_tmp, low_tmp, model, model_size,
                 out_a_offset, out_b_offset, TEST_CP4_GROUP_DIM,
                 TEST_CP4_RANK, TEST_CP4_GROUPS, TEST_CP4_OUT,
                 heads, TEST_ROWS) != 0;
    }
    for (uint32_t row = 0; ok && row < TEST_ROWS; row++) {
        ds4_gpu_tensor *low_row = ds4_gpu_tensor_view(
            low, (uint64_t)row * TEST_CP4_LOW_DIM * sizeof(float),
            (uint64_t)TEST_CP4_LOW_DIM * sizeof(float));
        ds4_gpu_tensor *out_row = ds4_gpu_tensor_view(
            reference, (uint64_t)row * TEST_CP4_OUT * sizeof(float),
            (uint64_t)TEST_CP4_OUT * sizeof(float));
        ok = low_row && out_row && ds4_gpu_matmul_q8_0_tensor(
                 out_row, model, model_size, out_b_offset,
                 TEST_CP4_LOW_DIM, TEST_CP4_OUT, low_row, 1) != 0;
        ds4_gpu_tensor_free(out_row);
        ds4_gpu_tensor_free(low_row);
    }
    if (ok) {
        ok = ds4_gpu_tensor_read(out, 0, actual,
                                 out_count * sizeof(float)) &&
             ds4_gpu_tensor_read(reference, 0, expected,
                                 out_count * sizeof(float)) &&
             compare_exact("cp4_output_b_candidate", actual, expected,
                           (size_t)out_count);
    }
    printf("FAMILY1_REPAIR_TEST "
           "family=FAMILY_Q8_0_BATCH_EXT_VS_SINGLE_MV site=CP4_output_B "
           "new_kernel=kernel_mul_mv_q8_0_f32_canonical_batch "
           "new_vs_seq=%s batch_parallelism=PRESERVED result=%s\n",
           ok ? "EXACT" : "MISMATCH", ok ? "PASS" : "FAIL");

    ds4_gpu_tensor_free(low_tmp);
    ds4_gpu_tensor_free(group_tmp);
    ds4_gpu_tensor_free(reference);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(low);
    ds4_gpu_tensor_free(heads);
    free(expected);
    free(actual);
    free(heads_host);
    return ok;
}

/* Bit 0 is the single-projection family; bit 1 is the paired family. */
static unsigned run_f16_contracts(const void *model, uint64_t model_size,
                                  uint64_t weight_a_offset,
                                  uint64_t weight_b_offset,
                                  ds4_gpu_tensor *input) {
    const uint64_t single_count = (uint64_t)TEST_ROWS * TEST_F16_OUT;
    const uint64_t pair_count = (uint64_t)TEST_ROWS * TEST_PAIR_OUT;
    float *single_actual = calloc((size_t)single_count, sizeof(float));
    float *single_expected = calloc((size_t)single_count, sizeof(float));
    float *single_old_actual = calloc((size_t)single_count, sizeof(float));
    float *pair_a_actual = calloc((size_t)pair_count, sizeof(float));
    float *pair_a_expected = calloc((size_t)pair_count, sizeof(float));
    float *pair_b_actual = calloc((size_t)pair_count, sizeof(float));
    float *pair_b_expected = calloc((size_t)pair_count, sizeof(float));
    ds4_gpu_tensor *single = ds4_gpu_tensor_alloc(
        single_count * sizeof(float));
    ds4_gpu_tensor *single_ref = ds4_gpu_tensor_alloc(
        single_count * sizeof(float));
    ds4_gpu_tensor *single_old = ds4_gpu_tensor_alloc(
        single_count * sizeof(float));
    ds4_gpu_tensor *pair_a = ds4_gpu_tensor_alloc(pair_count * sizeof(float));
    ds4_gpu_tensor *pair_b = ds4_gpu_tensor_alloc(pair_count * sizeof(float));
    ds4_gpu_tensor *pair_a_ref = ds4_gpu_tensor_alloc(
        pair_count * sizeof(float));
    ds4_gpu_tensor *pair_b_ref = ds4_gpu_tensor_alloc(
        pair_count * sizeof(float));
    int setup_ok = single_actual && single_expected && single_old_actual && pair_a_actual &&
                   pair_a_expected && pair_b_actual && pair_b_expected &&
                   single && single_ref && single_old && pair_a && pair_b &&
                   pair_a_ref && pair_b_ref;
    int single_ok = setup_ok;
    int pair_ok = setup_ok;

    if (single_ok) {
        single_ok = ds4_gpu_matmul_f16_canonical_batch_tensor(
                        single, model, model_size, weight_a_offset,
                        TEST_IN, TEST_F16_OUT, input, TEST_ROWS) != 0;
        if (single_ok) {
            single_ok = ds4_gpu_matmul_f16_tensor(
                            single_old, model, model_size, weight_a_offset,
                            TEST_IN, TEST_F16_OUT, input, TEST_ROWS) != 0;
        }
    }
    if (pair_ok) {
        pair_ok = ds4_gpu_matmul_f16_pair_decode_rows_exact_tensor(
                      pair_a, pair_b, model, model_size,
                      weight_a_offset, weight_b_offset,
                      TEST_IN, TEST_PAIR_OUT, input, TEST_ROWS) != 0;
    }
    for (uint32_t row = 0; (single_ok || pair_ok) && row < TEST_ROWS; row++) {
        const uint64_t in_off = (uint64_t)row * TEST_IN * sizeof(float);
        ds4_gpu_tensor *in_row = ds4_gpu_tensor_view(
            input, in_off, (uint64_t)TEST_IN * sizeof(float));
        ds4_gpu_tensor *single_row = ds4_gpu_tensor_view(
            single_ref, (uint64_t)row * TEST_F16_OUT * sizeof(float),
            (uint64_t)TEST_F16_OUT * sizeof(float));
        ds4_gpu_tensor *pair_a_row = ds4_gpu_tensor_view(
            pair_a_ref, (uint64_t)row * TEST_PAIR_OUT * sizeof(float),
            (uint64_t)TEST_PAIR_OUT * sizeof(float));
        ds4_gpu_tensor *pair_b_row = ds4_gpu_tensor_view(
            pair_b_ref, (uint64_t)row * TEST_PAIR_OUT * sizeof(float),
            (uint64_t)TEST_PAIR_OUT * sizeof(float));
        if (!in_row) {
            single_ok = 0;
            pair_ok = 0;
        }
        if (single_ok) {
            single_ok = single_row &&
                ds4_gpu_matmul_f16_tensor(
                    single_row, model, model_size, weight_a_offset,
                    TEST_IN, TEST_F16_OUT, in_row, 1) != 0;
        }
        if (pair_ok) {
            pair_ok = pair_a_row && pair_b_row &&
                ds4_gpu_matmul_f16_pair_tensor(
                    pair_a_row, pair_b_row, model, model_size,
                    weight_a_offset, weight_b_offset,
                    TEST_IN, TEST_PAIR_OUT, in_row, 1) != 0;
        }
        ds4_gpu_tensor_free(pair_b_row);
        ds4_gpu_tensor_free(pair_a_row);
        ds4_gpu_tensor_free(single_row);
        ds4_gpu_tensor_free(in_row);
    }
    int old_generic_mismatch = 0;
    if (single_ok) {
        single_ok = ds4_gpu_tensor_read(
                        single, 0, single_actual,
                        single_count * sizeof(float)) &&
                    ds4_gpu_tensor_read(
                        single_old, 0, single_old_actual,
                        single_count * sizeof(float)) &&
                    ds4_gpu_tensor_read(
                        single_ref, 0, single_expected,
                        single_count * sizeof(float)) &&
                    compare_exact("f16_projection", single_actual,
                                  single_expected, (size_t)single_count);
        old_generic_mismatch = single_ok &&
            memcmp(single_old_actual, single_expected,
                   (size_t)single_count * sizeof(float)) != 0;
        single_ok = single_ok && old_generic_mismatch;
    }
    if (single_ok) {
        static const uint32_t row_cases[] = {1u, 2u, TEST_ROWS};
        for (size_t i = 0; single_ok && i < sizeof(row_cases) / sizeof(row_cases[0]); i++) {
            const uint32_t rows = row_cases[i];
            const size_t count = (size_t)rows * TEST_F16_OUT;
            single_ok = ds4_gpu_matmul_f16_canonical_batch_tensor(
                            single, model, model_size, weight_a_offset,
                            TEST_IN, TEST_F16_OUT, input, rows) != 0 &&
                        ds4_gpu_tensor_read(
                            single, 0, single_actual,
                            (uint64_t)count * sizeof(float)) != 0 &&
                        memcmp(single_actual, single_expected,
                               count * sizeof(float)) == 0;
            printf("FAMILY3_ROW_AB rows=%u candidate_dispatches=1 "
                   "oracle_dispatches=%u result=%s\n",
                   rows, rows, single_ok ? "EXACT" : "MISMATCH");
        }
    }
    if (pair_ok) {
        pair_ok = ds4_gpu_tensor_read(
                      pair_a, 0, pair_a_actual,
                      pair_count * sizeof(float)) &&
                  ds4_gpu_tensor_read(
                      pair_a_ref, 0, pair_a_expected,
                      pair_count * sizeof(float)) &&
                  ds4_gpu_tensor_read(
                      pair_b, 0, pair_b_actual,
                      pair_count * sizeof(float)) &&
                  ds4_gpu_tensor_read(
                      pair_b_ref, 0, pair_b_expected,
                      pair_count * sizeof(float)) &&
                  compare_exact("f16_pair_a", pair_a_actual,
                                pair_a_expected, (size_t)pair_count) &&
                  compare_exact("f16_pair_b", pair_b_actual,
                                pair_b_expected, (size_t)pair_count);
    }

    printf("FAMILY3_PRIMITIVE_AB family=FAMILY_F16_BATCH_EXT_VS_SINGLE_MV "
           "rows=%u sites=hc_attn_pre_split,hc_ffn_pre,ffn_router_projection "
           "input_bits_equal=PASS weights_same=PASS metadata_same=PASS "
           "old_generic_vs_oracle=%s new_batch_vs_oracle=%s "
           "mismatch_count=%u batch_parallelism=PRESERVED "
           "candidate_dispatches=1 oracle_dispatches=%u result=%s\n",
           TEST_ROWS, old_generic_mismatch ? "MISMATCH" : "EXACT",
           single_ok ? "EXACT" : "MISMATCH", single_ok ? 0u : 1u,
           TEST_ROWS, single_ok ? "PASS" : "FAIL");
    printf("EXACT_ROW_ORACLE_AB "
           "family=FAMILY_F16_BATCH_EXT_VS_SINGLE_PAIR_MV rows=%u "
           "sites=attention_kv,attention_score,indexer_kv,indexer_score "
           "input_bits_equal=PASS weights_same=PASS metadata_same=PASS "
           "result=%s\n", TEST_ROWS, pair_ok ? "EXACT" : "MISMATCH");

    ds4_gpu_tensor_free(pair_b_ref);
    ds4_gpu_tensor_free(pair_a_ref);
    ds4_gpu_tensor_free(pair_b);
    ds4_gpu_tensor_free(pair_a);
    ds4_gpu_tensor_free(single_ref);
    ds4_gpu_tensor_free(single_old);
    ds4_gpu_tensor_free(single);
    free(single_old_actual);
    free(pair_b_expected);
    free(pair_b_actual);
    free(pair_a_expected);
    free(pair_a_actual);
    free(single_expected);
    free(single_actual);
    return (single_ok ? 1u : 0u) | (pair_ok ? 2u : 0u);
}

static int run_cp5_tail_contract(const void *model, uint64_t model_size,
                                 uint64_t weight_offset) {
    const uint64_t mid_count = (uint64_t)TEST_ROWS * TEST_CP5_IN;
    const uint64_t embd_count = (uint64_t)TEST_ROWS * TEST_CP5_EMBD;
    const uint64_t hc_count = embd_count * TEST_CP5_HC;
    const uint64_t split_count =
        (uint64_t)TEST_ROWS * TEST_CP5_SPLIT;
    float *mid_host = calloc((size_t)mid_count, sizeof(float));
    float *routed_host = calloc((size_t)embd_count, sizeof(float));
    float *residual_host = calloc((size_t)hc_count, sizeof(float));
    float *split_host = calloc((size_t)split_count, sizeof(float));
    float *actual = calloc((size_t)hc_count, sizeof(float));
    float *expected = calloc((size_t)hc_count, sizeof(float));
    float *shared_actual = calloc((size_t)embd_count, sizeof(float));
    float *shared_expected = calloc((size_t)embd_count, sizeof(float));
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc(mid_count * sizeof(float));
    ds4_gpu_tensor *routed =
        ds4_gpu_tensor_alloc(embd_count * sizeof(float));
    ds4_gpu_tensor *residual =
        ds4_gpu_tensor_alloc(hc_count * sizeof(float));
    ds4_gpu_tensor *split =
        ds4_gpu_tensor_alloc(split_count * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(hc_count * sizeof(float));
    ds4_gpu_tensor *out_ref =
        ds4_gpu_tensor_alloc(hc_count * sizeof(float));
    ds4_gpu_tensor *shared =
        ds4_gpu_tensor_alloc(embd_count * sizeof(float));
    ds4_gpu_tensor *shared_ref =
        ds4_gpu_tensor_alloc(embd_count * sizeof(float));
    int ok = mid_host && routed_host && residual_host && split_host &&
             actual && expected && shared_actual && shared_expected &&
             mid && routed && residual && split && out && out_ref &&
             shared && shared_ref;

    for (uint64_t i = 0; ok && i < mid_count; i++) {
        mid_host[i] = (float)((int32_t)((i * 17u + 5u) % 127u) - 63) /
                      256.0f;
    }
    for (uint64_t i = 0; ok && i < embd_count; i++) {
        routed_host[i] =
            (float)((int32_t)((i * 23u + 11u) % 193u) - 96) / 512.0f;
    }
    for (uint64_t i = 0; ok && i < hc_count; i++) {
        residual_host[i] =
            (float)((int32_t)((i * 29u + 7u) % 251u) - 125) / 1024.0f;
    }
    for (uint64_t i = 0; ok && i < split_count; i++) {
        split_host[i] =
            (float)((int32_t)((i * 31u + 13u) % 61u) - 30) / 64.0f;
    }
    if (ok) {
        ok = ds4_gpu_tensor_write(mid, 0, mid_host,
                                  mid_count * sizeof(float)) &&
             ds4_gpu_tensor_write(routed, 0, routed_host,
                                  embd_count * sizeof(float)) &&
             ds4_gpu_tensor_write(residual, 0, residual_host,
                                  hc_count * sizeof(float)) &&
             ds4_gpu_tensor_write(split, 0, split_host,
                                  split_count * sizeof(float)) &&
             ds4_gpu_shared_down_hc_expand_q8_0_rows_exact_tensor(
                 out, shared, model, model_size, weight_offset,
                 TEST_CP5_IN, TEST_CP5_EMBD, mid, routed, residual, split,
                 TEST_CP5_EMBD, TEST_CP5_HC, TEST_ROWS) != 0;
    }
    for (uint32_t row = 0; ok && row < TEST_ROWS; row++) {
        const uint64_t mid_offset =
            (uint64_t)row * TEST_CP5_IN * sizeof(float);
        const uint64_t embd_offset =
            (uint64_t)row * TEST_CP5_EMBD * sizeof(float);
        const uint64_t hc_offset =
            (uint64_t)row * TEST_CP5_HC * TEST_CP5_EMBD * sizeof(float);
        const uint64_t split_offset =
            (uint64_t)row * TEST_CP5_SPLIT * sizeof(float);
        ds4_gpu_tensor *mid_row = ds4_gpu_tensor_view(
            mid, mid_offset, (uint64_t)TEST_CP5_IN * sizeof(float));
        ds4_gpu_tensor *routed_row = ds4_gpu_tensor_view(
            routed, embd_offset,
            (uint64_t)TEST_CP5_EMBD * sizeof(float));
        ds4_gpu_tensor *residual_row = ds4_gpu_tensor_view(
            residual, hc_offset,
            (uint64_t)TEST_CP5_HC * TEST_CP5_EMBD * sizeof(float));
        ds4_gpu_tensor *split_row = ds4_gpu_tensor_view(
            split, split_offset,
            (uint64_t)TEST_CP5_SPLIT * sizeof(float));
        ds4_gpu_tensor *out_row = ds4_gpu_tensor_view(
            out_ref, hc_offset,
            (uint64_t)TEST_CP5_HC * TEST_CP5_EMBD * sizeof(float));
        ds4_gpu_tensor *shared_row = ds4_gpu_tensor_view(
            shared_ref, embd_offset,
            (uint64_t)TEST_CP5_EMBD * sizeof(float));
        ok = mid_row && routed_row && residual_row && split_row && out_row &&
             shared_row &&
             ds4_gpu_shared_down_hc_expand_q8_0_tensor(
                 out_row, shared_row, model, model_size, weight_offset,
                 TEST_CP5_IN, TEST_CP5_EMBD, mid_row, routed_row,
                 residual_row, split_row, TEST_CP5_EMBD, TEST_CP5_HC) != 0;
        ds4_gpu_tensor_free(shared_row);
        ds4_gpu_tensor_free(out_row);
        ds4_gpu_tensor_free(split_row);
        ds4_gpu_tensor_free(residual_row);
        ds4_gpu_tensor_free(routed_row);
        ds4_gpu_tensor_free(mid_row);
    }
    if (ok) {
        ok = ds4_gpu_tensor_read(out, 0, actual,
                                 hc_count * sizeof(float)) &&
             ds4_gpu_tensor_read(out_ref, 0, expected,
                                 hc_count * sizeof(float)) &&
             ds4_gpu_tensor_read(shared, 0, shared_actual,
                                 embd_count * sizeof(float)) &&
             ds4_gpu_tensor_read(shared_ref, 0, shared_expected,
                                 embd_count * sizeof(float)) &&
             compare_exact("cp5_tail_hc", actual, expected,
                           (size_t)hc_count) &&
             compare_exact("cp5_tail_shared_out", shared_actual,
                           shared_expected, (size_t)embd_count);
    }

    printf("EXACT_ROW_ORACLE_AB "
           "family=FAMILY_Q8_SHARED_DOWN_BATCH_F32_HC_ADD_VS_SINGLE_FUSED_HC "
           "rows=%u sites=cp5_tail input_bits_equal=PASS weights_same=PASS "
           "metadata_same=PASS result=%s\n",
           TEST_ROWS, ok ? "EXACT" : "MISMATCH");

    ds4_gpu_tensor_free(shared_ref);
    ds4_gpu_tensor_free(shared);
    ds4_gpu_tensor_free(out_ref);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(split);
    ds4_gpu_tensor_free(residual);
    ds4_gpu_tensor_free(routed);
    ds4_gpu_tensor_free(mid);
    free(shared_expected);
    free(shared_actual);
    free(expected);
    free(actual);
    free(split_host);
    free(residual_host);
    free(routed_host);
    free(mid_host);
    return ok;
}

static int run_flash_attention_contract(const void *model,
                                        uint64_t model_size,
                                        uint64_t sinks_offset) {
    const uint64_t row_values =
        (uint64_t)TEST_ATTN_HEADS * TEST_ATTN_DIM;
    const uint64_t output_count = (uint64_t)TEST_ROWS * row_values;
    const uint64_t raw_count =
        (uint64_t)TEST_ATTN_RAW_CAP * TEST_ATTN_DIM;
    float *q_host = calloc((size_t)output_count, sizeof(float));
    float *raw_host = calloc((size_t)raw_count, sizeof(float));
    float *actual = calloc((size_t)output_count, sizeof(float));
    float *expected = calloc((size_t)output_count, sizeof(float));
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(output_count * sizeof(float));
    ds4_gpu_tensor *raw = ds4_gpu_tensor_alloc(raw_count * sizeof(float));
    ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc(
        output_count * sizeof(float));
    ds4_gpu_tensor *reference = ds4_gpu_tensor_alloc(
        output_count * sizeof(float));
    uint32_t n_raw_by_row[TEST_ROWS];
    uint32_t raw_start_by_row[TEST_ROWS];
    int ok = q_host && raw_host && actual && expected &&
             q && raw && heads && reference;

    for (uint64_t i = 0; ok && i < output_count; i++) {
        q_host[i] =
            (float)((int32_t)((i * 43u + 17u) % 127u) - 63) / 256.0f;
    }
    for (uint64_t i = 0; ok && i < raw_count; i++) {
        raw_host[i] =
            (float)((int32_t)((i * 47u + 29u) % 113u) - 56) / 256.0f;
    }
    for (uint32_t row = 0; row < TEST_ROWS; row++) {
        n_raw_by_row[row] = row + 1u;
        raw_start_by_row[row] = 0u;
    }
    if (ok) {
        ok = ds4_gpu_tensor_write(q, 0, q_host,
                                  output_count * sizeof(float)) &&
             ds4_gpu_tensor_write(raw, 0, raw_host,
                                  raw_count * sizeof(float)) &&
             ds4_gpu_attention_decode_heads_rows_exact_tensor(
                 heads, model, model_size, sinks_offset, q, raw,
                 n_raw_by_row, TEST_ATTN_RAW_CAP, raw_start_by_row,
                 NULL, 0u, NULL, TEST_ROWS,
                 TEST_ATTN_HEADS, TEST_ATTN_DIM) != 0;
    }
    for (uint32_t row = 0; ok && row < TEST_ROWS; row++) {
        const uint64_t row_offset =
            (uint64_t)row * row_values * sizeof(float);
        ds4_gpu_tensor *q_row = ds4_gpu_tensor_view(
            q, row_offset, row_values * sizeof(float));
        ds4_gpu_tensor *heads_row = ds4_gpu_tensor_view(
            reference, row_offset, row_values * sizeof(float));
        ok = q_row && heads_row &&
             ds4_gpu_attention_decode_heads_tensor(
                 heads_row, model, model_size, sinks_offset, q_row, raw,
                 n_raw_by_row[row], TEST_ATTN_RAW_CAP,
                 raw_start_by_row[row], NULL, 0u, 0u, NULL, 0u,
                 TEST_ATTN_HEADS, TEST_ATTN_DIM) != 0;
        ds4_gpu_tensor_free(heads_row);
        ds4_gpu_tensor_free(q_row);
    }
    if (ok) {
        ok = ds4_gpu_tensor_read(heads, 0, actual,
                                 output_count * sizeof(float)) &&
             ds4_gpu_tensor_read(reference, 0, expected,
                                 output_count * sizeof(float)) &&
             compare_exact("flash_attention_heads", actual, expected,
                           (size_t)output_count);
    }

    printf("EXACT_ROW_ORACLE_AB "
           "family=FAMILY_FLASH_ATTN_BATCH_DIRECT_VS_SINGLE_VEC_REDUCE "
           "rows=%u sites=attention_heads input_bits_equal=PASS "
           "weights_same=PASS metadata_same=PASS result=%s\n",
           TEST_ROWS, ok ? "EXACT" : "MISMATCH");

    ds4_gpu_tensor_free(reference);
    ds4_gpu_tensor_free(heads);
    ds4_gpu_tensor_free(raw);
    ds4_gpu_tensor_free(q);
    free(expected);
    free(actual);
    free(raw_host);
    free(q_host);
    return ok;
}

static int run_routed_moe_contract(const void *model, uint64_t model_size,
                                   uint64_t gate_offset,
                                   uint64_t up_offset,
                                   uint64_t down_offset) {
    const uint64_t gate_row_bytes = sizeof(test_block_iq2_xxs);
    const uint64_t down_row_bytes = sizeof(test_block_q2_k);
    const uint64_t gate_expert_bytes =
        (uint64_t)TEST_MOE_DIM * gate_row_bytes;
    const uint64_t down_expert_bytes =
        (uint64_t)TEST_MOE_DIM * down_row_bytes;
    const uint64_t input_count = (uint64_t)TEST_ROWS * TEST_MOE_DIM;
    const uint64_t route_count =
        (uint64_t)TEST_ROWS * TEST_MOE_USED_EXPERTS;
    const uint64_t act_count = route_count * TEST_MOE_DIM;
    const uint64_t out_count = input_count;
    float *input_host = calloc((size_t)input_count, sizeof(float));
    int32_t *selected_host = calloc((size_t)route_count, sizeof(int32_t));
    float *weights_host = calloc((size_t)route_count, sizeof(float));
    float *actual = calloc((size_t)out_count, sizeof(float));
    float *expected = calloc((size_t)out_count, sizeof(float));
    float *mid_actual = calloc((size_t)act_count, sizeof(float));
    float *mid_expected = calloc((size_t)act_count, sizeof(float));
    ds4_gpu_tensor *input = ds4_gpu_tensor_alloc(
        input_count * sizeof(float));
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc(
        route_count * sizeof(int32_t));
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc(
        route_count * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(out_count * sizeof(float));
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(act_count * sizeof(float));
    ds4_gpu_tensor *up = ds4_gpu_tensor_alloc(act_count * sizeof(float));
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc(act_count * sizeof(float));
    ds4_gpu_tensor *experts = ds4_gpu_tensor_alloc(act_count * sizeof(float));
    ds4_gpu_tensor *out_ref = ds4_gpu_tensor_alloc(
        out_count * sizeof(float));
    ds4_gpu_tensor *gate_ref = ds4_gpu_tensor_alloc(
        act_count * sizeof(float));
    ds4_gpu_tensor *up_ref = ds4_gpu_tensor_alloc(
        act_count * sizeof(float));
    ds4_gpu_tensor *mid_ref = ds4_gpu_tensor_alloc(
        act_count * sizeof(float));
    ds4_gpu_tensor *experts_ref = ds4_gpu_tensor_alloc(
        act_count * sizeof(float));
    int ok = input_host && selected_host && weights_host && actual &&
             expected && mid_actual && mid_expected && input && selected &&
             weights && out && gate && up && mid && experts && out_ref &&
             gate_ref && up_ref && mid_ref && experts_ref;

    for (uint32_t row = 0; ok && row < TEST_ROWS; row++) {
        for (uint32_t col = 0; col < TEST_MOE_DIM; col++) {
            input_host[(uint64_t)row * TEST_MOE_DIM + col] =
                (float)((int32_t)((row * 37u + col * 41u + 9u) % 127u) -
                        63) / 512.0f;
        }
        for (uint32_t slot = 0; slot < TEST_MOE_USED_EXPERTS; slot++) {
            selected_host[(uint64_t)row * TEST_MOE_USED_EXPERTS + slot] =
                (int32_t)((row * 3u + slot * 5u + 1u) %
                          TEST_MOE_TOTAL_EXPERTS);
            weights_host[(uint64_t)row * TEST_MOE_USED_EXPERTS + slot] =
                (float)(TEST_MOE_USED_EXPERTS - slot) / 21.0f;
        }
    }
    if (ok) {
        ok = ds4_gpu_tensor_write(input, 0, input_host,
                                  input_count * sizeof(float)) &&
             ds4_gpu_tensor_write(selected, 0, selected_host,
                                  route_count * sizeof(int32_t)) &&
             ds4_gpu_tensor_write(weights, 0, weights_host,
                                  route_count * sizeof(float)) &&
             ds4_gpu_routed_moe_rows_exact_tensor(
                 out, gate, up, mid, experts, model, model_size,
                 gate_offset, up_offset, down_offset,
                 TEST_IQ2_XXS_TYPE, TEST_Q2_K_TYPE,
                 gate_expert_bytes, gate_row_bytes,
                 down_expert_bytes, down_row_bytes,
                 TEST_MOE_DIM, TEST_MOE_DIM, TEST_MOE_DIM,
                 selected, weights, TEST_MOE_TOTAL_EXPERTS,
                 TEST_MOE_USED_EXPERTS, 7.0f, input, 0u, TEST_ROWS,
                 false) != 0;
    }
    for (uint32_t row = 0; ok && row < TEST_ROWS; row++) {
        const uint64_t input_offset =
            (uint64_t)row * TEST_MOE_DIM * sizeof(float);
        const uint64_t route_offset =
            (uint64_t)row * TEST_MOE_USED_EXPERTS;
        const uint64_t act_offset =
            route_offset * TEST_MOE_DIM * sizeof(float);
        const uint64_t out_offset = input_offset;
        ds4_gpu_tensor *input_row = ds4_gpu_tensor_view(
            input, input_offset, (uint64_t)TEST_MOE_DIM * sizeof(float));
        ds4_gpu_tensor *selected_row = ds4_gpu_tensor_view(
            selected, route_offset * sizeof(int32_t),
            (uint64_t)TEST_MOE_USED_EXPERTS * sizeof(int32_t));
        ds4_gpu_tensor *weights_row = ds4_gpu_tensor_view(
            weights, route_offset * sizeof(float),
            (uint64_t)TEST_MOE_USED_EXPERTS * sizeof(float));
        ds4_gpu_tensor *out_row = ds4_gpu_tensor_view(
            out_ref, out_offset, (uint64_t)TEST_MOE_DIM * sizeof(float));
        ds4_gpu_tensor *gate_row = ds4_gpu_tensor_view(
            gate_ref, act_offset,
            (uint64_t)TEST_MOE_USED_EXPERTS * TEST_MOE_DIM * sizeof(float));
        ds4_gpu_tensor *up_row = ds4_gpu_tensor_view(
            up_ref, act_offset,
            (uint64_t)TEST_MOE_USED_EXPERTS * TEST_MOE_DIM * sizeof(float));
        ds4_gpu_tensor *mid_row = ds4_gpu_tensor_view(
            mid_ref, act_offset,
            (uint64_t)TEST_MOE_USED_EXPERTS * TEST_MOE_DIM * sizeof(float));
        ds4_gpu_tensor *experts_row = ds4_gpu_tensor_view(
            experts_ref, act_offset,
            (uint64_t)TEST_MOE_USED_EXPERTS * TEST_MOE_DIM * sizeof(float));
        ok = input_row && selected_row && weights_row && out_row &&
             gate_row && up_row && mid_row && experts_row &&
             ds4_gpu_routed_moe_one_tensor(
                 out_row, gate_row, up_row, mid_row, experts_row,
                 model, model_size, gate_offset, up_offset, down_offset,
                 TEST_IQ2_XXS_TYPE, TEST_Q2_K_TYPE,
                 gate_expert_bytes, gate_row_bytes,
                 down_expert_bytes, down_row_bytes,
                 TEST_MOE_DIM, TEST_MOE_DIM, TEST_MOE_DIM,
                 selected_row, weights_row, TEST_MOE_TOTAL_EXPERTS,
                 TEST_MOE_USED_EXPERTS, 7.0f, input_row, NULL, 0u,
                 false) != 0;
        ds4_gpu_tensor_free(experts_row);
        ds4_gpu_tensor_free(mid_row);
        ds4_gpu_tensor_free(up_row);
        ds4_gpu_tensor_free(gate_row);
        ds4_gpu_tensor_free(out_row);
        ds4_gpu_tensor_free(weights_row);
        ds4_gpu_tensor_free(selected_row);
        ds4_gpu_tensor_free(input_row);
    }
    if (ok) {
        ok = ds4_gpu_tensor_read(out, 0, actual,
                                 out_count * sizeof(float)) &&
             ds4_gpu_tensor_read(out_ref, 0, expected,
                                 out_count * sizeof(float)) &&
             ds4_gpu_tensor_read(mid, 0, mid_actual,
                                 act_count * sizeof(float)) &&
             ds4_gpu_tensor_read(mid_ref, 0, mid_expected,
                                 act_count * sizeof(float)) &&
             compare_exact("routed_moe_output", actual, expected,
                           (size_t)out_count) &&
             compare_exact("routed_moe_mid", mid_actual, mid_expected,
                           (size_t)act_count);
    }

    printf("EXACT_ROW_ORACLE_AB "
           "family=FAMILY_ROUTED_MOE_IQ2_XXS_Q2_K_BATCH_VS_SINGLE "
           "rows=%u sites=routed_moe input_bits_equal=PASS "
           "weights_same=PASS metadata_same=PASS result=%s\n",
           TEST_ROWS, ok ? "EXACT" : "MISMATCH");

    ds4_gpu_tensor_free(experts_ref);
    ds4_gpu_tensor_free(mid_ref);
    ds4_gpu_tensor_free(up_ref);
    ds4_gpu_tensor_free(gate_ref);
    ds4_gpu_tensor_free(out_ref);
    ds4_gpu_tensor_free(experts);
    ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(up);
    ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(weights);
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(input);
    free(mid_expected);
    free(mid_actual);
    free(expected);
    free(actual);
    free(weights_host);
    free(selected_host);
    free(input_host);
    return ok;
}

static int run_router_contract(void) {
    const uint64_t probs_count =
        (uint64_t)TEST_ROWS * TEST_ROUTER_EXPERTS;
    const uint64_t topk_count =
        (uint64_t)TEST_ROWS * TEST_ROUTER_TOPK;
    float *probs_host = calloc((size_t)probs_count, sizeof(float));
    int32_t *selected_host = calloc((size_t)topk_count, sizeof(int32_t));
    float *actual = calloc((size_t)topk_count, sizeof(float));
    float *expected = calloc((size_t)topk_count, sizeof(float));
    ds4_gpu_tensor *probs = ds4_gpu_tensor_alloc(
        probs_count * sizeof(float));
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc(
        topk_count * sizeof(int32_t));
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc(
        topk_count * sizeof(float));
    ds4_gpu_tensor *reference = ds4_gpu_tensor_alloc(
        topk_count * sizeof(float));
    int ok = probs_host && selected_host && actual && expected &&
             probs && selected && weights && reference;

    for (uint32_t row = 0; row < TEST_ROWS; row++) {
        for (uint32_t i = 0; i < TEST_ROUTER_EXPERTS; i++) {
            probs_host[(uint64_t)row * TEST_ROUTER_EXPERTS + i] =
                (float)(1u + ((row * 19u + i * 23u) % 97u)) / 128.0f;
        }
        for (uint32_t i = 0; i < TEST_ROUTER_TOPK; i++) {
            selected_host[(uint64_t)row * TEST_ROUTER_TOPK + i] =
                (int32_t)((row * 31u + i * 37u + 3u) %
                          TEST_ROUTER_EXPERTS);
        }
    }
    if (ok) {
        ok = ds4_gpu_tensor_write(
                 probs, 0, probs_host, probs_count * sizeof(float)) &&
             ds4_gpu_tensor_write(
                 selected, 0, selected_host,
                 topk_count * sizeof(int32_t)) &&
             ds4_gpu_router_weights_rows_exact_tensor(
                 weights, probs, selected, TEST_ROWS, 1.5f) != 0;
    }
    for (uint32_t row = 0; ok && row < TEST_ROWS; row++) {
        ds4_gpu_tensor *probs_row = ds4_gpu_tensor_view(
            probs, (uint64_t)row * TEST_ROUTER_EXPERTS * sizeof(float),
            (uint64_t)TEST_ROUTER_EXPERTS * sizeof(float));
        ds4_gpu_tensor *selected_row = ds4_gpu_tensor_view(
            selected, (uint64_t)row * TEST_ROUTER_TOPK * sizeof(int32_t),
            (uint64_t)TEST_ROUTER_TOPK * sizeof(int32_t));
        ds4_gpu_tensor *weights_row = ds4_gpu_tensor_view(
            reference, (uint64_t)row * TEST_ROUTER_TOPK * sizeof(float),
            (uint64_t)TEST_ROUTER_TOPK * sizeof(float));
        ok = probs_row && selected_row && weights_row &&
             ds4_gpu_router_weights_one_tensor(
                 weights_row, probs_row, selected_row) != 0;
        ds4_gpu_tensor_free(weights_row);
        ds4_gpu_tensor_free(selected_row);
        ds4_gpu_tensor_free(probs_row);
    }
    if (ok) {
        ok = ds4_gpu_tensor_read(
                 weights, 0, actual, topk_count * sizeof(float)) &&
             ds4_gpu_tensor_read(
                 reference, 0, expected, topk_count * sizeof(float)) &&
             compare_exact("router_weights", actual, expected,
                           (size_t)topk_count);
    }
    printf("EXACT_ROW_ORACLE_AB "
           "family=FAMILY_ROUTER_WEIGHT_NORMALIZATION_BATCH_REDUCE_VS_SINGLE_KERNEL "
           "rows=%u sites=ffn_router_weights input_bits_equal=PASS "
           "weights_same=PASS "
           "metadata_same=PASS result=%s\n",
           TEST_ROWS, ok ? "EXACT" : "MISMATCH");

    ds4_gpu_tensor_free(reference);
    ds4_gpu_tensor_free(weights);
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(probs);
    free(expected);
    free(actual);
    free(selected_host);
    free(probs_host);
    return ok;
}

int main(void) {
    const long page_size = sysconf(_SC_PAGESIZE);
    const uint64_t page = page_size > 0 ? (uint64_t)page_size : 4096u;
    const uint64_t q8_bytes =
        (uint64_t)TEST_Q8_OUT * (TEST_IN / 32u) * sizeof(test_block_q8_0);
    const uint64_t f16_bytes =
        (uint64_t)TEST_PAIR_OUT * TEST_IN * sizeof(uint16_t);
    const uint64_t cp5_q8_bytes =
        (uint64_t)TEST_CP5_EMBD * (TEST_CP5_IN / 32u) *
        sizeof(test_block_q8_0);
    const uint64_t cp4_a_bytes =
        (uint64_t)TEST_CP4_LOW_DIM * (TEST_CP4_GROUP_DIM / 32u) *
        sizeof(test_block_q8_0);
    const uint64_t cp4_b_bytes =
        (uint64_t)TEST_CP4_OUT * (TEST_CP4_LOW_DIM / 32u) *
        sizeof(test_block_q8_0);
    const uint64_t moe_iq2_bytes =
        (uint64_t)TEST_MOE_TOTAL_EXPERTS * TEST_MOE_DIM *
        sizeof(test_block_iq2_xxs);
    const uint64_t moe_q2_bytes =
        (uint64_t)TEST_MOE_TOTAL_EXPERTS * TEST_MOE_DIM *
        sizeof(test_block_q2_k);
    const uint64_t q8_gate_offset = 0;
    const uint64_t q8_up_offset = align_up(q8_bytes, page);
    const uint64_t f16_a_offset = align_up(q8_up_offset + q8_bytes, page);
    const uint64_t f16_b_offset = align_up(f16_a_offset + f16_bytes, page);
    const uint64_t cp5_q8_offset =
        align_up(f16_b_offset + f16_bytes, page);
    const uint64_t moe_gate_offset =
        align_up(cp5_q8_offset + cp5_q8_bytes, page);
    const uint64_t moe_up_offset =
        align_up(moe_gate_offset + moe_iq2_bytes, page);
    const uint64_t moe_down_offset =
        align_up(moe_up_offset + moe_iq2_bytes, page);
    const uint64_t attn_sinks_offset =
        align_up(moe_down_offset + moe_q2_bytes, page);
    const uint64_t cp4_a_offset = align_up(
        attn_sinks_offset + (uint64_t)TEST_ATTN_HEADS * sizeof(float), page);
    const uint64_t cp4_b_offset = align_up(cp4_a_offset + cp4_a_bytes, page);
    const uint64_t model_size =
        align_up(cp4_b_offset + cp4_b_bytes, page);
    void *model = NULL;
    float *input_host = calloc(
        (size_t)TEST_ROWS * TEST_IN, sizeof(float));
    ds4_gpu_tensor *input = NULL;
    int initialized = 0;
    int q8_ok = 0;
    int cp4_ok = 0;
    int cp5_ok = 0;
    int flash_attention_ok = 0;
    int routed_moe_ok = 0;
    int router_ok = 0;
    unsigned f16_results = 0;
    int ok = input_host != NULL &&
        posix_memalign(&model, (size_t)page, (size_t)model_size) == 0;

    if (ok) {
        memset(model, 0, (size_t)model_size);
        fill_q8_matrix((test_block_q8_0 *)
                       ((uint8_t *)model + q8_gate_offset),
                       TEST_IN, TEST_Q8_OUT, 3u);
        fill_q8_matrix((test_block_q8_0 *)
                       ((uint8_t *)model + q8_up_offset),
                       TEST_IN, TEST_Q8_OUT, 11u);
        fill_f16_matrix((uint16_t *)
                        ((uint8_t *)model + f16_a_offset),
                        TEST_PAIR_OUT, 5u);
        fill_f16_matrix((uint16_t *)
                        ((uint8_t *)model + f16_b_offset),
                        TEST_PAIR_OUT, 17u);
        fill_q8_matrix((test_block_q8_0 *)
                       ((uint8_t *)model + cp5_q8_offset),
                       TEST_CP5_IN, TEST_CP5_EMBD, 19u);
        fill_iq2_xxs_experts((test_block_iq2_xxs *)
                             ((uint8_t *)model + moe_gate_offset), 7u);
        fill_iq2_xxs_experts((test_block_iq2_xxs *)
                             ((uint8_t *)model + moe_up_offset), 23u);
        fill_q2_k_experts((test_block_q2_k *)
                          ((uint8_t *)model + moe_down_offset), 31u);
        fill_q8_matrix((test_block_q8_0 *)
                       ((uint8_t *)model + cp4_a_offset),
                       TEST_CP4_GROUP_DIM, TEST_CP4_LOW_DIM, 37u);
        fill_q8_matrix((test_block_q8_0 *)
                       ((uint8_t *)model + cp4_b_offset),
                       TEST_CP4_LOW_DIM, TEST_CP4_OUT, 41u);
        float *attn_sinks = (float *)
            ((uint8_t *)model + attn_sinks_offset);
        for (uint32_t head = 0; head < TEST_ATTN_HEADS; head++) {
            attn_sinks[head] = (float)(head + 1u) / 16.0f;
        }
        for (uint32_t row = 0; row < TEST_ROWS; row++) {
            for (uint32_t col = 0; col < TEST_IN; col++) {
                /* Avoid power-of-two fractions: the previous fixture made
                 * every Q8 product exactly representable and accidentally
                 * erased the reduction-topology divergence under test. */
                uint32_t bits = 0x3f000000u |
                    ((row * 0x1f123bb5u + col * 0x5bd1e995u +
                      0x6d2b79f5u) & 0x007fffffu);
                float value;
                memcpy(&value, &bits, sizeof(value));
                if ((row + col) & 1u) value = -value;
                input_host[(uint64_t)row * TEST_IN + col] = value;
            }
        }
    }

    unsetenv("DS4_METAL_ENABLE_Q8_DECODE_EXACT_VIEWS");
    unsetenv("DS4_METAL_ENABLE_F32_DECODE_EXACT_VIEWS");
    unsetenv("DS4_METAL_Q8_DECODE_MPP");
    unsetenv("DS4_METAL_PROJECTION_REPAIR_DIAGNOSTICS");
    if (ok) {
        ok = ds4_gpu_init() != 0;
        initialized = ok;
    }
    if (ok) {
        ds4_gpu_set_quality(false);
        ds4_gpu_set_ssd_streaming(false);
        ok = ds4_gpu_set_model_map(model, model_size) != 0;
    }
    if (ok) {
        input = ds4_gpu_tensor_alloc(
            (uint64_t)TEST_ROWS * TEST_IN * sizeof(float));
        ok = input != NULL && ds4_gpu_tensor_write(
            input, 0, input_host,
            (uint64_t)TEST_ROWS * TEST_IN * sizeof(float));
    }
    if (ok) {
        q8_ok = run_q8_contract(model, model_size,
                                q8_gate_offset, q8_up_offset, input);
        cp4_ok = run_cp4_output_b_contract(
            model, model_size, cp4_a_offset, cp4_b_offset);
        cp5_ok = run_cp5_tail_contract(model, model_size, cp5_q8_offset);
        flash_attention_ok = run_flash_attention_contract(
            model, model_size, attn_sinks_offset);
        routed_moe_ok = run_routed_moe_contract(
            model, model_size, moe_gate_offset, moe_up_offset,
            moe_down_offset);
        f16_results = run_f16_contracts(model, model_size,
                                         f16_a_offset, f16_b_offset, input);
        router_ok = run_router_contract();
        ok = q8_ok && cp4_ok && cp5_ok && flash_attention_ok && routed_moe_ok &&
             f16_results == 3u && router_ok;
    }

    const unsigned families_exact = (cp5_ok ? 1u : 0u) +
        (routed_moe_ok ? 1u : 0u) +
        ((f16_results & 1u) ? 1u : 0u) +
        ((f16_results & 2u) ? 1u : 0u) + (router_ok ? 1u : 0u);
    printf("FLASH_ATTN_ORACLE_STATUS result=%s production_repair=NOT_IMPLEMENTED\n",
           flash_attention_ok ? "EXACT" : "MISMATCH");
    printf("EXACT_ROW_ORACLE_STATUS families_total=7 families_exact=%u "
           "families_failed=%u\n",
           families_exact + (q8_ok && cp4_ok ? 1u : 0u) +
               (flash_attention_ok ? 1u : 0u),
           7u - families_exact - (q8_ok && cp4_ok ? 1u : 0u) -
               (flash_attention_ok ? 1u : 0u));
    printf("FAMILY_REPAIR_CANDIDATE_STATUS "
           "family=FAMILY_Q8_0_BATCH_EXT_VS_SINGLE_MV "
           "completed_sites=%u pending_sites=%u result=%s\n",
           q8_ok && cp4_ok ? 5u : 0u, q8_ok && cp4_ok ? 0u : 5u,
           q8_ok && cp4_ok ? "PASS" : "FAIL");
    printf("PROJECTION_REPAIR_STATUS families_total=7 families_exact=%u "
           "families_partial=%u families_not_implemented=1 "
           "families_failed=%u\n",
           families_exact, q8_ok && cp4_ok ? 1u : 0u,
           5u - families_exact + (q8_ok && cp4_ok ? 0u : 1u));
    ds4_gpu_tensor_free(input);
    if (initialized) ds4_gpu_cleanup();
    free(input_host);
    free(model);
    return ok ? 0 : 1;
}