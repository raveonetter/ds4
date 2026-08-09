#define _DARWIN_C_SOURCE

#include "ds4_float_compare.h"
#include "ds4_gpu.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum {
    TEST_ROWS = 5,
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
};

typedef struct __attribute__((packed)) {
    uint16_t d;
    int8_t qs[32];
} test_block_q8_0;

_Static_assert(sizeof(test_block_q8_0) == 34, "Q8_0 block layout changed");

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

static int run_q8_contract(const void *model, uint64_t model_size,
                           uint64_t gate_offset, uint64_t up_offset,
                           ds4_gpu_tensor *input) {
    const uint64_t count = (uint64_t)TEST_ROWS * TEST_Q8_OUT;
    const uint64_t bytes = count * sizeof(float);
    float *actual = calloc((size_t)count, sizeof(float));
    float *expected = calloc((size_t)count, sizeof(float));
    float *gate_actual = calloc((size_t)count, sizeof(float));
    float *gate_expected = calloc((size_t)count, sizeof(float));
    float *up_actual = calloc((size_t)count, sizeof(float));
    float *up_expected = calloc((size_t)count, sizeof(float));
    float *mid_actual = calloc((size_t)count, sizeof(float));
    float *mid_expected = calloc((size_t)count, sizeof(float));
    ds4_gpu_tensor *batch = ds4_gpu_tensor_alloc(bytes);
    ds4_gpu_tensor *reference = ds4_gpu_tensor_alloc(bytes);
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(bytes);
    ds4_gpu_tensor *up = ds4_gpu_tensor_alloc(bytes);
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc(bytes);
    ds4_gpu_tensor *gate_ref = ds4_gpu_tensor_alloc(bytes);
    ds4_gpu_tensor *up_ref = ds4_gpu_tensor_alloc(bytes);
    ds4_gpu_tensor *mid_ref = ds4_gpu_tensor_alloc(bytes);
    int ok = actual && expected && gate_actual && gate_expected &&
             up_actual && up_expected && mid_actual && mid_expected &&
             batch && reference && gate && up && mid &&
             gate_ref && up_ref && mid_ref;

    if (ok) {
        ok = ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(
                 batch, model, model_size, gate_offset,
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
        ok = ds4_gpu_shared_gate_up_swiglu_q8_0_rows_scalar_tensor(
                 gate, up, mid, model, model_size,
                 gate_offset, up_offset, TEST_IN, TEST_Q8_OUT,
                 input, TEST_ROWS, 7.0f) != 0;
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
        ok = ds4_gpu_tensor_read(batch, 0, actual, bytes) &&
             ds4_gpu_tensor_read(reference, 0, expected, bytes) &&
             ds4_gpu_tensor_read(gate, 0, gate_actual, bytes) &&
             ds4_gpu_tensor_read(gate_ref, 0, gate_expected, bytes) &&
             ds4_gpu_tensor_read(up, 0, up_actual, bytes) &&
             ds4_gpu_tensor_read(up_ref, 0, up_expected, bytes) &&
             ds4_gpu_tensor_read(mid, 0, mid_actual, bytes) &&
             ds4_gpu_tensor_read(mid_ref, 0, mid_expected, bytes) &&
             compare_exact("q8_projection", actual, expected, (size_t)count) &&
             compare_exact("q8_fused_gate", gate_actual, gate_expected,
                           (size_t)count) &&
             compare_exact("q8_fused_up", up_actual, up_expected,
                           (size_t)count) &&
             compare_exact("q8_fused_mid", mid_actual, mid_expected,
                           (size_t)count);
    }

    printf("PRODUCTION_FAMILY_AB family=FAMILY_Q8_0_BATCH_EXT_VS_SINGLE_MV "
           "rows=%u sites=QA,KV,QB,CP4_output_B,shared_gate_up "
           "input_bits_equal=PASS weights_same=PASS metadata_same=PASS "
           "result=%s\n", TEST_ROWS, ok ? "EXACT" : "MISMATCH");

    ds4_gpu_tensor_free(mid_ref);
    ds4_gpu_tensor_free(up_ref);
    ds4_gpu_tensor_free(gate_ref);
    ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(up);
    ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(reference);
    ds4_gpu_tensor_free(batch);
    free(mid_expected);
    free(mid_actual);
    free(up_expected);
    free(up_actual);
    free(gate_expected);
    free(gate_actual);
    free(expected);
    free(actual);
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
    float *pair_a_actual = calloc((size_t)pair_count, sizeof(float));
    float *pair_a_expected = calloc((size_t)pair_count, sizeof(float));
    float *pair_b_actual = calloc((size_t)pair_count, sizeof(float));
    float *pair_b_expected = calloc((size_t)pair_count, sizeof(float));
    ds4_gpu_tensor *single = ds4_gpu_tensor_alloc(
        single_count * sizeof(float));
    ds4_gpu_tensor *single_ref = ds4_gpu_tensor_alloc(
        single_count * sizeof(float));
    ds4_gpu_tensor *pair_a = ds4_gpu_tensor_alloc(pair_count * sizeof(float));
    ds4_gpu_tensor *pair_b = ds4_gpu_tensor_alloc(pair_count * sizeof(float));
    ds4_gpu_tensor *pair_a_ref = ds4_gpu_tensor_alloc(
        pair_count * sizeof(float));
    ds4_gpu_tensor *pair_b_ref = ds4_gpu_tensor_alloc(
        pair_count * sizeof(float));
    int setup_ok = single_actual && single_expected && pair_a_actual &&
                   pair_a_expected && pair_b_actual && pair_b_expected &&
                   single && single_ref && pair_a && pair_b &&
                   pair_a_ref && pair_b_ref;
    int single_ok = setup_ok;
    int pair_ok = setup_ok;

    if (single_ok) {
        single_ok = ds4_gpu_matmul_f16_decode_rows_exact_tensor(
                        single, model, model_size, weight_a_offset,
                        TEST_IN, TEST_F16_OUT, input, TEST_ROWS) != 0;
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
    if (single_ok) {
        single_ok = ds4_gpu_tensor_read(
                        single, 0, single_actual,
                        single_count * sizeof(float)) &&
                    ds4_gpu_tensor_read(
                        single_ref, 0, single_expected,
                        single_count * sizeof(float)) &&
                    compare_exact("f16_projection", single_actual,
                                  single_expected, (size_t)single_count);
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

    printf("PRODUCTION_FAMILY_AB family=FAMILY_F16_BATCH_EXT_VS_SINGLE_MV "
           "rows=%u sites=hc_attn_pre_split,hc_ffn_pre,ffn_router_projection "
           "input_bits_equal=PASS weights_same=PASS "
           "metadata_same=PASS result=%s\n",
           TEST_ROWS, single_ok ? "EXACT" : "MISMATCH");
    printf("PRODUCTION_FAMILY_AB "
           "family=FAMILY_F16_BATCH_EXT_VS_SINGLE_PAIR_MV rows=%u "
           "sites=attention_kv,attention_score,indexer_kv,indexer_score "
           "input_bits_equal=PASS weights_same=PASS metadata_same=PASS "
           "result=%s\n", TEST_ROWS, pair_ok ? "EXACT" : "MISMATCH");

    ds4_gpu_tensor_free(pair_b_ref);
    ds4_gpu_tensor_free(pair_a_ref);
    ds4_gpu_tensor_free(pair_b);
    ds4_gpu_tensor_free(pair_a);
    ds4_gpu_tensor_free(single_ref);
    ds4_gpu_tensor_free(single);
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

    printf("PRODUCTION_FAMILY_AB "
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
    printf("PRODUCTION_FAMILY_AB "
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
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t q8_bytes =
        (uint64_t)TEST_Q8_OUT * (TEST_IN / 32u) * sizeof(test_block_q8_0);
    const uint64_t f16_bytes =
        (uint64_t)TEST_PAIR_OUT * TEST_IN * sizeof(uint16_t);
    const uint64_t cp5_q8_bytes =
        (uint64_t)TEST_CP5_EMBD * (TEST_CP5_IN / 32u) *
        sizeof(test_block_q8_0);
    const uint64_t q8_gate_offset = 0;
    const uint64_t q8_up_offset = align_up(q8_bytes, page);
    const uint64_t f16_a_offset = align_up(q8_up_offset + q8_bytes, page);
    const uint64_t f16_b_offset = align_up(f16_a_offset + f16_bytes, page);
    const uint64_t cp5_q8_offset =
        align_up(f16_b_offset + f16_bytes, page);
    const uint64_t model_size =
        align_up(cp5_q8_offset + cp5_q8_bytes, page);
    void *model = NULL;
    float *input_host = calloc(
        (size_t)TEST_ROWS * TEST_IN, sizeof(float));
    ds4_gpu_tensor *input = NULL;
    int initialized = 0;
    int q8_ok = 0;
    int cp5_ok = 0;
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
        for (uint32_t row = 0; row < TEST_ROWS; row++) {
            for (uint32_t col = 0; col < TEST_IN; col++) {
                const int32_t raw =
                    (int32_t)((row * 29u + col * 31u + 7u) % 127u) - 63;
                input_host[(uint64_t)row * TEST_IN + col] =
                    (float)raw / 256.0f;
            }
        }
    }

    unsetenv("DS4_METAL_ENABLE_Q8_DECODE_EXACT_VIEWS");
    unsetenv("DS4_METAL_ENABLE_F32_DECODE_EXACT_VIEWS");
    setenv("DS4_METAL_PROJECTION_REPAIR_DIAGNOSTICS", "1", 1);
    if (ok) {
        ok = ds4_gpu_init() != 0;
        initialized = ok;
    }
    if (ok) {
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
        cp5_ok = run_cp5_tail_contract(model, model_size, cp5_q8_offset);
        f16_results = run_f16_contracts(model, model_size,
                                         f16_a_offset, f16_b_offset, input);
        router_ok = run_router_contract();
        ok = q8_ok && cp5_ok && f16_results == 3u && router_ok;
    }

    const unsigned families_exact = (q8_ok ? 1u : 0u) +
        (cp5_ok ? 1u : 0u) +
        ((f16_results & 1u) ? 1u : 0u) +
        ((f16_results & 2u) ? 1u : 0u) + (router_ok ? 1u : 0u);
    printf("PROJECTION_REPAIR_STATUS families_total=5 families_exact=%u "
           "families_failed=%u\n", families_exact, 5u - families_exact);
    ds4_gpu_tensor_free(input);
    if (initialized) ds4_gpu_cleanup();
    free(input_host);
    free(model);
    return ok ? 0 : 1;
}
