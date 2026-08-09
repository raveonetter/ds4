#include "family_repair.h"

#include <ctype.h>
#include <stdlib.h>
#include <string.h>

#define DS4_REPAIR_FAMILY_BIT(family) (1u << (unsigned)(family))
#define DS4_REPAIR_FAMILY_ALL_MASK \
    ((1u << (unsigned)DS4_REPAIR_FAMILY_COUNT) - 1u)

/* Frozen from the completed 177-site canonical sweep.  P2 repairs the four
 * pure projection/reduction families; P3 repairs the remaining mixed
 * families.  The site taxonomy and proven-site counts stay unchanged. */
static const ds4_family_repair_manifest_entry g_family_manifest[] = {
    {
        DS4_REPAIR_FAMILY_Q8_0_BATCH_EXT_VS_SINGLE_MV,
        "FAMILY_Q8_0_BATCH_EXT_VS_SINGLE_MV",
        "QA,KV,QB,CP4_output_B,shared_gate_up",
        5,
        "kernel_mul_mv_ext_q8_0_f32_r1 batch rows",
        "kernel_mul_mv_q8_0_f32 ordinary single MV",
        DS4_FAMILY_REPAIR_ENTRY_Q8_0_BATCH_EXT,
        "ds4_repair_q8_0_batch_ext",
        DS4_FAMILY_REPAIR_EXACT,
    },
    {
        DS4_REPAIR_FAMILY_FLASH_ATTN_BATCH_DIRECT_VS_SINGLE_VEC_REDUCE,
        "FAMILY_FLASH_ATTN_BATCH_DIRECT_VS_SINGLE_VEC_REDUCE",
        "CP4-HEADS-RAW,mixed_attention_layers",
        42,
        "kernel_flash_attn_ext_f16_dk512_dv512 direct batch output",
        "kernel_flash_attn_ext_vec_f16_dk512_dv512 + kernel_flash_attn_reduce",
        DS4_FAMILY_REPAIR_ENTRY_FLASH_ATTN_BATCH_DIRECT,
        "ds4_repair_flash_attn_batch_direct",
        DS4_FAMILY_REPAIR_NOT_IMPLEMENTED,
    },
    {
        DS4_REPAIR_FAMILY_F16_BATCH_EXT_VS_SINGLE_MV,
        "FAMILY_F16_BATCH_EXT_VS_SINGLE_MV",
        "hc_attn_pre_split,hc_ffn_pre,ffn_router_projection",
        3,
        "F16 batch rows projection",
        "F16 ordinary single MV projection",
        DS4_FAMILY_REPAIR_ENTRY_F16_BATCH_EXT,
        "ds4_repair_f16_batch_ext",
        DS4_FAMILY_REPAIR_EXACT,
    },
    {
        DS4_REPAIR_FAMILY_ROUTER_WEIGHT_NORMALIZATION_BATCH_REDUCE_VS_SINGLE_KERNEL,
        "FAMILY_ROUTER_WEIGHT_NORMALIZATION_BATCH_REDUCE_VS_SINGLE_KERNEL",
        "ffn_router_weights",
        1,
        "get_rows + sum_rows + div_row + mul_scalar",
        "kernel_dsv4_router_weights_one",
        DS4_FAMILY_REPAIR_ENTRY_ROUTER_WEIGHT_NORMALIZATION,
        "ds4_repair_router_weight_normalization",
        DS4_FAMILY_REPAIR_EXACT,
    },
    {
        DS4_REPAIR_FAMILY_ROUTED_MOE_IQ2_XXS_Q2_K_BATCH_VS_SINGLE,
        "FAMILY_ROUTED_MOE_IQ2_XXS_Q2_K_BATCH_VS_SINGLE",
        "routed_moe",
        1,
        "routed_moe_batch_iq2_xxs_q2_k",
        "routed_moe_single_iq2_xxs_q2_k",
        DS4_FAMILY_REPAIR_ENTRY_ROUTED_MOE_IQ2_XXS_Q2_K,
        "ds4_repair_routed_moe_iq2_xxs_q2_k",
        DS4_FAMILY_REPAIR_EXACT,
    },
    {
        DS4_REPAIR_FAMILY_Q8_SHARED_DOWN_BATCH_F32_HC_ADD_VS_SINGLE_FUSED_HC,
        "FAMILY_Q8_SHARED_DOWN_BATCH_F32_HC_ADD_VS_SINGLE_FUSED_HC",
        "cp5_tail",
        1,
        "q8_0 batch F32 projection + HC expand/add/split",
        "q8_0 single fused shared-down HC expand",
        DS4_FAMILY_REPAIR_ENTRY_Q8_SHARED_DOWN_BATCH_F32_HC_ADD,
        "ds4_repair_q8_shared_down_batch_f32_hc_add",
        DS4_FAMILY_REPAIR_EXACT,
    },
    {
        DS4_REPAIR_FAMILY_F16_BATCH_EXT_VS_SINGLE_PAIR_MV,
        "FAMILY_F16_BATCH_EXT_VS_SINGLE_PAIR_MV",
        "attention_kv,attention_score,indexer_kv,indexer_score",
        124,
        "F16 batch-row paired compressor/indexer projections",
        "F16 ordinary single-pair MV projections",
        DS4_FAMILY_REPAIR_ENTRY_F16_BATCH_EXT_PAIR,
        "ds4_repair_f16_batch_ext_pair",
        DS4_FAMILY_REPAIR_EXACT,
    },
};

/* A site class describes one semantic call-site group, not a production
 * switch.  Repeated layers share the same family dispatch and repair entry. */
static const ds4_family_repair_site_entry g_family_sites[] = {
    {DS4_REPAIR_SITE_QA, "QA", DS4_REPAIR_FAMILY_Q8_0_BATCH_EXT_VS_SINGLE_MV, 1},
    {DS4_REPAIR_SITE_KV, "KV", DS4_REPAIR_FAMILY_Q8_0_BATCH_EXT_VS_SINGLE_MV, 1},
    {DS4_REPAIR_SITE_QB, "QB", DS4_REPAIR_FAMILY_Q8_0_BATCH_EXT_VS_SINGLE_MV, 1},
    {DS4_REPAIR_SITE_CP4_OUTPUT_B, "CP4_output_B", DS4_REPAIR_FAMILY_Q8_0_BATCH_EXT_VS_SINGLE_MV, 1},
    {DS4_REPAIR_SITE_SHARED_GATE_UP, "shared_gate_up", DS4_REPAIR_FAMILY_Q8_0_BATCH_EXT_VS_SINGLE_MV, 1},
    {DS4_REPAIR_SITE_ATTENTION_HEADS, "attention_heads", DS4_REPAIR_FAMILY_FLASH_ATTN_BATCH_DIRECT_VS_SINGLE_VEC_REDUCE, 42},
    {DS4_REPAIR_SITE_HC_ATTN_PRE_SPLIT, "hc_attn_pre_split", DS4_REPAIR_FAMILY_F16_BATCH_EXT_VS_SINGLE_MV, 1},
    {DS4_REPAIR_SITE_HC_FFN_PRE, "hc_ffn_pre", DS4_REPAIR_FAMILY_F16_BATCH_EXT_VS_SINGLE_MV, 1},
    {DS4_REPAIR_SITE_FFN_ROUTER_PROJECTION, "ffn_router_projection", DS4_REPAIR_FAMILY_F16_BATCH_EXT_VS_SINGLE_MV, 1},
    {DS4_REPAIR_SITE_FFN_ROUTER_WEIGHTS, "ffn_router_weights", DS4_REPAIR_FAMILY_ROUTER_WEIGHT_NORMALIZATION_BATCH_REDUCE_VS_SINGLE_KERNEL, 1},
    {DS4_REPAIR_SITE_ROUTED_MOE, "routed_moe", DS4_REPAIR_FAMILY_ROUTED_MOE_IQ2_XXS_Q2_K_BATCH_VS_SINGLE, 1},
    {DS4_REPAIR_SITE_CP5_TAIL, "cp5_tail", DS4_REPAIR_FAMILY_Q8_SHARED_DOWN_BATCH_F32_HC_ADD_VS_SINGLE_FUSED_HC, 1},
    {DS4_REPAIR_SITE_ATTN_COMPRESSOR_KV, "attention_compressor_kv", DS4_REPAIR_FAMILY_F16_BATCH_EXT_VS_SINGLE_PAIR_MV, 41},
    {DS4_REPAIR_SITE_ATTN_COMPRESSOR_SCORE, "attention_compressor_score", DS4_REPAIR_FAMILY_F16_BATCH_EXT_VS_SINGLE_PAIR_MV, 41},
    {DS4_REPAIR_SITE_INDEXER_KV, "indexer_kv", DS4_REPAIR_FAMILY_F16_BATCH_EXT_VS_SINGLE_PAIR_MV, 21},
    {DS4_REPAIR_SITE_INDEXER_SCORE, "indexer_score", DS4_REPAIR_FAMILY_F16_BATCH_EXT_VS_SINGLE_PAIR_MV, 21},
};

const ds4_family_repair_manifest_entry *ds4_family_repair_manifest(
        size_t *count) {
    if (count) *count = sizeof(g_family_manifest) / sizeof(g_family_manifest[0]);
    return g_family_manifest;
}

const ds4_family_repair_site_entry *ds4_family_repair_sites(size_t *count) {
    if (count) *count = sizeof(g_family_sites) / sizeof(g_family_sites[0]);
    return g_family_sites;
}

const ds4_family_repair_manifest_entry *ds4_family_repair_family(
        ds4_repair_family family) {
    if ((unsigned)family >= (unsigned)DS4_REPAIR_FAMILY_COUNT) return NULL;
    return &g_family_manifest[(unsigned)family];
}

const ds4_family_repair_site_entry *ds4_family_repair_site(
        ds4_repair_site site) {
    if ((unsigned)site >= (unsigned)DS4_REPAIR_SITE_CLASS_COUNT) return NULL;
    return &g_family_sites[(unsigned)site];
}

bool ds4_family_repair_manifest_validate(uint32_t *unclassified_proven_sites) {
    uint32_t totals[DS4_REPAIR_FAMILY_COUNT] = {0};
    uint32_t unclassified = 0;
    size_t i;

    if (sizeof(g_family_manifest) / sizeof(g_family_manifest[0]) !=
            DS4_REPAIR_FAMILY_COUNT ||
        sizeof(g_family_sites) / sizeof(g_family_sites[0]) !=
            DS4_REPAIR_SITE_CLASS_COUNT) {
        return false;
    }
    for (i = 0; i < DS4_REPAIR_FAMILY_COUNT; i++) {
        const ds4_family_repair_manifest_entry *entry = &g_family_manifest[i];
        if (entry->family != (ds4_repair_family)i || !entry->name ||
            !entry->proven_sites || !entry->generic_topology ||
            !entry->canonical_reference ||
            entry->production_repair_entry != (ds4_family_repair_entry)i ||
            !entry->production_repair_entry_name ||
            entry->proven_site_count == 0) {
            return false;
        }
    }
    for (i = 0; i < DS4_REPAIR_SITE_CLASS_COUNT; i++) {
        const ds4_family_repair_site_entry *site = &g_family_sites[i];
        if (site->site != (ds4_repair_site)i || !site->name ||
            site->runtime_sites == 0 ||
            (unsigned)site->family >= (unsigned)DS4_REPAIR_FAMILY_COUNT) {
            unclassified += site->runtime_sites;
            continue;
        }
        totals[(unsigned)site->family] += site->runtime_sites;
    }
    if (unclassified_proven_sites) *unclassified_proven_sites = unclassified;
    if (unclassified != 0) return false;
    for (i = 0; i < DS4_REPAIR_FAMILY_COUNT; i++) {
        if (totals[i] != g_family_manifest[i].proven_site_count) return false;
    }
    for (i = 0, unclassified = 0; i < DS4_REPAIR_FAMILY_COUNT; i++) {
        unclassified += totals[i];
    }
    return unclassified == DS4_FAMILY_REPAIR_PROVEN_SITE_COUNT;
}

static bool ds4_family_repair_token_equal(const char *start, size_t length,
                                          const char *name) {
    return strlen(name) == length && strncmp(start, name, length) == 0;
}

bool ds4_family_repair_parse_selection(const char *value,
                                       ds4_family_repair_mask *mask) {
    const char *cursor;

    if (!mask) return false;
    *mask = 0;
    if (!value || !value[0]) return true;
    cursor = value;
    while (*cursor) {
        const char *start;
        const char *end;
        size_t i;
        bool matched = false;

        while (isspace((unsigned char)*cursor)) cursor++;
        start = cursor;
        while (*cursor && *cursor != ',') cursor++;
        end = cursor;
        while (end > start && isspace((unsigned char)end[-1])) end--;
        if (start == end) return false;
        if (ds4_family_repair_token_equal(start, (size_t)(end - start),
                                          "all")) {
            *mask = DS4_REPAIR_FAMILY_ALL_MASK;
            matched = true;
        } else {
            for (i = 0; i < DS4_REPAIR_FAMILY_COUNT; i++) {
                if (ds4_family_repair_token_equal(
                        start, (size_t)(end - start),
                        g_family_manifest[i].name)) {
                    *mask |= DS4_REPAIR_FAMILY_BIT(i);
                    matched = true;
                    break;
                }
            }
        }
        if (!matched) return false;
        if (*cursor == ',') cursor++;
    }
    return true;
}

bool ds4_family_repair_enabled(ds4_family_repair_mask mask,
                               ds4_repair_family family) {
    return (unsigned)family < (unsigned)DS4_REPAIR_FAMILY_COUNT &&
        (mask & DS4_REPAIR_FAMILY_BIT(family)) != 0;
}

bool ds4_family_repair_runtime_enabled(ds4_repair_family family) {
    ds4_family_repair_mask mask = 0;
    const char *selection = getenv("DS4_FAMILY_REPAIRS");

    if (!selection || !selection[0] ||
        !ds4_family_repair_parse_selection(selection, &mask)) {
        return false;
    }
    return ds4_family_repair_enabled(mask, family) &&
        ds4_family_repair_family(family)->status == DS4_FAMILY_REPAIR_EXACT;
}

bool ds4_family_repair_select(ds4_repair_site site,
                              ds4_family_repair_mask enabled,
                              ds4_family_repair_dispatch *dispatch) {
    const ds4_family_repair_site_entry *site_entry;
    const ds4_family_repair_manifest_entry *family;

    if (!dispatch) return false;
    site_entry = ds4_family_repair_site(site);
    if (!site_entry) return false;
    family = ds4_family_repair_family(site_entry->family);
    if (!family) return false;
    dispatch->family = family->family;
    dispatch->production_repair_entry = family->production_repair_entry;
    if (!ds4_family_repair_enabled(enabled, family->family)) {
        dispatch->status = DS4_FAMILY_REPAIR_DISPATCH_DISABLED;
    } else if (family->status == DS4_FAMILY_REPAIR_NOT_IMPLEMENTED) {
        dispatch->status = DS4_FAMILY_REPAIR_DISPATCH_NOT_IMPLEMENTED;
    } else if (family->status == DS4_FAMILY_REPAIR_EXACT) {
        dispatch->status = DS4_FAMILY_REPAIR_DISPATCH_READY;
    } else {
        dispatch->status = DS4_FAMILY_REPAIR_DISPATCH_FAILED;
    }
    return true;
}

const char *ds4_family_repair_status_name(ds4_family_repair_status status) {
    switch (status) {
    case DS4_FAMILY_REPAIR_NOT_IMPLEMENTED: return "NOT_IMPLEMENTED";
    case DS4_FAMILY_REPAIR_EXACT: return "EXACT";
    case DS4_FAMILY_REPAIR_FAILED: return "FAILED";
    }
    return "UNKNOWN";
}

const char *ds4_family_repair_dispatch_status_name(
        ds4_family_repair_dispatch_status status) {
    switch (status) {
    case DS4_FAMILY_REPAIR_DISPATCH_DISABLED: return "DISABLED";
    case DS4_FAMILY_REPAIR_DISPATCH_NOT_IMPLEMENTED: return "NOT_IMPLEMENTED";
    case DS4_FAMILY_REPAIR_DISPATCH_READY: return "READY";
    case DS4_FAMILY_REPAIR_DISPATCH_FAILED: return "FAILED";
    }
    return "UNKNOWN";
}
