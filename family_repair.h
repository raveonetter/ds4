#ifndef DS4_FAMILY_REPAIR_H
#define DS4_FAMILY_REPAIR_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define DS4_FAMILY_REPAIR_PROVEN_SITE_COUNT 177u

typedef enum {
    DS4_REPAIR_FAMILY_Q8_0_BATCH_EXT_VS_SINGLE_MV = 0,
    DS4_REPAIR_FAMILY_FLASH_ATTN_BATCH_DIRECT_VS_SINGLE_VEC_REDUCE,
    DS4_REPAIR_FAMILY_F16_BATCH_EXT_VS_SINGLE_MV,
    DS4_REPAIR_FAMILY_ROUTER_WEIGHT_NORMALIZATION_BATCH_REDUCE_VS_SINGLE_KERNEL,
    DS4_REPAIR_FAMILY_ROUTED_MOE_IQ2_XXS_Q2_K_BATCH_VS_SINGLE,
    DS4_REPAIR_FAMILY_Q8_SHARED_DOWN_BATCH_F32_HC_ADD_VS_SINGLE_FUSED_HC,
    DS4_REPAIR_FAMILY_F16_BATCH_EXT_VS_SINGLE_PAIR_MV,
    DS4_REPAIR_FAMILY_COUNT
} ds4_repair_family;

typedef enum {
    DS4_REPAIR_SITE_QA = 0,
    DS4_REPAIR_SITE_KV,
    DS4_REPAIR_SITE_QB,
    DS4_REPAIR_SITE_CP4_OUTPUT_B,
    DS4_REPAIR_SITE_SHARED_GATE_UP,
    DS4_REPAIR_SITE_ATTENTION_HEADS,
    DS4_REPAIR_SITE_HC_ATTN_PRE_SPLIT,
    DS4_REPAIR_SITE_HC_FFN_PRE,
    DS4_REPAIR_SITE_FFN_ROUTER_PROJECTION,
    DS4_REPAIR_SITE_FFN_ROUTER_WEIGHTS,
    DS4_REPAIR_SITE_ROUTED_MOE,
    DS4_REPAIR_SITE_CP5_TAIL,
    DS4_REPAIR_SITE_ATTN_COMPRESSOR_KV,
    DS4_REPAIR_SITE_ATTN_COMPRESSOR_SCORE,
    DS4_REPAIR_SITE_INDEXER_KV,
    DS4_REPAIR_SITE_INDEXER_SCORE,
    DS4_REPAIR_SITE_CLASS_COUNT
} ds4_repair_site;

typedef enum {
    DS4_FAMILY_REPAIR_NOT_IMPLEMENTED = 0,
    DS4_FAMILY_REPAIR_EXACT,
    DS4_FAMILY_REPAIR_FAILED
} ds4_family_repair_status;

typedef enum {
    DS4_FAMILY_REPAIR_ENTRY_Q8_0_BATCH_EXT = 0,
    DS4_FAMILY_REPAIR_ENTRY_FLASH_ATTN_BATCH_DIRECT,
    DS4_FAMILY_REPAIR_ENTRY_F16_BATCH_EXT,
    DS4_FAMILY_REPAIR_ENTRY_ROUTER_WEIGHT_NORMALIZATION,
    DS4_FAMILY_REPAIR_ENTRY_ROUTED_MOE_IQ2_XXS_Q2_K,
    DS4_FAMILY_REPAIR_ENTRY_Q8_SHARED_DOWN_BATCH_F32_HC_ADD,
    DS4_FAMILY_REPAIR_ENTRY_F16_BATCH_EXT_PAIR,
    DS4_FAMILY_REPAIR_ENTRY_COUNT
} ds4_family_repair_entry;

typedef uint32_t ds4_family_repair_mask;

typedef struct {
    ds4_repair_family family;
    const char *name;
    const char *proven_sites;
    uint32_t proven_site_count;
    const char *generic_topology;
    const char *canonical_reference;
    ds4_family_repair_entry production_repair_entry;
    const char *production_repair_entry_name;
    ds4_family_repair_status status;
} ds4_family_repair_manifest_entry;

typedef struct {
    ds4_repair_site site;
    const char *name;
    ds4_repair_family family;
    uint32_t runtime_sites;
} ds4_family_repair_site_entry;

typedef enum {
    DS4_FAMILY_REPAIR_DISPATCH_DISABLED = 0,
    DS4_FAMILY_REPAIR_DISPATCH_NOT_IMPLEMENTED,
    DS4_FAMILY_REPAIR_DISPATCH_READY,
    DS4_FAMILY_REPAIR_DISPATCH_FAILED
} ds4_family_repair_dispatch_status;

typedef struct {
    ds4_repair_family family;
    ds4_family_repair_entry production_repair_entry;
    ds4_family_repair_dispatch_status status;
} ds4_family_repair_dispatch;

const ds4_family_repair_manifest_entry *ds4_family_repair_manifest(
        size_t *count);
const ds4_family_repair_site_entry *ds4_family_repair_sites(size_t *count);
const ds4_family_repair_manifest_entry *ds4_family_repair_family(
        ds4_repair_family family);
const ds4_family_repair_site_entry *ds4_family_repair_site(
        ds4_repair_site site);

bool ds4_family_repair_manifest_validate(uint32_t *unclassified_proven_sites);
bool ds4_family_repair_parse_selection(const char *value,
                                       ds4_family_repair_mask *mask);
bool ds4_family_repair_enabled(ds4_family_repair_mask mask,
                               ds4_repair_family family);
bool ds4_family_repair_runtime_enabled(ds4_repair_family family);
bool ds4_family_repair_select(ds4_repair_site site,
                              ds4_family_repair_mask enabled,
                              ds4_family_repair_dispatch *dispatch);

const char *ds4_family_repair_status_name(ds4_family_repair_status status);
const char *ds4_family_repair_dispatch_status_name(
        ds4_family_repair_dispatch_status status);

#endif
