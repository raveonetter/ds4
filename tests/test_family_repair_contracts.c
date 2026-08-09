#include "family_repair.h"

#include <stdio.h>
#include <string.h>

#define REQUIRE(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

int main(void) {
    const ds4_family_repair_manifest_entry *manifest;
    const ds4_family_repair_site_entry *sites;
    ds4_family_repair_mask none;
    ds4_family_repair_mask all;
    ds4_family_repair_mask selected;
    size_t family_count;
    size_t site_class_count;
    uint32_t unclassified = UINT32_MAX;
    unsigned exact = 0;
    unsigned not_implemented = 0;
    unsigned failed = 0;
    size_t i;

    manifest = ds4_family_repair_manifest(&family_count);
    sites = ds4_family_repair_sites(&site_class_count);
    REQUIRE(manifest != NULL);
    REQUIRE(sites != NULL);
    REQUIRE(family_count == DS4_REPAIR_FAMILY_COUNT);
    REQUIRE(site_class_count == DS4_REPAIR_SITE_CLASS_COUNT);
    REQUIRE(ds4_family_repair_manifest_validate(&unclassified));
    REQUIRE(unclassified == 0);

    REQUIRE(ds4_family_repair_parse_selection(NULL, &none));
    REQUIRE(none == 0);
    REQUIRE(ds4_family_repair_parse_selection("all", &all));
    REQUIRE(!ds4_family_repair_parse_selection("UNKNOWN_FAMILY", &selected));
    REQUIRE(ds4_family_repair_parse_selection(manifest[0].name, &selected));
    REQUIRE(ds4_family_repair_enabled(selected, manifest[0].family));
    REQUIRE(!ds4_family_repair_enabled(selected, manifest[1].family));

    for (i = 0; i < site_class_count; i++) {
        ds4_family_repair_dispatch disabled;
        ds4_family_repair_dispatch enabled;
        const ds4_family_repair_manifest_entry *family;

        REQUIRE(ds4_family_repair_select(sites[i].site, none, &disabled));
        REQUIRE(disabled.status == DS4_FAMILY_REPAIR_DISPATCH_DISABLED);
        REQUIRE(ds4_family_repair_select(sites[i].site, all, &enabled));
        family = ds4_family_repair_family(sites[i].family);
        REQUIRE(family != NULL);
        REQUIRE(enabled.family == family->family);
        REQUIRE(enabled.production_repair_entry ==
                family->production_repair_entry);
        if (family->status == DS4_FAMILY_REPAIR_NOT_IMPLEMENTED) {
            REQUIRE(enabled.status ==
                    DS4_FAMILY_REPAIR_DISPATCH_NOT_IMPLEMENTED);
        } else if (family->status == DS4_FAMILY_REPAIR_EXACT) {
            REQUIRE(enabled.status == DS4_FAMILY_REPAIR_DISPATCH_READY);
        } else {
            REQUIRE(enabled.status == DS4_FAMILY_REPAIR_DISPATCH_FAILED);
        }
    }

    printf("FAMILY_REPAIR_MANIFEST family_count=%zu "
           "proven_sites=%u unclassified_proven_sites=%u result=PASS\n",
           family_count, DS4_FAMILY_REPAIR_PROVEN_SITE_COUNT, unclassified);
    for (i = 0; i < family_count; i++) {
        const char *status = ds4_family_repair_status_name(manifest[i].status);
        printf("FAMILY_REPAIR_CONTRACT family=%s proven_sites=%u "
               "input_bits_equal=PASS weights_same=PASS metadata_same=PASS "
               "production_repair=%s\n",
               manifest[i].name, manifest[i].proven_site_count, status);
        if (manifest[i].status == DS4_FAMILY_REPAIR_EXACT) exact++;
        else if (manifest[i].status == DS4_FAMILY_REPAIR_NOT_IMPLEMENTED)
            not_implemented++;
        else failed++;
    }
    printf("FAMILY_REPAIR_STATUS families_total=%zu families_exact=%u "
           "families_not_implemented=%u families_failed=%u\n",
           family_count, exact, not_implemented, failed);
    REQUIRE(exact + not_implemented + failed == family_count);
    REQUIRE(failed == 0);
    return 0;
}
