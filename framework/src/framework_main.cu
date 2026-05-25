// framework_main.cu -- Host-side launcher for the GPU-resident SAG
// solver introduced in iter300, with iter310 on-GPU merge, iter320
// multi-wave layer processing, iter330 sparse F-pair encoding, and
// iter340 in-kernel streaming compaction with cross-wave global merge.
// iter360 adds DRAM spill: when the post-merge layer exceeds layer_capacity,
// excess states overflow into host-pinned memory; on the next layer's input,
// spill is paged back in chunks via a small d_paged_input GPU buffer.
// iter370 redesigns the spill so each state touches PCIe AT MOST ONCE per
// layer: the mid-layer concat-and-re-merge of iter360 is replaced by a
// non-merging scratch spill ring (D2H copy + reset) plus ONE final merge
// at layer end that pages all chunks back via d_aggregator. No more O(K^2)
// data movement when scratch overflows multiple times.
// iter380 adds K-way pairwise merge for the layer-end consolidate path:
// when even after per-chunk pre-sort the combined chunk total still exceeds
// (d_aggregator + d_layer_scratch) capacity, perform log2(K) rounds of
// pairwise merging that combine adjacent host-spill chunks two at a time.
// Each pair-merge loads two chunks into d_aggregator + d_layer_scratch,
// runs the existing dual-source sort+merge, and writes the dedup'd output
// back to the host-spill ring as a single, smaller chunk. This bounds the
// peak in-flight VRAM to one pair (~2 chunks ~= aggregator) plus d_next
// as the merge target, fitting in the contended 48 GB envelope. After all
// pair rounds the chunk count and combined total are small enough for the
// existing single-pass consolidate.
//
// Architecture (iter370):
//   - Outer host loop drives one layer per iteration.
//   - INNER WAVE LOOP per layer: chunks larger-than-VRAM layers into waves
//     of cap.wave_states states. Per-wave merge writes into a per-layer
//     scratch accumulator d_layer_scratch (NOT into d_next yet).
//     * If d_layer_scratch is full enough that the NEXT wave's worst-case
//       output would overflow: spill the entire d_layer_scratch tail to a
//       host-pinned ring buffer (h_scratch_spill[chunk_idx]). Reset
//       total_pre_cw_merged=0. Continue. NO MERGE at this point.
//   - After all waves complete:
//     If no scratch spill happened: existing single-source layer-end merge
//       (sort d_layer_scratch by D-key, detect groups, V5MergeKernel into
//       d_next; spill overflow to h_layer_spill_next as needed).
//     Otherwise: page all spilled scratch chunks back into d_aggregator
//       (one big GPU buffer), append the residual scratch, run ONE global
//       sort + group + merge pass over the whole combined input. Each
//       state crosses PCIe at most once out + once back per layer.
//   - After layer-end merge: layer ping-pong (swap d_curr <-> d_next).
//
// Memory budget (iter340):
//   - 2x persistent layer buffers (d_layerA / d_layerB) sized to
//     layer_capacity * bps   ~25% of usable VRAM each
//   - 1x cross-wave scratch (d_layer_scratch)
//                            sized to scratch_capacity * bps   ~40% of VRAM
//   - 1x per-wave K2 output (d_wave_output)
//                            sized to wave_max_output * bps    ~10% of VRAM
//   - per-wave merge scratch (dkeys/idx/etc), CUB temp         remainder
//
// Sizing chain (in order, each constrained by what's left over):
//   * wave_states from per-wave K2 output cap (iter112 ~12% rule)
//   * scratch_capacity = (post-budget-deduction VRAM) * 0.50 / bps
//   * layer_capacity   = (post-budget-deduction VRAM - scratch) / (2*bps)
//
// FAIL-LOUD discipline (iter340):
//   Every overflow path printf's the exact context (layer, total, cap) and
//   sets V5Status::TRUNC. NEVER silently drops a valid successor.
//
// Generic A/H/B-series; no Hopper-only features. Cooperative launch +
// this_grid().sync() are SM_70+. CUB host-side dispatch (no CDP).
//
// Usage:
//   expand_test_v5 [-m cores] jobs.csv jobsprec.csv

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <sys/mman.h>             // mmap, munmap, MAP_HUGETLB
#include <unistd.h>               // sysconf

// MAP_HUGE_2MB lives in linux/mman.h on glibc; sys/mman.h doesn't expose it.
// Define it manually if missing (encoded log2(2 MB) = 21 in bits 26-31).
#ifndef MAP_HUGE_2MB
#define MAP_HUGE_2MB (21 << 26)
#endif

#include <cuda_runtime.h>
#include <cooperative_groups.h>

// CUB for radix sort + merge sort + exclusive scan
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_merge_sort.cuh>
#include <cub/device/device_scan.cuh>

#include "sag_config.h"
#include "sag_types.h"            // ValidPair (still needed for K1->K2 pipeline)
#include "sag_types_v5.h"         // SAGStateLayoutV5 -- sparse F-pair layout
#include "ijp_detect_v5.cuh"      // IJPInfoV5 -- IJP detection output struct
#include "v5_variant.h"           // V5Variant enum + parser + tricks table

using namespace sag;
using namespace sag::config;
using sag::v5::SAGStateLayoutV5;
using sag::v5::F_MAX_PER_STATE;
using sag::v5::ijp::IJPInfoV5;

// Forward decls for accessors / kernels exposed by controller.cu
extern "C" size_t      V5_smem_bytes(int n, int W, int m);
extern "C" size_t      V5_merge_smem_bytes();
extern "C" const void* V5_k1k2_kernel_ptr();
extern "C" const void* V5_k1k2_kernel_ijp_ptr();
extern "C" const void* V5_k1k2_kernel_rtss17_ptr();
// POR Phase B observability hooks (defined in controller.cu).
extern "C" bool V5_set_por_observe(int enabled);
extern "C" bool V5_get_por_dbg(int* out, int n_ints);
extern "C" bool V5_clear_por_dbg();
extern "C" const void* V5_k1k2_kernel_ecrts19_ptr();
extern "C" const void* V5_k1k2_kernel_ecrts22_ptr();
extern "C" const void* V5_merge_kernel_ptr();
extern "C" int         V5_block_size();
extern "C" int         V5_block_warps();
extern "C" int         V5_merge_block_size();
extern "C" int         V5_merge_warps_per_block();

extern "C" __global__ void V5ExtractDKeysIotaKernel(
    const char* d_states, SAGStateLayoutV5 layout,
    const int* d_num_states_ptr, int W,
    uint64_t* d_dkeys_out, int* d_idx_out);

extern "C" __global__ void V5ExtractDKeysIotaOffsetKernel(
    const char* d_states, SAGStateLayoutV5 layout,
    int num_states, int out_offset, int idx_offset, int W,
    uint64_t* d_dkeys_out, int* d_idx_out);

extern "C" __global__ void V5DetectBoundariesKernel(
    const uint64_t* d_dkeys, const int* d_sorted_idx,
    const int* d_num_states_ptr, int W,
    int* d_is_start);

extern "C" __global__ void V5CompactGroupStartsKernel(
    const int* d_is_start, const int* d_group_id,
    const int* d_num_states_ptr,
    int* d_group_starts, int* d_num_groups);

extern "C" __global__ void V5MergeKernel(
    const char* d_pre, const char* d_pre2, int pre1_count,
    const char* d_pre3, int pre2_end,
    const int* d_sorted_idx,
    const int* d_group_starts,
    int group_offset, int num_groups_in_batch, int num_groups_total,
    const int* d_num_states_ptr,
    SAGStateLayoutV5 layout, int n, int m, int W,
    char* d_post, int max_post_states,
    int* d_merge_count, int* d_trunc_flag, int layer_dbg);

// RTSS 2017 conservative-merge twin of V5MergeKernel (see controller.cu).
extern "C" __global__ void V5MergeKernelRTSS17(
    const char* d_pre, const char* d_pre2, int pre1_count,
    const char* d_pre3, int pre2_end,
    const int* d_sorted_idx,
    const int* d_group_starts,
    int group_offset, int num_groups_in_batch, int num_groups_total,
    const int* d_num_states_ptr,
    SAGStateLayoutV5 layout, int n, int m, int W,
    char* d_post, int max_post_states,
    int* d_merge_count, int* d_trunc_flag, int layer_dbg);

// ECRTS 2019 segment-deadline merge twin.
extern "C" __global__ void V5MergeKernelECRTS19(
    const char* d_pre, const char* d_pre2, int pre1_count,
    const char* d_pre3, int pre2_end,
    const int* d_sorted_idx,
    const int* d_group_starts,
    int group_offset, int num_groups_in_batch, int num_groups_total,
    const int* d_num_states_ptr,
    SAGStateLayoutV5 layout, int n, int m, int W,
    char* d_post, int max_post_states,
    int* d_merge_count, int* d_trunc_flag, int layer_dbg);

// ECRTS 2022 gang-aware merge twin.
extern "C" __global__ void V5MergeKernelECRTS22(
    const char* d_pre, const char* d_pre2, int pre1_count,
    const char* d_pre3, int pre2_end,
    const int* d_sorted_idx,
    const int* d_group_starts,
    int group_offset, int num_groups_in_batch, int num_groups_total,
    const int* d_num_states_ptr,
    SAGStateLayoutV5 layout, int n, int m, int W,
    char* d_post, int max_post_states,
    int* d_merge_count, int* d_trunc_flag, int layer_dbg);

// Variant-aware merge launcher.
//
// Each variant's wrapper kernel takes the same arg list ending with
// `..., d_post, max_post_states, d_merge_count, d_trunc_flag, layer_dbg`.
// max_post_states == 0 (or negative) disables the bounds check (legacy
// behaviour); positive values set d_trunc_flag if slot_idx >= max_post_states.
#define V5_LAUNCH_MERGE(GRID, BLOCK, SMEM, ...) \
    do { \
        if (g_variant == sag::v5::V5Variant::NP_UNI_17) { \
            V5MergeKernelRTSS17<<<(GRID), (BLOCK), (SMEM)>>>(__VA_ARGS__); \
        } else if (g_variant == sag::v5::V5Variant::LP_DAG_19) { \
            V5MergeKernelECRTS19<<<(GRID), (BLOCK), (SMEM)>>>(__VA_ARGS__); \
        } else if (g_variant == sag::v5::V5Variant::GANG_22 && \
                   !g_ecrts22_route_to_rtss24) { \
            V5MergeKernelECRTS22<<<(GRID), (BLOCK), (SMEM)>>>(__VA_ARGS__); \
        } else { \
            /* rtss24/ecrts22-routed-to-rtss24 */ \
            V5MergeKernel<<<(GRID), (BLOCK), (SMEM)>>>(__VA_ARGS__); \
        } \
    } while (0)

// varwidth-spill kernels (defined in controller.cu).
extern "C" __global__ void V5ChunkMaxFCountKernel(
    const char* d_states, SAGStateLayoutV5 layout,
    int count, int* d_max_out);
extern "C" __global__ void V5PackForSpillKernel(
    const char* d_in, SAGStateLayoutV5 in_layout,
    char* d_out, int packed_bps, int max_F_used,
    int count, int W, int m);
extern "C" __global__ void V5UnpackFromSpillKernel(
    const char* d_in, int packed_bps, int max_F_used,
    char* d_out, SAGStateLayoutV5 out_layout,
    int count, int W, int m);
extern "C" int V5_layout_header_bytes(int W, int m);
extern "C" int V5_packed_bps(int W, int m, int max_F_used);

enum class V5Status : int32_t {
    UNKNOWN  = 0,
    SCHED    = 1,
    UNSCHED  = 2,
    TRUNC    = 3,
    EMPTY    = 4,
    MAX_LAYER_EXCEEDED = 5
};

#define V5_CUDA(call) do {                                                    \
    cudaError_t _e = (call);                                                  \
    if (_e != cudaSuccess) {                                                  \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                           \
                __FILE__, __LINE__, cudaGetErrorString(_e));                  \
        std::exit(1);                                                         \
    }                                                                         \
} while(0)

// ---------------------------------------------------------------------------
// V5 mmap-pinned host alloc (port of legacy K8 mh_alloc/mh_free + HugeTLB).
//
// Replaces cudaHostAlloc(...,cudaHostAllocPortable) for the V5 spill ring
// allocations. cudaHostAlloc on Linux internally page-locks bytes one-by-one
// at a few GB/s. mmap+cudaHostRegister is the standard alternative and is
// typically 1.2-1.5x faster for multi-GB pinned allocations.
//
// HugeTLB (MAP_HUGETLB | MAP_HUGE_2MB): for >= 64 MB allocations we first try
// 2 MB pages, which gives ~512x fewer TLB entries and noticeably faster
// page-locking. If HugeTLB pages are not reserved by the kernel
// (vm.nr_hugepages=0), the mmap fails and we transparently fall back to
// regular 4 KB pages -- still faster than cudaHostAlloc, no harm done.
//
// We track the rounded allocation size in a small map so the free path can
// pass the same size to munmap. Same per-buffer semantics as cudaHostAlloc /
// cudaFreeHost; only the function names change at the callsite.
// ---------------------------------------------------------------------------
static size_t v5_mh_page_size() {
    static size_t pg = 0;
    if (pg == 0) {
        long v = sysconf(_SC_PAGESIZE);
        pg = (v > 0) ? (size_t)v : 4096;
    }
    return pg;
}
static std::unordered_map<void*, size_t> g_v5_mh_size;
static std::mutex g_v5_mh_mutex;
static cudaError_t v5_mh_alloc(void** out, size_t bytes) {
    if (bytes == 0) { *out = nullptr; return cudaSuccess; }
    size_t pg = v5_mh_page_size();
    size_t aligned = (bytes + pg - 1) & ~(pg - 1);

    // Try HugeTLB 2 MB pages first for multi-GB allocations (better TLB).
    void* p = MAP_FAILED;
    if (aligned >= 64L * 1024L * 1024L) {  // only worth it for >= 64 MB
        const size_t hp = (1L << 21);  // 2 MB
        size_t huge_aligned = (aligned + hp - 1) & ~(hp - 1);
#if defined(MAP_HUGETLB) && defined(MAP_HUGE_2MB)
        p = mmap(nullptr, huge_aligned, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_2MB,
                 -1, 0);
        if (p != MAP_FAILED) aligned = huge_aligned;
#endif
    }
    if (p == MAP_FAILED) {
        // Fall back to regular pages: HugeTLB might not be reserved.
        p = mmap(nullptr, aligned, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) return cudaErrorMemoryAllocation;
    }
    cudaError_t e = cudaHostRegister(p, aligned, cudaHostRegisterDefault);
    if (e != cudaSuccess) { munmap(p, aligned); return e; }
    {
        std::lock_guard<std::mutex> lk(g_v5_mh_mutex);
        g_v5_mh_size[p] = aligned;
    }
    *out = p;
    return cudaSuccess;
}
static cudaError_t v5_mh_free(void* p) {
    if (!p) return cudaSuccess;
    size_t aligned = 0;
    {
        std::lock_guard<std::mutex> lk(g_v5_mh_mutex);
        auto it = g_v5_mh_size.find(p);
        if (it == g_v5_mh_size.end()) {
            // Fallback: was allocated by cudaHostAlloc somewhere else.
            return cudaFreeHost(p);
        }
        aligned = it->second;
        g_v5_mh_size.erase(it);
    }
    cudaError_t e = cudaHostUnregister(p);
    if (e != cudaSuccess) return e;
    if (munmap(p, aligned) != 0) return cudaErrorUnknown;
    return cudaSuccess;
}

// ---------------------------------------------------------------------------
// Host-RAM safety cutoff (read /proc/meminfo every layer, fail loud at limit)
// ---------------------------------------------------------------------------
//
// User feedback (feedback_ram_safety_cutoffs.md): the V5 binary itself MUST
// self-monitor host RAM and fail loud BEFORE host-OOM. Tier-2 host-pinned
// spill is bounded by SAG_V5_HOST_SPILL_GB, but other processes (aminian1,
// other users) may starve the system mid-run. Per-layer self-check.
//
// Threshold: env var SAG_HOST_RAM_LIMIT_PCT (default 90.0).
// On trigger: prints `[V5 HOST-RAM TRUNC]` and returns false; caller sets
// d_trunc=1 and the run reports `Schedulable: TRUNCATED`.

static double v5_host_ram_used_pct() {
    long long mem_total_kb = 0;
    long long mem_avail_kb = 0;
    FILE* fmem = std::fopen("/proc/meminfo", "r");
    if (!fmem) return -1.0;
    char line[256];
    while (std::fgets(line, sizeof(line), fmem)) {
        long long kb = 0;
        if (std::sscanf(line, "MemTotal: %lld kB", &kb) == 1) {
            mem_total_kb = kb;
        } else if (std::sscanf(line, "MemAvailable: %lld kB", &kb) == 1) {
            mem_avail_kb = kb;
        }
        if (mem_total_kb > 0 && mem_avail_kb > 0) break;
    }
    std::fclose(fmem);
    if (mem_total_kb <= 0) return -1.0;
    if (mem_avail_kb < 0) mem_avail_kb = 0;
    if (mem_avail_kb > mem_total_kb) mem_avail_kb = mem_total_kb;
    long long used_kb = mem_total_kb - mem_avail_kb;
    return 100.0 * (double)used_kb / (double)mem_total_kb;
}

// Returns true if RAM usage is safe (below threshold OR /proc/meminfo missing).
// Returns false (and prints fail-loud message) if usage exceeds threshold.
static bool v5_check_host_ram_or_fail(int layer_dbg) {
    static double cached_threshold = -1.0;
    if (cached_threshold < 0.0) {
        cached_threshold = 90.0;
        const char* p = std::getenv("SAG_HOST_RAM_LIMIT_PCT");
        if (p && p[0] != '\0') {
            double v = std::atof(p);
            if (v > 0.0 && v <= 100.0) cached_threshold = v;
        }
    }
    double pct = v5_host_ram_used_pct();
    if (pct < 0.0) return true;  // /proc/meminfo unavailable; don't gate.
    if (pct >= cached_threshold) {
        std::fprintf(stderr,
            "[V5 HOST-RAM TRUNC] layer=%d host RAM usage %.2f%% >= "
            "SAG_HOST_RAM_LIMIT_PCT %.2f%% -- aborting run BEFORE OOM. "
            "Reduce SAG_V5_HOST_SPILL_GB or free other processes' RAM.\n",
            layer_dbg, pct, cached_threshold);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// CSV parsers (lifted from legacy expand_test_main.cu)
// ---------------------------------------------------------------------------

struct V5JobRecord {
    int job_id, instance_id;
    int r_min, r_max, C_min, C_max, deadline, priority;
    // Phase 3 (ECRTS 2022) gang fields. Default 1 (sequential job, fits the
    // standard non-gang variants without affecting behaviour).
    int p_min;       // minimum parallelism level
    int p_max;       // maximum parallelism level
    // Phase 3.3 moldable gang full cost map. Each entry = (p, c_min, c_max).
    // For rigid gang this has one entry == (p_max, C_min, C_max). For non-
    // gang (sequential) it can be empty -- consumers default to (1, C_min,
    // C_max). For moldable, all (p, c_min, c_max) tuples from `{p:cmin:cmax;
    // p:cmin:cmax;...}` are stored in CSV order.
    std::vector<std::tuple<int,int,int>> moldable_costs;
};

// Forward decl for gang-format parser branch (Phase 3.3 moldable extension).
static bool v5_parse_gang_cost_field(
    const char* field_start, int* p_min, int* p_max,
    int* C_min, int* C_max,
    std::vector<std::tuple<int,int,int>>* out_costs = nullptr);

// Variant-aware CSV parser. Auto-detects per-row whether column 5 is a gang
// cost map `{p:cmin:cmax;...}` or the standard pair of int columns
// `C_min, C_max`. This means a single jobs.csv can mix sequential (p=1) and
// gang rows -- a property the upstream nptest also relies on -- and the
// `gang_format` hint is informational.
//
//   Non-gang row : Task_ID, Job_ID, r_min, r_max, C_min, C_max, deadline, prio
//   Gang row     : Task_ID, Job_ID, r_min, r_max, {p:cmin:cmax;...}, deadline, prio
//
// The gang parser fills C_min/C_max with the SINGLE-p cost (rigid gang); for
// moldable gang the additional (p, cmin, cmax) entries must be read into a
// secondary cost map (Phase 3 follow-up; rigid is the MVP).
static bool v5_parse_jobs_csv(const char* path,
                              std::vector<V5JobRecord>& out_records,
                              int& out_max_iid,
                              bool gang_format)
{
    (void)gang_format;  // auto-detected per row; param kept for API stability.
    FILE* fj = std::fopen(path, "r");
    if (!fj) return false;
    int max_iid = -1;
    char line[512];
    while (std::fgets(line, sizeof(line), fj)) {
        if (line[0] == '\n' || line[0] == '\r' || line[0] == '\0') continue;

        V5JobRecord rec;
        rec.p_min = 1;
        rec.p_max = 1;

        // Locate the start of column 5 to decide gang vs sequential. Skip
        // first 4 commas.
        const char* col5 = line;
        int commas_seen = 0;
        while (*col5 && commas_seen < 4) {
            if (*col5 == ',') commas_seen++;
            col5++;
        }
        const char* col5_skipws = col5;
        while (*col5_skipws == ' ' || *col5_skipws == '\t') col5_skipws++;

        if (*col5_skipws == '{') {
            // Gang row.
            int tid_jid_count = std::sscanf(line, "%d,%d,%d,%d",
                &rec.job_id, &rec.instance_id, &rec.r_min, &rec.r_max);
            if (tid_jid_count != 4) continue;

            const char* close = std::strchr(col5_skipws, '}');
            if (!close) continue;

            if (!v5_parse_gang_cost_field(col5_skipws, &rec.p_min, &rec.p_max,
                                          &rec.C_min, &rec.C_max,
                                          &rec.moldable_costs)) {
                continue;
            }

            const char* tail = close + 1;
            while (*tail == ' ' || *tail == ',' || *tail == '\t') tail++;
            int parsed_dl_prio = std::sscanf(tail, "%d,%d",
                &rec.deadline, &rec.priority);
            if (parsed_dl_prio != 2) continue;

            if (rec.p_min < 1) rec.p_min = 1;
            if (rec.p_max < rec.p_min) rec.p_max = rec.p_min;
        } else {
            // Sequential row (standard 8-column format).
            int parsed = std::sscanf(line, "%d,%d,%d,%d,%d,%d,%d,%d",
                &rec.job_id, &rec.instance_id,
                &rec.r_min, &rec.r_max, &rec.C_min, &rec.C_max,
                &rec.deadline, &rec.priority);
            if (parsed != 8) continue;
        }

        out_records.push_back(rec);
        if (rec.instance_id > max_iid) max_iid = rec.instance_id;
    }
    std::fclose(fj);
    out_max_iid = max_iid;
    return true;
}

// Parse `{p:cmin:cmax;p:cmin:cmax;...}`. Stores all entries in `*out_costs`
// (Phase 3.3 moldable cost map). For backward compatibility, *C_min /
// *C_max are also populated with the cost of the largest-p entry (so the
// sequential K1+K2 path that ignores the moldable list still works).
// For rigid gang p_min == p_max == p (and out_costs has one entry).
static bool v5_parse_gang_cost_field(const char* field_start,
                                     int* p_min, int* p_max,
                                     int* C_min, int* C_max,
                                     std::vector<std::tuple<int,int,int>>* out_costs)
{
    if (*field_start != '{') return false;
    const char* p = field_start + 1;
    int min_p_seen = INT_MAX;
    int max_p_seen = -1;
    int smallest_p_cmin = 0, smallest_p_cmax = 0;
    int smallest_p = INT_MAX;
    bool any = false;
    if (out_costs) out_costs->clear();
    while (*p && *p != '}') {
        int paral = 0, cmin = 0, cmax = 0;
        int n = 0;
        if (std::sscanf(p, " %d : %d : %d %n", &paral, &cmin, &cmax, &n) != 3) {
            return false;
        }
        if (paral < min_p_seen) min_p_seen = paral;
        if (paral > max_p_seen) max_p_seen = paral;
        // Phase 3.3 moldable heuristic: default rigid-p uses the SMALLEST
        // parallelism. Rationale: the gang.cpp test cases nptest reports
        // expected F-times consistent with picking the smallest p (e.g.
        // Job 7 cost {1:15:17;3:6:7} -> nptest picks p=1 -> F=[15,17]).
        // Until full moldable greedy-earliest enumeration lands (Phase
        // 3.3.b), defaulting to smallest-p produces matching WCRT on the
        // gang.cpp tests. For RIGID gang (one entry) smallest_p == max_p,
        // so behaviour is unchanged.
        if (paral < smallest_p) {
            smallest_p = paral;
            smallest_p_cmin = cmin;
            smallest_p_cmax = cmax;
        }
        if (out_costs) out_costs->emplace_back(paral, cmin, cmax);
        any = true;
        p += n;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == ';') { p++; continue; }
        if (*p == '}') break;
        if (*p != '}') return false;
    }
    if (!any) return false;
    *p_min = min_p_seen;
    *p_max = (smallest_p == INT_MAX) ? max_p_seen : smallest_p;
    *C_min = smallest_p_cmin;
    *C_max = smallest_p_cmax;
    return true;
}

struct V5PrecEdge {
    int pred_iid, succ_iid;
    int sus_min, sus_max;
};

static bool v5_parse_prec_csv(const char* path,
                              std::vector<V5PrecEdge>& out_edges)
{
    FILE* fp = std::fopen(path, "r");
    if (!fp) return false;
    char line[512];
    while (std::fgets(line, sizeof(line), fp)) {
        if (line[0] == '\n' || line[0] == '\r' || line[0] == '\0') continue;
        int pred_jid, pred_iid, succ_jid, succ_iid, smin, smax;
        char type_ch;
        int parsed = std::sscanf(line, "%d,%d,%d,%d,%d,%d,%c",
            &pred_jid, &pred_iid, &succ_jid, &succ_iid,
            &smin, &smax, &type_ch);
        if (parsed < 6) continue;
        out_edges.push_back({pred_iid, succ_iid, smin, smax});
    }
    std::fclose(fp);
    return true;
}

// ---------------------------------------------------------------------------
// Topology preprocessing (mirrors legacy)
// ---------------------------------------------------------------------------

struct V5Topology {
    int N, W;
    std::vector<int32_t> r_min, r_max, C_min, C_max, deadline, priority;
    std::vector<int32_t> task_id;
    std::vector<uint64_t> Pred, Succ, TC, PO;
    std::vector<int32_t> sus_min, sus_max;
    std::vector<int>     candidates;
    std::vector<int32_t> prio_order;
    std::vector<int>     topo_depth;
    std::vector<int32_t> min_path_delay;
    // Phase 3 (ECRTS 2022) gang parallelism per job. Default 1 for non-gang.
    std::vector<int32_t> p_min, p_max;
    // Phase 3.3 moldable cost map in CSR form.
    std::vector<int32_t> costmap_offset;
    std::vector<int32_t> costmap_p;
    std::vector<int32_t> costmap_cmin;
    std::vector<int32_t> costmap_cmax;
    // Per-depth candidate truncation: depth_prefix[d] = number of
    // candidates whose topo_depth <= d (after candidates is sorted by
    // depth ascending). Per-layer L, the K1 inner loop only needs to
    // walk the first `depth_prefix[L + 1]` candidates -- jobs deeper
    // than L+1 are precedence-blocked and would be skipped anyway.
    // Size: max_depth + 2 (so depth_prefix[max_depth + 1] == N).
    std::vector<int>     depth_prefix;
    int                  max_depth;
};

static bool v5_build_topology(
    const std::vector<V5JobRecord>& records,
    int N, int W,
    const std::vector<V5PrecEdge>& edges,
    V5Topology& out)
{
    out.N = N; out.W = W;
    out.r_min.assign(N, 0); out.r_max.assign(N, 0);
    out.C_min.assign(N, 0); out.C_max.assign(N, 0);
    out.deadline.assign(N, 0); out.priority.assign(N, 0);
    out.task_id.assign(N, 0);
    out.p_min.assign(N, 1);  // default sequential (non-gang)
    out.p_max.assign(N, 1);

    // Build moldable cost-map storage indexed by instance_id.
    std::vector<std::vector<std::tuple<int,int,int>>> per_job_costs(N);

    for (const auto& rec : records) {
        int id = rec.instance_id;
        out.r_min[id]    = rec.r_min;
        out.r_max[id]    = rec.r_max;
        out.C_min[id]    = rec.C_min;
        out.C_max[id]    = rec.C_max;
        out.deadline[id] = rec.deadline;
        out.priority[id] = rec.priority;
        out.task_id[id]  = rec.job_id;
        out.p_min[id]    = rec.p_min;
        out.p_max[id]    = rec.p_max;
        per_job_costs[id] = rec.moldable_costs;
    }

    // Flatten the moldable cost map into CSR form.
    out.costmap_offset.assign(N + 1, 0);
    int total_entries = 0;
    for (int id = 0; id < N; id++) {
        out.costmap_offset[id] = total_entries;
        // For a non-gang job with no moldable_costs, synthesize one entry from
        // C_min/C_max so the CSR is uniform.
        if (per_job_costs[id].empty()) {
            total_entries += 1;
        } else {
            total_entries += (int)per_job_costs[id].size();
        }
    }
    out.costmap_offset[N] = total_entries;
    out.costmap_p.assign(total_entries, 1);
    out.costmap_cmin.assign(total_entries, 0);
    out.costmap_cmax.assign(total_entries, 0);
    for (int id = 0; id < N; id++) {
        int off = out.costmap_offset[id];
        if (per_job_costs[id].empty()) {
            out.costmap_p[off]    = out.p_max[id];
            out.costmap_cmin[off] = out.C_min[id];
            out.costmap_cmax[off] = out.C_max[id];
        } else {
            int k = 0;
            for (const auto& t : per_job_costs[id]) {
                out.costmap_p[off + k]    = std::get<0>(t);
                out.costmap_cmin[off + k] = std::get<1>(t);
                out.costmap_cmax[off + k] = std::get<2>(t);
                k++;
            }
        }
    }

    out.Pred.assign((size_t)N * W, 0ULL);
    out.Succ.assign((size_t)N * W, 0ULL);
    out.sus_min.assign((size_t)N * N, 0);
    out.sus_max.assign((size_t)N * N, 0);

    for (const auto& e : edges) {
        out.Pred[(size_t)e.succ_iid * W + (e.pred_iid / 64)] |= (1ULL << (e.pred_iid % 64));
        out.Succ[(size_t)e.pred_iid * W + (e.succ_iid / 64)] |= (1ULL << (e.succ_iid % 64));
        out.sus_min[(size_t)e.succ_iid * N + e.pred_iid] = e.sus_min;
        out.sus_max[(size_t)e.succ_iid * N + e.pred_iid] = e.sus_max;
    }

    // TC = transitive closure of Pred
    out.TC.assign((size_t)N * W, 0ULL);
    for (int j = 0; j < N; j++)
        for (int w = 0; w < W; w++)
            out.TC[(size_t)j * W + w] = out.Pred[(size_t)j * W + w];
    for (int it = 0; it < N; it++) {
        bool changed = false;
        for (int j = 0; j < N; j++) {
            for (int pw = 0; pw < W; pw++) {
                uint64_t pbits = out.TC[(size_t)j * W + pw];
                while (pbits) {
                    int bit = __builtin_ctzll(pbits);
                    pbits &= pbits - 1;
                    int p = pw * 64 + bit;
                    for (int w = 0; w < W; w++) {
                        uint64_t old = out.TC[(size_t)j * W + w];
                        out.TC[(size_t)j * W + w] |= out.TC[(size_t)p * W + w];
                        if (out.TC[(size_t)j * W + w] != old) changed = true;
                    }
                }
            }
        }
        if (!changed) break;
    }

    // Topo depth
    out.topo_depth.assign(N, 0);
    {
        bool changed = true;
        while (changed) {
            changed = false;
            for (int j = 0; j < N; j++) {
                for (int w = 0; w < W; w++) {
                    uint64_t pbits = out.Pred[(size_t)j * W + w];
                    while (pbits) {
                        int bit = __builtin_ctzll(pbits);
                        pbits &= pbits - 1;
                        int pred_id = w * 64 + bit;
                        if (out.topo_depth[pred_id] + 1 > out.topo_depth[j]) {
                            out.topo_depth[j] = out.topo_depth[pred_id] + 1;
                            changed = true;
                        }
                    }
                }
            }
        }
    }

    // PO guard
    out.PO.assign((size_t)N * W, 0ULL);
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            if (i == j) continue;
            if (out.priority[i] >= out.priority[j]) continue;
            if (out.r_max[i] > out.r_min[j]) continue;
            bool pred_subset = true;
            for (int w = 0; w < W; w++) {
                if (out.Pred[(size_t)i * W + w] & ~out.Pred[(size_t)j * W + w]) {
                    pred_subset = false; break;
                }
            }
            if (!pred_subset) continue;
            bool sus_ok = true;
            for (int pw = 0; pw < W && sus_ok; pw++) {
                uint64_t pbits = out.Pred[(size_t)i * W + pw];
                while (pbits) {
                    int bit = __builtin_ctzll(pbits);
                    pbits &= pbits - 1;
                    int p = pw * 64 + bit;
                    if (out.sus_max[(size_t)i * N + p] > out.sus_min[(size_t)j * N + p]) {
                        sus_ok = false; break;
                    }
                }
            }
            if (!sus_ok) continue;
            out.PO[(size_t)j * W + (i / 64)] |= (1ULL << (i % 64));
        }
    }
    for (int it = 0; it < N; it++) {
        bool changed = false;
        for (int j = 0; j < N; j++) {
            for (int pw = 0; pw < W; pw++) {
                uint64_t pobits = out.PO[(size_t)j * W + pw];
                while (pobits) {
                    int bit = __builtin_ctzll(pobits);
                    pobits &= pobits - 1;
                    int dom = pw * 64 + bit;
                    for (int w = 0; w < W; w++) {
                        uint64_t old = out.PO[(size_t)j * W + w];
                        out.PO[(size_t)j * W + w] |= out.PO[(size_t)dom * W + w];
                        if (out.PO[(size_t)j * W + w] != old) changed = true;
                    }
                }
            }
        }
        if (!changed) break;
    }

    out.candidates.resize(N);
    for (int i = 0; i < N; i++) out.candidates[i] = i;
    std::sort(out.candidates.begin(), out.candidates.end(),
        [&](int a, int b) {
            if (out.topo_depth[a] != out.topo_depth[b])
                return out.topo_depth[a] < out.topo_depth[b];
            return out.priority[a] < out.priority[b];
        });

    // Phase 5.x: per-depth prefix counts so K1 launches can truncate the
    // candidate scan to "depth <= layer + 1". The candidates array is now
    // sorted by depth ascending, so depth_prefix[d] is the index of the
    // first candidate whose depth > d. We compute up to max_depth + 2 so
    // depth_prefix[max_depth + 1] == N (all candidates included).
    int max_depth_seen = 0;
    for (int d : out.topo_depth) if (d > max_depth_seen) max_depth_seen = d;
    out.max_depth = max_depth_seen;
    out.depth_prefix.assign(max_depth_seen + 2, 0);
    for (size_t i = 0; i < out.candidates.size(); i++) {
        int d = out.topo_depth[out.candidates[i]];
        // depth_prefix[d+1] gets one more for this candidate
        for (int dd = d + 1; dd <= max_depth_seen + 1; dd++) {
            out.depth_prefix[dd]++;
        }
    }

    out.min_path_delay.assign(N, 0);
    for (int j = 0; j < N; j++) out.min_path_delay[j] = out.r_min[j];
    {
        bool changed = true;
        while (changed) {
            changed = false;
            for (int j = 0; j < N; j++) {
                for (int w = 0; w < W; w++) {
                    uint64_t pbits = out.Pred[(size_t)j * W + w];
                    while (pbits) {
                        int bit = __builtin_ctzll(pbits);
                        pbits &= pbits - 1;
                        int pred_id = w * 64 + bit;
                        int32_t via_pred = out.min_path_delay[pred_id]
                                         + out.C_min[pred_id]
                                         + out.sus_min[(size_t)j * N + pred_id];
                        if (via_pred > out.min_path_delay[j]) {
                            out.min_path_delay[j] = via_pred;
                            changed = true;
                        }
                    }
                }
            }
        }
    }

    out.candidates.erase(
        std::remove_if(out.candidates.begin(), out.candidates.end(),
            [&](int j) {
                return out.min_path_delay[j] > out.deadline[j] - out.C_min[j];
            }),
        out.candidates.end());

    out.prio_order.assign(N, 0);
    for (int i = 0; i < N; i++) out.prio_order[i] = i;
    std::sort(out.prio_order.begin(), out.prio_order.end(),
        [&](int a, int b) { return out.priority[a] < out.priority[b]; });

    return true;
}

// ---------------------------------------------------------------------------
// VRAM-fraction-based capacity (iter112) -- multi-wave + cross-wave (iter340)
// ---------------------------------------------------------------------------
//
// iter310 was single-wave: every layer's input had to fit in one cooperative
// kernel launch and one merge launch, with layer_capacity == max wave size
// (~7M states). On hard workloads (n28/t000) layer L has ~4.6M parents and
// K2 expansion of ~7x produces 31M+ pre-merge successors -- the single-wave
// scheme TRUNCATEd at layer 89.
//
// iter320 split per-layer work into multiple waves. iter340 adds a cross-wave
// global merge pass (the per-wave-only scheme of iter320 left duplicate D-keys
// across waves, which inflated the next layer's input by ~3-7x and caused
// n28/t000 to overflow the layer cap at layer 90).
//
// iter340 budgets THREE layer-sized buffers:
//   - layerA, layerB: ping-pong post-cross-wave-merge layer accumulators
//                     (= the deduplicated layer L+1, fed into K1 next layer)
//   - layer_scratch: per-layer pre-cross-wave-merge accumulator
//                    (where per-wave merge results pile up across all waves
//                     of a single layer; cross-wave merge consumes this and
//                     emits the post-merge layer into d_next)
//
// Per-wave caps unchanged from iter320:
//   - wave_states: per-wave input slice (~80K states for n=140 on A100 80GB)
//   - wave_max_pairs / wave_max_output = wave_states * N
//
// Memory budget split (fractions of usable VRAM, after topology + slack):
//   - layer scratch:           ~50%   scratch_capacity * bps
//   - 2x layer buffers:        ~30%   2 * layer_capacity * bps
//   - per-wave K2 output buf:  ~10%   wave_max_output * bps
//   - per-wave aux + CUB:      remainder
//
// FAIL-LOUD: any time we hit the cap (scratch, layer, wave_output, F_count),
// we printf the location and set d_trunc. NEVER drop silently.
struct V5Capacity {
    // Layer accumulators.
    int   layer_capacity;     // post-cross-wave-merge ping-pong (d_layerA/B)
    int   scratch_capacity;   // pre-cross-wave-merge per-layer accumulator
    int   aggregator_capacity; // iter370: layer-end consolidate paging window

    // Per-wave caps.
    int   wave_states;        // input states per wave
    int   wave_max_pairs;     // valid_pairs cap per wave (= wave_states * n)
    int   wave_max_output;    // K2 output cap per wave    (= wave_states * n)

    long long usable_gpu_bytes;
    int   bytes_per_state;
};

static V5Capacity v5_compute_capacity(int n, int m, int W, int bps,
                                      bool spill_enabled) {
    (void)m;
    V5Capacity cap{};
    cap.bytes_per_state = bps;

    size_t gpu_free = 0, gpu_total = 0;
    V5_CUDA(cudaMemGetInfo(&gpu_free, &gpu_total));

    if (const char* env_cap = std::getenv("SAG_VRAM_LIMIT_MB")) {
        long long cap_bytes = std::atoll(env_cap) * 1024LL * 1024LL;
        if (cap_bytes > 0 && (size_t)cap_bytes < gpu_free) {
            gpu_free = (size_t)cap_bytes;
        }
    }

    double vram_pct = 1.0;
    if (const char* p = std::getenv("SAG_MAX_VRAM_PCT")) {
        double v = std::atof(p);
        if (v > 0.0 && v <= 1.0) vram_pct = v;
    }

    long long usable = (long long)((double)gpu_free * vram_pct);
    cap.usable_gpu_bytes = usable;

    long long topology_bytes = (long long)(4LL * n * W * sizeof(uint64_t)
        + 6LL * n * sizeof(int32_t)
        + 2LL * n * n * sizeof(int32_t)
        + (long long)n * sizeof(int32_t)
        + 1024);
    long long avail = (long long)(0.85 * usable) - topology_bytes;
    if (avail <= 0) avail = (long long)(0.5 * usable);

    // ------- Wave cap -------
    //
    // legacy iter112: cap K2 output buffer at ~12% of usable VRAM, which
    // equates to a wave_states bound of (0.12 * usable_gpu) / (n * bps).
    // Hard upper bound 32 GiB to avoid runaway allocations on data-center
    // GPUs (kernel compute, not buffer alloc, is the bottleneck above that).
    long long wave_output_cap_bytes = (long long)(usable * 0.12);
    const long long WAVE_OUTPUT_HARD_LIMIT = 32LL * 1024 * 1024 * 1024;
    if (wave_output_cap_bytes > WAVE_OUTPUT_HARD_LIMIT)
        wave_output_cap_bytes = WAVE_OUTPUT_HARD_LIMIT;

    long long per_state_wave_output = (long long)n * (long long)bps;
    long long wave_states = 1;
    if (per_state_wave_output > 0) {
        wave_states = wave_output_cap_bytes / per_state_wave_output;
    }
    if (wave_states < 1) wave_states = 1;
    // Sanity upper bound (legacy uses 1M).
    if (wave_states > 1000000) wave_states = 1000000;

    // iter370: SAG_V5_WAVE_STATES_OVERRIDE allows shrinking wave_states for
    // contended GPUs (smaller wave_output buffer, more room for aggregator
    // and scratch). Halving wave_states halves d_wave_output bytes.
    if (const char* p = std::getenv("SAG_V5_WAVE_STATES_OVERRIDE")) {
        long long v = std::atoll(p);
        if (v > 0 && v < wave_states) wave_states = v;
    }

    // wave_max_pairs (= wave_states * n) must fit int indexing in CUB.
    long long pairs_cap_int = (long long)INT_MAX / (long long)((n > 0) ? n : 1);
    if (wave_states > pairs_cap_int) wave_states = pairs_cap_int;
    cap.wave_states     = (int)wave_states;
    cap.wave_max_pairs  = (int)(wave_states * (long long)n);
    cap.wave_max_output = cap.wave_max_pairs;
    if (cap.wave_max_pairs  < 1) cap.wave_max_pairs  = 1;
    if (cap.wave_max_output < 1) cap.wave_max_output = 1;

    // ------- Layer + scratch caps (iter350) -------
    //
    // iter350 budgets FOUR layer-scale buffers:
    //   - d_layer_scratch: SMALL streaming buffer (~0.10 of avail_for_layer)
    //                      that accumulates per-wave merge results within a
    //                      single layer. When it would overflow on the next
    //                      wave's output, a mid-layer flush is triggered.
    //   - d_layerA, d_layerB: ping-pong layer buffers (d_curr, d_next) sized
    //                      to layer_capacity.
    //   - d_layer_temp: flush-merge destination (also sized to layer_capacity).
    //                   Each flush merges (scratch + partial d_next) into temp,
    //                   then swaps temp <-> d_next pointers. After the layer's
    //                   final wave + final flush, d_next has the merged layer.
    //
    // Mid-layer flush (the iter350 fix for n24/t000 + n28/t004 truncation):
    //   When per-wave outputs would overflow d_layer_scratch, flush:
    //     1. Sort the UNION (scratch + partial d_next) by D-key.
    //     2. Detect groups across the union.
    //     3. V5MergeKernel with DUAL-SOURCE input writes deduplicated layer
    //        into d_layer_temp.
    //     4. Swap d_next <-> d_layer_temp (d_next now holds new merged layer).
    //     5. Reset total_pre_cw_merged = 0.
    //
    // FAIL-LOUD: any path overflowing scratch / layer caps printf's the
    // exact state and sets V5Status::TRUNC. NEVER silently drop.

    long long fixed_wave_bytes =
          (long long)cap.wave_max_pairs  * (long long)sizeof(ValidPair)
        + (long long)cap.wave_max_output * bps                       // wave K2 output
        + (long long)cap.wave_max_output * W * (long long)sizeof(uint64_t) * 2  // dkeys_in/out at wave size (lower bound)
        + (long long)cap.wave_max_output * sizeof(int) * 5           // idx_in/out, is_start, group_id, group_starts at wave size
        + 1024LL * 1024LL * 256LL;                                   // CUB temp slack

    long long avail_for_layer = avail - fixed_wave_bytes;
    if (avail_for_layer < 0) avail_for_layer = (long long)(0.3 * avail);

    // iter370: when spill is enabled, reserve aggregator space. Default
    // aggregator fraction is 0.30 of avail_for_layer. User can override
    // via SAG_V5_AGGREGATOR_GB or SAG_V5_AGGREGATOR_FRAC. Larger
    // aggregator = bigger single-pass merge ceiling, smaller scratch +
    // layer caps. Tune for spill-heavy workloads.
    long long aggregator_bytes = 0;
    if (spill_enabled) {
        if (const char* p = std::getenv("SAG_V5_AGGREGATOR_GB")) {
            long long v_gb = std::atoll(p);
            if (v_gb > 0) aggregator_bytes = v_gb * 1024LL * 1024LL * 1024LL;
        }
        double aggregator_frac = 0.30;
        if (const char* p = std::getenv("SAG_V5_AGGREGATOR_FRAC")) {
            double v = std::atof(p);
            if (v > 0.0 && v < 1.0) aggregator_frac = v;
        }
        if (aggregator_bytes <= 0) {
            aggregator_bytes = (long long)(avail_for_layer * aggregator_frac);
            // Floor: at least 1 GB.
            long long floor_bytes = 1024LL * 1024 * 1024;
            if (aggregator_bytes < floor_bytes) aggregator_bytes = floor_bytes;
        }
        avail_for_layer -= aggregator_bytes;
        if (avail_for_layer < 0) avail_for_layer = 0;
    }

    // Layer-end merge sorts the largest single buffer it sees: scratch_cap
    // (no-spill path) or aggregator_cap (spill path). The merge scratch
    // buffers (dkeys/idx/is_start/group_id/group_starts) must be sized to
    // the maximum of these. Per-state cost:
    //   2 * W * 8 bytes (dkeys_in/out) + 5 * 4 bytes (idx_in/out, is_start,
    //   group_id, group_starts) = 16*W + 20 bytes.
    long long per_state_merge_aux =
          2LL * W * (long long)sizeof(uint64_t)   // dkeys_in/out
        + 5LL * (long long)sizeof(int);           // idx_in/out, is_start, group_id, group_starts

    // Default: scratch ~50% of avail_for_layer (preserves iter340 behavior on
    // n28/t000-t007 where the entire layer's pre-merge fits in scratch and no
    // overflow happens -- 30M states scratch fits all observed n28/u20 cases).
    //
    // iter370: when scratch overflows, the iter360 mid-layer concat+merge is
    // replaced by a non-merging spill ring (D2H copy, no compute work). This
    // means scratch is purely a streaming buffer; smaller scratch -> more
    // chunks per layer. Default 0.55 still works (preserves iter340 perf for
    // non-spill workloads); user can drop to e.g. 0.20 to make more room for
    // d_aggregator if the layer-end consolidation is the bottleneck.
    //
    // Override via:
    //   SAG_V5_SCRATCH_FRAC = fraction of avail-for-layer (default 0.55)
    //   SAG_V5_SCRATCH_STATES = absolute state count (overrides FRAC)
    double scratch_frac = 0.55;
    if (const char* p = std::getenv("SAG_V5_SCRATCH_FRAC")) {
        double v = std::atof(p);
        if (v > 0.0 && v < 1.0) scratch_frac = v;
    }

    // iter370 budget breakdown:
    //   d_layer_scratch:     scratch_cap * bps   (intra-layer streaming buffer)
    //   2 layer buffers (curr/next): 2 * L * bps  (no separate d_layer_temp;
    //                                              merge writes directly to
    //                                              d_next, which is the empty
    //                                              ping-pong slot)
    //   merge_aux:           scratch_cap * per_state_merge_aux
    //
    // d_aggregator (eagerly allocated when SAG_V5_HOST_SPILL_GB > 0):
    //   sized to ~20% of avail_for_layer (capped at 16 GB), reserves space
    //   for layer-end consolidate.
    //
    // Per-state effective costs:
    long long per_state_scratch_eff = (long long)bps + per_state_merge_aux;
    long long per_state_3layer_eff = 2LL * (long long)bps;  // iter370: 2 layer bufs

    long long avail_for_scratch = (long long)(avail_for_layer * scratch_frac);
    long long scratch_states = avail_for_scratch / per_state_scratch_eff;

    // SAG_V5_SCRATCH_STATES override (absolute state count).
    if (const char* p = std::getenv("SAG_V5_SCRATCH_STATES")) {
        long long v = std::atoll(p);
        if (v > 0) scratch_states = v;
    }

    // Floor: scratch must hold at least one wave's WORST-CASE OUTPUT
    // (wave_max_output = wave_states * n). This guarantees a single wave's
    // K2 output can never overflow scratch -- mid-layer flush only triggers
    // ACROSS waves, never WITHIN a wave.
    //
    // Override via SAG_V5_SCRATCH_FORCE_BELOW_FLOOR=1 to bypass for testing.
    long long min_states = cap.wave_max_output;
    bool force_below_floor = false;
    if (const char* p = std::getenv("SAG_V5_SCRATCH_FORCE_BELOW_FLOOR")) {
        force_below_floor = (p[0] != '0');
    }
    if (!force_below_floor && scratch_states < min_states) scratch_states = min_states;

    // After scratch carved out, the rest goes to 3 layer buffers.
    //
    // Note: if SAG_V5_ENABLE_FLUSH_CONCAT=1 is set later, d_layer_scratch
    // grows by layer_cap * bps to accommodate the concat target. The budget
    // here doesn't account for that; if FLUSH_CONCAT=1 causes OOM, the user
    // should reduce SAG_V5_SCRATCH_FRAC. Default is 0.55 with FLUSH_CONCAT
    // OFF, which fits comfortably on a 48 GB A100.
    long long scratch_bytes_used = scratch_states * per_state_scratch_eff;
    long long avail_for_layer_bufs = avail_for_layer - scratch_bytes_used;
    if (avail_for_layer_bufs < 0) avail_for_layer_bufs = (long long)(0.5 * avail_for_layer);

    long long layer_states = avail_for_layer_bufs / per_state_3layer_eff;
    if (layer_states < min_states) layer_states = min_states;

    // Hard cap: protect int indexing throughout the kernels and CUB calls.
    // Also, merge_scratch_max = scratch + layer must fit in INT_MAX (CUB index).
    long long lay_cap_int = (long long)INT_MAX;
    if (scratch_states > lay_cap_int) scratch_states = lay_cap_int;
    if (layer_states   > lay_cap_int / 2) layer_states = lay_cap_int / 2;
    if (scratch_states + layer_states > lay_cap_int) {
        layer_states = lay_cap_int - scratch_states;
        if (layer_states < min_states) layer_states = min_states;
    }

    cap.scratch_capacity = (int)scratch_states;
    cap.layer_capacity   = (int)layer_states;

    if (cap.scratch_capacity < cap.wave_states) cap.scratch_capacity = cap.wave_states;
    if (cap.layer_capacity   < cap.wave_states) cap.layer_capacity   = cap.wave_states;

    // iter370: aggregator capacity (only meaningful when spill is enabled).
    if (spill_enabled) {
        long long agg_states_ll = aggregator_bytes / bps;
        if (agg_states_ll > (long long)INT_MAX) agg_states_ll = INT_MAX;
        cap.aggregator_capacity = (int)agg_states_ll;
        if (cap.aggregator_capacity < cap.layer_capacity)
            cap.aggregator_capacity = cap.layer_capacity;

        // Ensure scratch_capacity <= aggregator_capacity so that the per-chunk
        // pre-sort pass can load each spilled chunk into d_aggregator alone.
        if (cap.aggregator_capacity > 0
            && cap.scratch_capacity > cap.aggregator_capacity) {
            cap.scratch_capacity = cap.aggregator_capacity;
        }
    } else {
        cap.aggregator_capacity = 0;  // unused when spill is disabled
    }

    // The merge_scratch_max needs to size for whichever is larger:
    //   - scratch_capacity (no-spill path)
    //   - aggregator_capacity (spill path)
    // We let the caller compute merge_scratch_max from these.
    return cap;
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    int M = 1;
    bool want_help = false;
    const char* cli_variant = nullptr;  // overrides SAG_V5_VARIANT if set
    std::vector<const char*> pos;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            M = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--variant") == 0 && i + 1 < argc) {
            cli_variant = argv[++i];
        } else if (std::strncmp(argv[i], "--variant=", 10) == 0) {
            cli_variant = argv[i] + 10;
        } else if (std::strcmp(argv[i], "-h") == 0 ||
                   std::strcmp(argv[i], "--help") == 0) {
            want_help = true;
        } else if (argv[i][0] != '-') {
            pos.push_back(argv[i]);
        } else {
            fprintf(stderr, "Unknown flag: %s\n", argv[i]);
            return 2;
        }
    }
    if (want_help) {
        fprintf(stdout,
            "expand_test_v5 -- V5 GPU SAG analyser, 4-paper-variant\n"
            "Usage: %s [-m cores] [--variant V] jobs.csv jobsprec.csv\n"
            "\n"
            "Variant selection (CLI takes precedence over env var):\n"
            "  --variant {rtss24|rtss17|ecrts19|ecrts22}  (default: rtss24)\n"
            "  SAG_V5_VARIANT=...                          (env, fallback)\n"
            "    rtss24:  Limited-preemptive SS+ED (Srinivasan/Gunzel/Nelissen RTSS 2024)\n"
            "    rtss17:  Exact uniprocessor non-preemptive (Nasri/Brandenburg RTSS 2017)\n"
            "    ecrts19: Limited-preemptive parallel DAG (Nasri/Nelissen/Brandenburg ECRTS 2019)\n"
            "    ecrts22: Non-preemptive moldable gang (Nelissen/Marce-i-Igual/Nasri ECRTS 2022)\n"
            "\n"
            "Other env vars:\n"
            "  SAG_V5_VERBOSE=1, SAG_V5_MAX_LAYERS=N, SAG_V5_F_MAX_PER_STATE=N\n"
            "  SAG_HOST_RAM_LIMIT_PCT=N (default 90), SAG_VRAM_LIMIT_MB=N\n"
            "\n"
            "See framework/VARIANTS.md for variant model details.\n",
            argv[0]);
        return 0;
    }
    if (M < 1 || M > MAX_CORES) {
        fprintf(stderr, "Error: -m must be in [1, %d]\n", MAX_CORES);
        return 2;
    }
    if (pos.size() != 2) {
        fprintf(stderr,
            "Usage: %s [-m cores] jobs.csv jobsprec.csv\n"
            "\n"
            "Environment variables:\n"
            "  SAG_V5_VARIANT={rtss24|rtss17|ecrts19|ecrts22}  (default: rtss24)\n"
            "  SAG_V5_IJP=1                  (deprecated; rtss24 only)\n"
            "  SAG_V5_POR=1                  (Phase A scaffold; predicate not yet shipped)\n"
            "  SAG_V5_VERBOSE=1              (per-layer diagnostics)\n"
            "  SAG_V5_MAX_LAYERS=N           (override default N+4)\n"
            "  SAG_V5_F_MAX_PER_STATE=N      (sparse F-pair cap; default 32)\n"
            "  SAG_HOST_RAM_LIMIT_PCT=N      (host RAM safety threshold; default 90)\n"
            "  SAG_VRAM_LIMIT_MB=N, SAG_MAX_VRAM_PCT=N\n"
            "\n"
            "See framework/VARIANTS.md for variant model details.\n",
            argv[0]);
        return 2;
    }

    const char* jobs_path = pos[0];
    const char* prec_path = pos[1];

    std::printf("framework (iter380: scratch-spill ring + pairwise K-way merge + single layer-end merge)\n");
    std::printf("  jobs:  %s\n", jobs_path);
    std::printf("  prec:  %s\n", prec_path);
    std::printf("  cores: %d\n", M);

    // ---- SAG variant selection (one user-facing knob) -----------------------
    // SAG_V5_VARIANT={rtss24|rtss17|ecrts19|ecrts22}; default rtss24.
    // Internal optimization toggles (IJP, merge level, etc.) are auto-selected
    // per variant via v5_default_tricks(); see framework/VARIANTS.md.
    sag::v5::V5Variant g_variant = sag::v5::V5Variant::NPG_RTSS24;
    {
        bool ok = true;
        // CLI --variant takes precedence over SAG_V5_VARIANT env var.
        const char* p_var = cli_variant
                          ? cli_variant
                          : std::getenv("SAG_V5_VARIANT");
        g_variant = sag::v5::v5_parse_variant(p_var, &ok);
        if (!ok) {
            std::fprintf(stderr,
                "Error: variant='%s' is not a recognized SAG variant.\n",
                p_var ? p_var : "");
            sag::v5::v5_print_variant_choices(stderr);
            return 7;
        }
    }
    sag::v5::V5VariantTricks g_tricks = sag::v5::v5_default_tricks(g_variant);
    std::printf("V5 variant: %s -- %s\n",
                sag::v5::v5_variant_name(g_variant),
                sag::v5::v5_variant_paper(g_variant));
    std::printf("  auto-tricks: %s\n", g_tricks.notes);
    if (g_tricks.m_eq_one && M != 1) {
        std::printf("  variant forces m=1 (was %d via -m); using m=1.\n", M);
        M = 1;
    }

    // All four variants are implemented in Phase 4. ECRTS 2019 (LP DAG) uses
    // the V5 default K1+K2+Merge on segment-expanded inputs (each task's
    // segments are independent jobs in jobs.csv with chain edges in
    // jobsprec.csv -- the existing test data already follows this pattern).
    // The limited-preemptive semantics fall out of segment-level precedence.

    std::vector<V5JobRecord> records;
    int max_iid = -1;
    bool gang_format = (g_variant == sag::v5::V5Variant::GANG_22);
    if (!v5_parse_jobs_csv(jobs_path, records, max_iid, gang_format)) {
        fprintf(stderr, "Error: cannot open jobs file '%s'\n", jobs_path);
        return 1;
    }
    int N = max_iid + 1;
    if (N < 1) {
        fprintf(stderr,
            "Error: jobs file '%s' contains 0 valid jobs. "
            "Expected at least one row 'tid, jid, r_min, r_max, ...'.\n",
            jobs_path);
        return 1;
    }
    int W = bitset_words(N);
    if (W > MAX_BITSET_WORDS) {
        fprintf(stderr,
            "iter310: N=%d exceeds compile-time MAX_BITSET_WORDS*64=%d.\n",
            N, MAX_BITSET_WORDS * 64);
        return 3;
    }

    std::vector<V5PrecEdge> edges;
    if (!v5_parse_prec_csv(prec_path, edges)) {
        fprintf(stderr, "Error: cannot open prec file '%s'\n", prec_path);
        return 1;
    }
    std::printf("Parsed %d jobs, %zu precedence edges (N=%d, W=%d)\n",
                N, edges.size(), N, W);

    V5Topology topo;
    if (!v5_build_topology(records, N, W, edges, topo)) {
        fprintf(stderr, "Error: topology build failed\n");
        return 1;
    }
    int num_candidates = (int)topo.candidates.size();
    std::printf("Topology built: %d candidates after pruning, max_depth=%d "
                "(per-layer K1 truncation enabled via depth_prefix[%d])\n",
                num_candidates, topo.max_depth, (int)topo.depth_prefix.size());

    SAGStateLayoutV5 layout;
    layout.W = W; layout.n = N; layout.m = M;
    int bps = layout.bytes_per_state();
    {
        // Report sparse-vs-dense bytes ratio for visibility (not load-bearing).
        int dense_bps = 3 * W * (int)sizeof(uint64_t)
                      + 2 * M * (int)sizeof(int32_t)
                      + 2 * N * (int)sizeof(int32_t)
                      + (int)sizeof(int32_t);
        dense_bps = (dense_bps + 7) & ~7;
        std::printf("V5 sparse layout: F_MAX_PER_STATE=%d, bps_v5=%d, "
                    "bps_dense=%d (ratio %.2fx)\n",
                    F_MAX_PER_STATE, bps, dense_bps,
                    (double)bps / (double)dense_bps);
    }

    // iter370: spill_enabled affects budget (reserves aggregator space).
    bool spill_enabled = false;
    if (const char* p = std::getenv("SAG_V5_HOST_SPILL_GB")) {
        if (std::atoll(p) > 0) spill_enabled = true;
    }
    V5Capacity cap = v5_compute_capacity(N, M, W, bps, spill_enabled);
    std::printf("VRAM: usable=%.1f MB, bps=%d\n",
                cap.usable_gpu_bytes / (1024.0 * 1024.0), bps);
    std::printf("Layer cap:    %d states (%.1f MB x 2 layer buffers: curr/next)\n",
                cap.layer_capacity,
                (double)cap.layer_capacity * bps / (1024.0 * 1024.0));
    std::printf("Scratch cap:  %d states (%.1f MB streaming buffer; "
                "scratch-spill ring on overflow)\n",
                cap.scratch_capacity,
                (double)cap.scratch_capacity * bps / (1024.0 * 1024.0));
    if (spill_enabled) {
        std::printf("Aggregator:   %d states (%.1f MB; layer-end consolidate "
                    "paging window)\n",
                    cap.aggregator_capacity,
                    (double)cap.aggregator_capacity * bps / (1024.0 * 1024.0));
    }
    std::printf("Wave cap:     %d states/wave, max_pairs=%d, max_output=%d "
                "(%.1f MB output buf)\n",
                cap.wave_states, cap.wave_max_pairs, cap.wave_max_output,
                (double)cap.wave_max_output * bps / (1024.0 * 1024.0));

    // ----- Allocate GPU buffers (iter370) -----
    //
    // iter370 layout (2-buffer ping-pong + scratch + optional aggregator):
    //   - d_layerA / d_layerB: ping-pong layer buffers (d_curr, d_next).
    //     Each sized to layer_capacity * bps. NO d_layer_temp -- the merge
    //     writes directly to d_next, which is the empty ping-pong slot.
    //   - d_layer_scratch: streaming accumulator. Holds per-wave merge
    //     results until it would overflow on the next wave's output. On
    //     overflow: D2H copy of the scratch tail to h_scratch_spill (no
    //     merge), reset to 0, continue. NO mid-layer compute work.
    //   - d_wave_output: per-wave K2 output scratch, wave_max_output * bps.
    //     Reused for every wave; per-wave merge consumes it and emits into
    //     d_layer_scratch at offset = total_pre_cw_merged.
    //   - d_valid_pairs: per-wave K1 output scratch, wave_max_pairs * sizeof.
    //   - d_aggregator (lazy, only when SAG_V5_HOST_SPILL_GB > 0): paging
    //     window for the layer-end consolidate pass that pages spilled
    //     chunks back from h_scratch_spill, sorts+dedupes, merges to d_next.
    //
    // Layer-end merge dataflow:
    //   No-spill path (scratch ring empty):
    //     d_pre  = d_layer_scratch[0..total_pre_cw_merged)
    //     d_pre2 = nullptr, pre1_count = 0
    //     d_post = d_next
    //   Spill path (scratch ring has chunks):
    //     PRE-SORT (per chunk, only if combined exceeds aggregator + scratch
    //     capacity): each chunk loaded into d_aggregator, sort+merge-deduped,
    //     deduped result D2H'd back to h_scratch_spill (overwriting the
    //     larger pre-sort source).
    //     CONSOLIDATE: chunks copied to d_aggregator (filling it),
    //     overflow + residual scratch to d_layer_scratch.
    //     Dual-source merge:
    //       d_pre  = d_aggregator,  pre1_count = agg_count
    //       d_pre2 = d_layer_scratch
    //       d_post = d_next
    char *d_layerA = nullptr, *d_layerB = nullptr;
    V5_CUDA(cudaMalloc(&d_layerA, (size_t)cap.layer_capacity * bps));
    V5_CUDA(cudaMalloc(&d_layerB, (size_t)cap.layer_capacity * bps));
    V5_CUDA(cudaMemset(d_layerA, 0, (size_t)cap.layer_capacity * bps));

    // d_layer_scratch sized to scratch_cap * bps (streaming buffer).
    // iter370: no longer grown by layer_cap*bps -- the iter360 concat-into-
    // scratch-then-resort path is replaced by the scratch ring + d_aggregator
    // layer-end merge.
    long long d_layer_scratch_states = (long long)cap.scratch_capacity;
    char* d_layer_scratch = nullptr;
    V5_CUDA(cudaMalloc(&d_layer_scratch, (size_t)d_layer_scratch_states * bps));

    char* d_wave_output = nullptr;
    V5_CUDA(cudaMalloc(&d_wave_output, (size_t)cap.wave_max_output * bps));

    ValidPair* d_valid_pairs = nullptr;
    V5_CUDA(cudaMalloc(&d_valid_pairs,
                       (size_t)cap.wave_max_pairs * sizeof(ValidPair)));
    // Phase 3.3.b: per-pair side arrays for moldable enumeration. Allocated
    // for ECRTS22 only; other variants leave them nullptr so K2BodyV5 falls
    // back to d_C_min[j]/d_C_max[j]. (Declared at module scope below; this
    // block is the actual cudaMalloc.) Note: g_variant has been parsed
    // earlier (line ~1225) so this branch is sound here.
    int8_t  *d_pair_p    = nullptr;
    int32_t *d_pair_cmin = nullptr;
    int32_t *d_pair_cmax = nullptr;
    // ECRTS22 perf: detect all-rigid-p=1 input. When true, the entire
    // ECRTS22 path becomes mathematically equivalent to rtss24 (single
    // p=1 dispatch, A_pair[0] eligibility, single-core K2 commit, default
    // merge). We re-route to rtss24's K1+K2+merge kernels for these
    // inputs -- skips the per-pair side-array reads/writes entirely
    // and eliminates the d_p_max/CSR-cost-map kernel-arg overhead.
    bool g_ecrts22_route_to_rtss24 = false;
    if (g_variant == sag::v5::V5Variant::GANG_22) {
        // Two perf-significant input shapes:
        //   (A) all-rigid-p=1: the entire ECRTS22 path is math-equivalent to
        //       rtss24. Route to rtss24's K1+K2+merge entirely.
        //   (B) all-rigid (any p): CSR has 1 entry per job. The per-pair
        //       side arrays carry no info beyond d_p_max[j]/d_C_min[j]/
        //       d_C_max[j]. K1ECRTS22 + K2BodyV5 already handle nullptr
        //       fallback. Skip side-array alloc but keep ECRTS22 path
        //       (parallelism-aware A_pair[p-1] still needed for p>=2).
        bool all_rigid_p1 = true;
        bool all_rigid    = true;
        for (int i = 0; i < N; i++) {
            int n_p = (i + 1 < (int)topo.costmap_offset.size())
                      ? (topo.costmap_offset[i + 1] - topo.costmap_offset[i])
                      : 1;
            if (n_p > 1) {
                all_rigid    = false;
                all_rigid_p1 = false;
                break;
            }
            if (topo.p_max[i] > 1) {
                all_rigid_p1 = false;
            }
        }
        if (all_rigid_p1) {
            g_ecrts22_route_to_rtss24 = true;
            std::printf("ECRTS22 perf: all-rigid-p=1 detected; routing "
                        "K1+K2+merge through rtss24 path "
                        "(saves %.1f MB VRAM + per-call side-array I/O)\n",
                        (double)cap.wave_max_pairs *
                        (sizeof(int8_t) + 2 * sizeof(int32_t)) / (1024.0 * 1024.0));
            // Keep d_pair_* nullptr; rtss24 path doesn't read them anyway.
        } else if (all_rigid) {
            // Rigid p>=2: skip side arrays; K1+K2 fall back to per-job
            // d_p_max[j] / d_C_min[j] / d_C_max[j]. ECRTS22 K1 path
            // still runs (parallelism-aware eligibility for p>=2).
            std::printf("ECRTS22 perf: all-rigid (any p) detected; "
                        "skipping per-pair side arrays "
                        "(saves %.1f MB VRAM)\n",
                        (double)cap.wave_max_pairs *
                        (sizeof(int8_t) + 2 * sizeof(int32_t)) / (1024.0 * 1024.0));
        } else {
            // Moldable: side arrays carry K1's chosen p + cost.
            V5_CUDA(cudaMalloc(&d_pair_p,    (size_t)cap.wave_max_pairs * sizeof(int8_t)));
            V5_CUDA(cudaMalloc(&d_pair_cmin, (size_t)cap.wave_max_pairs * sizeof(int32_t)));
            V5_CUDA(cudaMalloc(&d_pair_cmax, (size_t)cap.wave_max_pairs * sizeof(int32_t)));
        }
    }

    // ----- iter370: DRAM spill (scratch ring + layer-output ping-pong) -----
    //
    // Two separate kinds of host-pinned spill:
    //
    // A. SCRATCH SPILL RING (h_scratch_spill, single buffer):
    //    Used WITHIN a single layer when d_layer_scratch overflows. When the
    //    next wave's worst-case output would not fit in d_layer_scratch, we
    //    D2H-copy the entire current scratch tail into the next slot of the
    //    ring buffer and reset total_pre_cw_merged = 0. NO MERGE happens at
    //    that point. Each chunk is recorded in scratch_spill_chunks (host
    //    vector) for the layer-end consolidation pass.
    //    iter360's "concat-then-merge mid-flush" path was O(K^2) PCIe
    //    movement; this ring preserves the iter370 invariant: each state
    //    spilled at MOST ONCE per layer (out + back = 2 PCIe touches max).
    //    Reset at every layer start.
    //
    // B. LAYER-OUTPUT PING-PONG (h_layer_spill_curr / h_layer_spill_next,
    //    iter360 design preserved): used between layers. When the post-merge
    //    layer (i.e. the input to the NEXT layer) exceeds layer_capacity,
    //    excess states go to h_layer_spill_next; next layer reads them as
    //    h_layer_spill_curr via d_paged_input.
    //
    // Sizing knobs (env vars):
    //   SAG_V5_HOST_SPILL_GB        : total host-pinned cap (BOTH layer-output
    //                                 ping-pong buffers + scratch ring) in GB.
    //                                 Default: 0 (spill DISABLED for safety
    //                                 on small smoke tests).
    //                                 Per-buffer split: 50% scratch ring,
    //                                 25% h_layer_spill_curr, 25% h_layer_spill_next.
    //                                 Adjust SAG_V5_HOST_SPILL_GB to make room
    //                                 for the workload's peak per-layer total.
    //   SAG_V5_SPILL_CHUNK_MB       : staging chunk size in MB. Default 2048
    //                                 (2 GB) for d_paged_input + d_spill_staging.
    //   SAG_V5_AGGREGATOR_GB        : GPU aggregator size in GB for layer-end
    //                                 merge consolidation. Default: auto from
    //                                 remaining VRAM after other allocations.
    //                                 Larger -> fewer multi-pass folds.
    //   SAG_V5_ENABLE_FLUSH_CONCAT  : DEPRECATED; iter370 ignores this and
    //                                 prints a warning when it is set. Mid-
    //                                 layer flush is replaced by scratch ring.
    //
    // LAZY ALLOC: scratch ring + layer-output ping-pong are allocated on the
    // FIRST overflow event (preserves small smoke-test perf).
    //
    // FAIL-LOUD: if pinned alloc fails we fall back to plain malloc; if that
    // fails or the workload's peak fits no available buffer, TRUNC LOUD.

    long long sys_avail_bytes = 0;
    {
        FILE* fmem = std::fopen("/proc/meminfo", "r");
        if (fmem) {
            char line[256];
            while (std::fgets(line, sizeof(line), fmem)) {
                long long kb;
                if (std::sscanf(line, "MemAvailable: %lld kB", &kb) == 1) {
                    sys_avail_bytes = kb * 1024LL;
                    break;
                }
            }
            std::fclose(fmem);
        }
    }
    if (sys_avail_bytes <= 0) sys_avail_bytes = 32LL * 1024 * 1024 * 1024;

    // iter370: total host-pinned spill budget, split as
    //   50% scratch ring, 25% layer_spill_curr, 25% layer_spill_next.
    // Default 0 (disabled). Set SAG_V5_HOST_SPILL_GB=N to enable.
    long long host_spill_total_bytes = 0;
    if (const char* p = std::getenv("SAG_V5_HOST_SPILL_GB")) {
        long long v_gb = std::atoll(p);
        host_spill_total_bytes = v_gb * 1024LL * 1024LL * 1024LL;
    }
    if (host_spill_total_bytes < 0) host_spill_total_bytes = 0;

    // Per-buffer caps.
    //   scratch ring   (per-layer intra) :   ~50% of total
    //   layer_spill_curr/next (per-layer to-next): ~25% each
    long long host_spill_per_buf_bytes_cap = host_spill_total_bytes / 4LL;
    long long host_scratch_ring_bytes_cap = host_spill_total_bytes / 2LL;

    // iter360 had ONE knob splitting in half; iter370 splits in quarters.
    // To preserve iter360 backward-compat for users who set SAG_V5_HOST_SPILL_GB=64
    // we keep the same total budget and give half to the new scratch ring.

    // DEPRECATION warning for SAG_V5_ENABLE_FLUSH_CONCAT (iter360 knob).
    if (const char* p = std::getenv("SAG_V5_ENABLE_FLUSH_CONCAT")) {
        if (p[0] != '0') {
            std::fprintf(stderr,
                "[V5 WARN] SAG_V5_ENABLE_FLUSH_CONCAT=1 is DEPRECATED in iter370. "
                "The mid-layer concat-and-re-merge path has been replaced by a "
                "non-merging scratch spill ring + single layer-end merge. "
                "Ignoring the env var.\n");
        }
    }

    // iter370: default spill staging chunk is 512 MB (was 2 GB in iter360);
    // smaller default leaves more headroom for d_aggregator on contended
    // GPUs. User can bump via SAG_V5_SPILL_CHUNK_MB.
    long long spill_chunk_bytes = 512LL * 1024 * 1024;
    if (const char* p = std::getenv("SAG_V5_SPILL_CHUNK_MB")) {
        long long v_mb = std::atoll(p);
        if (v_mb > 0) spill_chunk_bytes = v_mb * 1024LL * 1024LL;
    }
    int spill_chunk_states = (int)(spill_chunk_bytes / bps);
    if (spill_chunk_states < 1024) spill_chunk_states = 1024;
    if ((long long)spill_chunk_states > (long long)cap.layer_capacity)
        spill_chunk_states = cap.layer_capacity;

    // varwidth-spill: per-chunk variable-width F_entries packing on PCIe
    // boundary. Saves ~25-30% of spill bytes when typical F_count is much
    // less than F_MAX_PER_STATE (n=140 sees popcount(F_mask|X) <= 21 vs
    // F_MAX_PER_STATE=32 -> packed_bps = 392 vs dense bps = 488).
    // Default ON; SAG_V5_VARWIDTH_SPILL=0 disables.
    bool varwidth_spill_enabled = true;
    if (const char* p = std::getenv("SAG_V5_VARWIDTH_SPILL")) {
        varwidth_spill_enabled = (p[0] != '\0' && p[0] != '0');
    }
    // For pack/unpack: warp-per-state. Use 256 threads/block = 8 warps/block.
    constexpr int VARWIDTH_THREADS_PER_BLOCK = 256;
    constexpr int VARWIDTH_WARPS_PER_BLOCK   =
        VARWIDTH_THREADS_PER_BLOCK / WARP_SIZE;
    // Diagnostic counters (track varwidth savings).
    long long total_spill_bytes_dense_eq = 0;   // bytes the dense path WOULD have spilled
    long long total_spill_states_packed   = 0;

    // Lazy-init state: not allocated until first overflow.
    char* h_layer_spill_curr = nullptr;
    char* h_layer_spill_next = nullptr;
    char* h_scratch_spill    = nullptr;     // iter370: scratch ring buffer
    char* d_paged_input      = nullptr;
    char* d_spill_staging    = nullptr;
    char* d_aggregator       = nullptr;     // iter370: layer-end paging window
    bool  spill_initialized  = false;
    bool  spill_pinned       = false;
    // iter400: chunked lazy allocation. *_bytes is the LOGICAL cap (user's
    // SAG_V5_HOST_SPILL_GB-derived ceiling); *_bytes_alloced is the actual
    // currently-allocated size. We grow in 32 GB increments up to the cap.
    long long host_spill_per_buf_bytes = 0;          // logical cap (= _cap)
    long long host_scratch_ring_bytes  = 0;          // logical cap (= _cap)
    long long host_spill_per_buf_bytes_alloced = 0;  // currently allocated
    long long host_scratch_ring_bytes_alloced  = 0;  // currently allocated
    int  h_layer_spill_per_buf_states  = 0;          // states fitting in alloced
    int  h_scratch_ring_capacity_states = 0;         // states fitting in alloced
    int  d_aggregator_capacity_states  = 0; // iter370

    cudaStream_t spill_stream = 0;
    cudaStream_t page_stream  = 0;

    // iter370: when spill is enabled, EAGERLY allocate d_aggregator at
    // startup so OOM happens here (not mid-run). The remaining lazy state
    // (host pinned ring, ping-pong) stays lazy because it's much larger
    // and not always needed.
    if (spill_enabled && cap.aggregator_capacity > 0) {
        long long agg_bytes = (long long)cap.aggregator_capacity * (long long)bps;
        // Cap to current free VRAM minus slack.
        size_t free_now = 0, total_now = 0;
        cudaMemGetInfo(&free_now, &total_now);
        long long slack = 256LL * 1024 * 1024;
        long long free_capped = (long long)free_now - slack;
        if (free_capped < 0) free_capped = 0;
        if (agg_bytes > free_capped) {
            std::fprintf(stderr,
                "[V5 WARN] eager d_aggregator: budget %lld MB > free %lld MB; "
                "capping to free.\n",
                agg_bytes / (1024*1024), free_capped / (1024*1024));
            agg_bytes = free_capped;
        }
        if (agg_bytes < (long long)cap.layer_capacity * bps) {
            std::fprintf(stderr,
                "[V5 WARN] eager d_aggregator: free VRAM %lld MB too small "
                "to hold one layer's worth (need %lld MB); spill DISABLED.\n",
                agg_bytes / (1024*1024),
                (long long)cap.layer_capacity * bps / (1024*1024));
            // Leave d_aggregator = nullptr; spill_enabled stays true but
            // the lazy-init will fail later if needed. For workloads where
            // spill never triggers, this is fine.
        } else {
            cudaError_t ge = cudaMalloc(&d_aggregator, (size_t)agg_bytes);
            if (ge != cudaSuccess) {
                std::fprintf(stderr,
                    "[V5 WARN] eager d_aggregator alloc (%lld bytes) failed: "
                    "%s; spill DISABLED.\n",
                    agg_bytes, cudaGetErrorString(ge));
                d_aggregator = nullptr;
            } else {
                long long agg_states = agg_bytes / bps;
                if (agg_states > (long long)INT_MAX) agg_states = INT_MAX;
                d_aggregator_capacity_states = (int)agg_states;
                std::printf("V5 d_aggregator EAGER alloc: %.2f GB (%d states)\n",
                            (double)agg_bytes / (1024.0*1024.0*1024.0),
                            d_aggregator_capacity_states);
            }
        }
    }

    // iter400: chunked lazy allocation. The LAYER_CHUNK is the increment used
    // when growing a spill buffer. We start with min(cap, INITIAL_*_CHUNK) and
    // grow up to the cap on demand. This avoids paying the O(95s) up-front
    // alloc cost on tier-2 cases that don't actually need the full cap.
    const long long ALLOC_CHUNK_BYTES = 32LL * 1024 * 1024 * 1024;  // 32 GB
    // For each ping-pong buffer use up to half a 32GB chunk initially (16 GB)
    // so we don't blow the budget on two ping-pong buffers + one ring.
    const long long INITIAL_PER_BUF_BYTES = ALLOC_CHUNK_BYTES / 2;  // 16 GB
    const long long INITIAL_RING_BYTES    = ALLOC_CHUNK_BYTES;      // 32 GB
    // Helper: pinned/malloc allocator with fallback. Used by both the initial
    // alloc and grow paths. Returns true on success and sets *out_pinned to
    // whether the result is pinned.
    auto host_alloc = [&](void** out_buf, long long bytes,
                          bool prefer_pinned, bool* out_pinned) -> bool {
        *out_buf = nullptr;
        if (out_pinned) *out_pinned = false;
        if (bytes <= 0) return false;
        if (prefer_pinned) {
            // Use v5_mh_alloc (mmap + cudaHostRegister + 2MB HugeTLB attempt)
            // instead of cudaHostAlloc. Saves ~5% on multi-GB pinned alloc time.
            cudaError_t e = v5_mh_alloc(out_buf, (size_t)bytes);
            if (e == cudaSuccess && *out_buf != nullptr) {
                if (out_pinned) *out_pinned = true;
                return true;
            }
            // pinned alloc failed; fall back to malloc
            std::fprintf(stderr,
                "[V5 WARN] v5_mh_alloc(%lld bytes) failed (%s); falling back "
                "to plain malloc (slower).\n",
                bytes, cudaGetErrorString(e));
            *out_buf = nullptr;
        }
        *out_buf = std::malloc((size_t)bytes);
        if (out_pinned) *out_pinned = false;
        return *out_buf != nullptr;
    };
    auto host_free = [&](void* buf) {
        if (!buf) return;
        if (spill_pinned) v5_mh_free(buf);
        else              std::free(buf);
    };

    // Lazy initializer (called on first overflow event -- either scratch ring
    // overflow during a wave, or post-merge n_groups > layer_capacity).
    auto ensure_spill_initialized = [&]() -> bool {
        if (spill_initialized) return true;
        if (host_spill_total_bytes <= 0) {
            std::fprintf(stderr,
                "[V5 WARN] spill required but SAG_V5_HOST_SPILL_GB=0 "
                "(default for safety). Set SAG_V5_HOST_SPILL_GB=N to enable.\n");
            return false;
        }

        // iter400: only allocate the INITIAL chunk size (or cap if smaller).
        // Subsequent growth happens lazily in chunked-realloc paths inside
        // do_scratch_spill / do_flush / do_layer_end_consolidate.
        long long initial_per_buf =
            std::min((long long)host_spill_per_buf_bytes_cap,
                     INITIAL_PER_BUF_BYTES);
        long long initial_ring =
            std::min((long long)host_scratch_ring_bytes_cap,
                     INITIAL_RING_BYTES);

        // Try cudaHostAlloc; on first failure fall back to plain malloc for
        // ALL three buffers (consistent pinning state).
        bool p1 = false, p2 = false, p3 = false;
        bool ok1 = host_alloc((void**)&h_layer_spill_curr, initial_per_buf,
                              /*prefer_pinned=*/true, &p1);
        bool ok2 = host_alloc((void**)&h_layer_spill_next, initial_per_buf,
                              /*prefer_pinned=*/true, &p2);
        bool ok3 = host_alloc((void**)&h_scratch_spill, initial_ring,
                              /*prefer_pinned=*/true, &p3);
        if (!ok1 || !ok2 || !ok3 || (p1 != p2) || (p2 != p3)) {
            // Inconsistent pinning or any failure: free + redo as malloc.
            if (h_layer_spill_curr) {
                if (p1) v5_mh_free(h_layer_spill_curr);
                else    std::free(h_layer_spill_curr);
                h_layer_spill_curr = nullptr;
            }
            if (h_layer_spill_next) {
                if (p2) v5_mh_free(h_layer_spill_next);
                else    std::free(h_layer_spill_next);
                h_layer_spill_next = nullptr;
            }
            if (h_scratch_spill) {
                if (p3) v5_mh_free(h_scratch_spill);
                else    std::free(h_scratch_spill);
                h_scratch_spill = nullptr;
            }
            h_layer_spill_curr = (char*)std::malloc((size_t)initial_per_buf);
            h_layer_spill_next = (char*)std::malloc((size_t)initial_per_buf);
            h_scratch_spill    = (char*)std::malloc((size_t)initial_ring);
            spill_pinned = false;
        } else {
            spill_pinned = true;
        }
        if (!h_layer_spill_curr || !h_layer_spill_next || !h_scratch_spill) {
            std::fprintf(stderr,
                "[V5 WARN] host spill alloc failed entirely; spill UNAVAILABLE.\n");
            return false;
        }
        host_spill_per_buf_bytes = host_spill_per_buf_bytes_cap;
        host_scratch_ring_bytes  = host_scratch_ring_bytes_cap;
        host_spill_per_buf_bytes_alloced = initial_per_buf;
        host_scratch_ring_bytes_alloced  = initial_ring;

        long long states_ll = host_spill_per_buf_bytes_alloced / bps;
        if (states_ll > (long long)INT_MAX) states_ll = INT_MAX;
        h_layer_spill_per_buf_states = (int)states_ll;

        long long ring_states_ll = host_scratch_ring_bytes_alloced / bps;
        if (ring_states_ll > (long long)INT_MAX) ring_states_ll = INT_MAX;
        h_scratch_ring_capacity_states = (int)ring_states_ll;

        // GPU-side staging.
        cudaError_t ge1 = cudaMalloc(&d_paged_input,
                                     (size_t)spill_chunk_states * bps);
        cudaError_t ge2 = cudaMalloc(&d_spill_staging,
                                     (size_t)spill_chunk_states * bps);
        if (ge1 != cudaSuccess || ge2 != cudaSuccess) {
            std::fprintf(stderr,
                "[V5 WARN] GPU staging alloc failed (%s/%s); spill DISABLED.\n",
                cudaGetErrorString(ge1), cudaGetErrorString(ge2));
            if (d_paged_input)   { cudaFree(d_paged_input);   d_paged_input = nullptr; }
            if (d_spill_staging) { cudaFree(d_spill_staging); d_spill_staging = nullptr; }
            return false;
        }

        // iter370: d_aggregator was eagerly allocated at startup (if
        // spill_enabled), so we expect it here. If it's null, the eager
        // alloc failed -- spill is unavailable.
        if (!d_aggregator) {
            std::fprintf(stderr,
                "[V5 WARN] d_aggregator not available (eager alloc failed at "
                "startup, see prior WARN); spill DISABLED.\n");
            if (d_paged_input)   { cudaFree(d_paged_input);   d_paged_input = nullptr; }
            if (d_spill_staging) { cudaFree(d_spill_staging); d_spill_staging = nullptr; }
            return false;
        }
        long long aggregator_bytes = (long long)d_aggregator_capacity_states * bps;

        V5_CUDA(cudaStreamCreate(&spill_stream));
        V5_CUDA(cudaStreamCreate(&page_stream));

        std::printf("V5 spill INIT (iter400 chunked): scratch_ring init=%.2f GB "
                    "/ cap=%.2f GB (%d states init), layer_spill init=%.2f GB "
                    "/ cap=%.2f GB x2 (%d states each init), "
                    "aggregator=%.2f GB (%d states), pinned=%s\n",
                    (double)host_scratch_ring_bytes_alloced / (1024.0*1024.0*1024.0),
                    (double)host_scratch_ring_bytes / (1024.0*1024.0*1024.0),
                    h_scratch_ring_capacity_states,
                    (double)host_spill_per_buf_bytes_alloced / (1024.0*1024.0*1024.0),
                    (double)host_spill_per_buf_bytes / (1024.0*1024.0*1024.0),
                    h_layer_spill_per_buf_states,
                    (double)aggregator_bytes / (1024.0*1024.0*1024.0),
                    d_aggregator_capacity_states,
                    spill_pinned ? "yes" : "no");
        spill_initialized = true;
        return true;
    };

    // iter400: grow the scratch ring buffer to satisfy `min_bytes_needed`.
    // Returns true if the ring is now sized >= min_bytes_needed (up to cap).
    // Returns false if growth is impossible (already at cap or alloc failed).
    // On growth, copies `used_bytes` of existing data to the new buffer.
    auto grow_scratch_ring = [&](long long min_bytes_needed,
                                  long long used_bytes) -> bool {
        if (host_scratch_ring_bytes_alloced >= min_bytes_needed) return true;
        if (host_scratch_ring_bytes_alloced >= host_scratch_ring_bytes) {
            return false;  // already at cap
        }
        // Exponential (doubling) growth: at least double each grow event,
        // plus enough to fit need. Initial grow from base ~64 GB to 128 GB;
        // then 256 GB; then cap. Round up to ALLOC_CHUNK_BYTES alignment.
        long long new_alloced = host_scratch_ring_bytes_alloced;
        long long new_target = std::max(new_alloced * 2, min_bytes_needed);
        if (new_target % ALLOC_CHUNK_BYTES != 0) {
            new_target = ((new_target + ALLOC_CHUNK_BYTES - 1) / ALLOC_CHUNK_BYTES)
                         * ALLOC_CHUNK_BYTES;
        }
        new_alloced = new_target;
        if (new_alloced > host_scratch_ring_bytes) {
            new_alloced = host_scratch_ring_bytes;
        }
        if (new_alloced <= host_scratch_ring_bytes_alloced) return false;
        if (new_alloced < min_bytes_needed) {
            // even at cap we cannot fit
            return false;
        }
        char* new_buf = nullptr;
        bool new_pinned = false;
        bool ok = host_alloc((void**)&new_buf, new_alloced,
                             /*prefer_pinned=*/spill_pinned, &new_pinned);
        if (!ok || !new_buf) {
            std::fprintf(stderr,
                "[V5 WARN] grow_scratch_ring(%lld bytes) alloc failed.\n",
                new_alloced);
            return false;
        }
        if (new_pinned != spill_pinned) {
            // Honor the global pinning state. If pinned alloc fell back to
            // malloc, treat as failure to keep state consistent.
            std::fprintf(stderr,
                "[V5 WARN] grow_scratch_ring: alloc pin-state mismatch "
                "(want %s, got %s); rejecting growth.\n",
                spill_pinned ? "pinned" : "malloc",
                new_pinned ? "pinned" : "malloc");
            if (new_pinned) v5_mh_free(new_buf);
            else            std::free(new_buf);
            return false;
        }
        if (used_bytes > 0 && h_scratch_spill != nullptr) {
            std::memcpy(new_buf, h_scratch_spill, (size_t)used_bytes);
        }
        host_free(h_scratch_spill);
        h_scratch_spill = new_buf;
        host_scratch_ring_bytes_alloced = new_alloced;
        long long ring_states_ll = host_scratch_ring_bytes_alloced / bps;
        if (ring_states_ll > (long long)INT_MAX) ring_states_ll = INT_MAX;
        h_scratch_ring_capacity_states = (int)ring_states_ll;
        {
            const char* vp = std::getenv("V5_VERBOSE");
            if (vp && vp[0] != '0') {
                std::fprintf(stderr,
                    "[V5 iter400] grew scratch_ring to %.2f GB (cap %.2f GB).\n",
                    (double)host_scratch_ring_bytes_alloced
                        / (1024.0*1024.0*1024.0),
                    (double)host_scratch_ring_bytes / (1024.0*1024.0*1024.0));
            }
        }
        return true;
    };

    // iter400: grow a layer-spill ping-pong buffer (curr or next) to satisfy
    // `min_bytes_needed`. Both curr and next are kept the same allocated size
    // so the swap remains symmetric. `used_curr_bytes` / `used_next_bytes`
    // are the bytes-of-live-data in each buffer to preserve on grow.
    auto grow_layer_spill_buffers =
        [&](long long min_bytes_needed,
            long long used_curr_bytes,
            long long used_next_bytes) -> bool {
        if (host_spill_per_buf_bytes_alloced >= min_bytes_needed) return true;
        if (host_spill_per_buf_bytes_alloced >= host_spill_per_buf_bytes) {
            return false;  // already at cap
        }
        // Exponential (doubling) growth: at least double each grow event,
        // plus enough to fit need. Round up to ALLOC_CHUNK_BYTES alignment.
        long long new_alloced = host_spill_per_buf_bytes_alloced;
        long long new_target = std::max(new_alloced * 2, min_bytes_needed);
        if (new_target % ALLOC_CHUNK_BYTES != 0) {
            new_target = ((new_target + ALLOC_CHUNK_BYTES - 1) / ALLOC_CHUNK_BYTES)
                         * ALLOC_CHUNK_BYTES;
        }
        new_alloced = new_target;
        if (new_alloced > host_spill_per_buf_bytes) {
            new_alloced = host_spill_per_buf_bytes;
        }
        if (new_alloced <= host_spill_per_buf_bytes_alloced) return false;
        if (new_alloced < min_bytes_needed) return false;

        char* new_curr = nullptr;
        char* new_next = nullptr;
        bool p_curr = false, p_next = false;
        bool ok1 = host_alloc((void**)&new_curr, new_alloced,
                              /*prefer_pinned=*/spill_pinned, &p_curr);
        bool ok2 = host_alloc((void**)&new_next, new_alloced,
                              /*prefer_pinned=*/spill_pinned, &p_next);
        if (!ok1 || !ok2 || p_curr != spill_pinned || p_next != spill_pinned) {
            std::fprintf(stderr,
                "[V5 WARN] grow_layer_spill_buffers(%lld) alloc failed or "
                "pin-state mismatch.\n", new_alloced);
            if (new_curr) {
                if (p_curr) v5_mh_free(new_curr);
                else        std::free(new_curr);
            }
            if (new_next) {
                if (p_next) v5_mh_free(new_next);
                else        std::free(new_next);
            }
            return false;
        }
        if (used_curr_bytes > 0 && h_layer_spill_curr != nullptr) {
            std::memcpy(new_curr, h_layer_spill_curr, (size_t)used_curr_bytes);
        }
        if (used_next_bytes > 0 && h_layer_spill_next != nullptr) {
            std::memcpy(new_next, h_layer_spill_next, (size_t)used_next_bytes);
        }
        host_free(h_layer_spill_curr);
        host_free(h_layer_spill_next);
        h_layer_spill_curr = new_curr;
        h_layer_spill_next = new_next;
        host_spill_per_buf_bytes_alloced = new_alloced;
        long long states_ll = host_spill_per_buf_bytes_alloced / bps;
        if (states_ll > (long long)INT_MAX) states_ll = INT_MAX;
        h_layer_spill_per_buf_states = (int)states_ll;
        {
            const char* vp = std::getenv("V5_VERBOSE");
            if (vp && vp[0] != '0') {
                std::fprintf(stderr,
                    "[V5 iter400] grew layer_spill buffers to %.2f GB each "
                    "(cap %.2f GB).\n",
                    (double)host_spill_per_buf_bytes_alloced
                        / (1024.0*1024.0*1024.0),
                    (double)host_spill_per_buf_bytes
                        / (1024.0*1024.0*1024.0));
            }
        }
        return true;
    };

    std::printf("V5 spill (iter370): lazy-alloc, total cap=%.2f GB "
                "(scratch=%.2f GB, layer_spill=%.2f GB x2), staging=%.2f GB "
                "(%d states); SAG_V5_HOST_SPILL_GB env=%s\n",
                (double)host_spill_total_bytes / (1024.0*1024.0*1024.0),
                (double)host_scratch_ring_bytes_cap / (1024.0*1024.0*1024.0),
                (double)host_spill_per_buf_bytes_cap / (1024.0*1024.0*1024.0),
                (double)((long long)spill_chunk_states * bps) / (1024.0*1024.0*1024.0),
                spill_chunk_states,
                std::getenv("SAG_V5_HOST_SPILL_GB") ? std::getenv("SAG_V5_HOST_SPILL_GB") : "(unset)");
    std::printf("V5 spill: varwidth_F_entries=%s\n",
                varwidth_spill_enabled ? "ENABLED" : "disabled");

    // Per-layer spill state.
    int   curr_spill_count_h = 0;
    int   next_spill_count_h = 0;
    long long total_spill_states_seen = 0;
    long long total_spill_bytes_xferd = 0;
    long long total_scratch_spill_states = 0;  // iter370 stat
    long long total_scratch_spill_chunks = 0;  // iter370 stat

    // iter370: per-layer scratch spill ring metadata.
    //   scratch_chunks: list of (host_offset_bytes, count_states, packed_bps,
    //                            max_F_used) tuples.
    //   scratch_ring_used_bytes: running offset into h_scratch_spill.
    // Reset at every layer start.
    //
    // varwidth-spill: packed_bps may be < bps when the chunk's max F_count
    // is < F_MAX_PER_STATE. Pack/unpack at PCIe boundary. When packed_bps==0
    // the chunk has not been packed and uses the dense bps. Default chunks
    // emitted by do_scratch_spill have packed_bps set per chunk-max.
    struct ScratchChunk {
        long long host_offset;
        int count;
        int packed_bps;     // 0 sentinel == dense (bps); else packed stride
        int max_F_used;     // 0 sentinel == not packed; else chunk-wide max F_count
    };
    std::vector<ScratchChunk> scratch_chunks;
    scratch_chunks.reserve(64);
    long long scratch_ring_used_bytes = 0;

    // Topology arrays
    uint64_t *d_TC = nullptr, *d_PO = nullptr, *d_Pred = nullptr, *d_Succ = nullptr;
    V5_CUDA(cudaMalloc(&d_TC,   (size_t)N * W * sizeof(uint64_t)));
    V5_CUDA(cudaMalloc(&d_PO,   (size_t)N * W * sizeof(uint64_t)));
    V5_CUDA(cudaMalloc(&d_Pred, (size_t)N * W * sizeof(uint64_t)));
    V5_CUDA(cudaMalloc(&d_Succ, (size_t)N * W * sizeof(uint64_t)));
    V5_CUDA(cudaMemcpy(d_TC,   topo.TC.data(),   (size_t)N * W * sizeof(uint64_t), cudaMemcpyHostToDevice));
    {
        // Phase 3.2: ECRTS 2022 gang priority-dominance is non-trivial because
        // a higher-priority job with parallelism p_i may not actually be able
        // to dispatch before a lower-priority job j_l with parallelism p_l if
        // p_i + (concurrent dispatches) > m. V5's static PO predicate does
        // not model this and over-constrains gang dispatch (e.g. forces Job 9
        // to wait for Job 7 even when Job 9 fits in idle cores). For ecrts22
        // we therefore zero out PO -- K1 then explores all dispatch orders
        // consistent with precedence + core availability, recovering nptest's
        // schedule semantics. The trade-off is a wider state space which the
        // merge can usually collapse.
        std::vector<uint64_t> po_to_upload = topo.PO;
        if (g_variant == sag::v5::V5Variant::GANG_22) {
            std::fill(po_to_upload.begin(), po_to_upload.end(), 0ULL);
        }
        V5_CUDA(cudaMemcpy(d_PO, po_to_upload.data(),
                           (size_t)N * W * sizeof(uint64_t),
                           cudaMemcpyHostToDevice));
    }
    V5_CUDA(cudaMemcpy(d_Pred, topo.Pred.data(), (size_t)N * W * sizeof(uint64_t), cudaMemcpyHostToDevice));
    V5_CUDA(cudaMemcpy(d_Succ, topo.Succ.data(), (size_t)N * W * sizeof(uint64_t), cudaMemcpyHostToDevice));

    int32_t *d_r_min = nullptr, *d_r_max = nullptr;
    int32_t *d_C_min = nullptr, *d_C_max = nullptr;
    int32_t *d_deadline = nullptr, *d_priority = nullptr, *d_prio_order = nullptr;
    int32_t *d_sus_min = nullptr, *d_sus_max = nullptr;
    int     *d_candidates = nullptr;
    int32_t *d_p_max = nullptr;  // Phase 3 (ECRTS 2022) gang parallelism per job
    int32_t *d_p_min = nullptr;  // Phase 3.5: minimum gang width for ECRTS22 Eq. 9 rho_hp lift
    // Phase 3.3 moldable cost map (CSR). Sizes: offset[N+1], p/cmin/cmax[total].
    int32_t *d_costmap_offset = nullptr;
    int32_t *d_costmap_p      = nullptr;
    int32_t *d_costmap_cmin   = nullptr;
    int32_t *d_costmap_cmax   = nullptr;
    // Phase 3.3.b moldable per-pair side arrays are declared + allocated
    // earlier in main() (right after d_valid_pairs); see the comment there.

    V5_CUDA(cudaMalloc(&d_r_min,     N * sizeof(int32_t)));
    V5_CUDA(cudaMalloc(&d_r_max,     N * sizeof(int32_t)));
    V5_CUDA(cudaMalloc(&d_C_min,     N * sizeof(int32_t)));
    V5_CUDA(cudaMalloc(&d_C_max,     N * sizeof(int32_t)));
    V5_CUDA(cudaMalloc(&d_deadline,  N * sizeof(int32_t)));
    V5_CUDA(cudaMalloc(&d_priority,  N * sizeof(int32_t)));
    V5_CUDA(cudaMalloc(&d_prio_order, N * sizeof(int32_t)));
    V5_CUDA(cudaMalloc(&d_sus_min,   (size_t)N * N * sizeof(int32_t)));
    V5_CUDA(cudaMalloc(&d_sus_max,   (size_t)N * N * sizeof(int32_t)));
    V5_CUDA(cudaMalloc(&d_candidates, (size_t)num_candidates * sizeof(int)));
    V5_CUDA(cudaMalloc(&d_p_max,     N * sizeof(int32_t)));
    V5_CUDA(cudaMalloc(&d_p_min,     N * sizeof(int32_t)));

    V5_CUDA(cudaMemcpy(d_r_min,     topo.r_min.data(),     N * sizeof(int32_t), cudaMemcpyHostToDevice));
    V5_CUDA(cudaMemcpy(d_r_max,     topo.r_max.data(),     N * sizeof(int32_t), cudaMemcpyHostToDevice));
    V5_CUDA(cudaMemcpy(d_C_min,     topo.C_min.data(),     N * sizeof(int32_t), cudaMemcpyHostToDevice));
    V5_CUDA(cudaMemcpy(d_C_max,     topo.C_max.data(),     N * sizeof(int32_t), cudaMemcpyHostToDevice));
    V5_CUDA(cudaMemcpy(d_deadline,  topo.deadline.data(),  N * sizeof(int32_t), cudaMemcpyHostToDevice));
    V5_CUDA(cudaMemcpy(d_priority,  topo.priority.data(),  N * sizeof(int32_t), cudaMemcpyHostToDevice));
    V5_CUDA(cudaMemcpy(d_prio_order, topo.prio_order.data(), N * sizeof(int32_t), cudaMemcpyHostToDevice));
    V5_CUDA(cudaMemcpy(d_sus_min,   topo.sus_min.data(),   (size_t)N * N * sizeof(int32_t), cudaMemcpyHostToDevice));
    V5_CUDA(cudaMemcpy(d_sus_max,   topo.sus_max.data(),   (size_t)N * N * sizeof(int32_t), cudaMemcpyHostToDevice));
    V5_CUDA(cudaMemcpy(d_candidates, topo.candidates.data(),
                       (size_t)num_candidates * sizeof(int), cudaMemcpyHostToDevice));
    V5_CUDA(cudaMemcpy(d_p_max,     topo.p_max.data(),     N * sizeof(int32_t), cudaMemcpyHostToDevice));
    V5_CUDA(cudaMemcpy(d_p_min,     topo.p_min.data(),     N * sizeof(int32_t), cudaMemcpyHostToDevice));

    // Phase 3.3 moldable CSR cost map upload. Allocated and copied even for
    // non-gang variants (synthesized to single-entry from C_min/C_max in
    // v5_build_topology). Allows future K1 to iterate without a host-only
    // fallback path.
    {
        int total_costmap = (int)topo.costmap_p.size();
        if (total_costmap < 1) total_costmap = 1;  // defensive minimum
        V5_CUDA(cudaMalloc(&d_costmap_offset, (size_t)(N + 1) * sizeof(int32_t)));
        V5_CUDA(cudaMalloc(&d_costmap_p,      (size_t)total_costmap * sizeof(int32_t)));
        V5_CUDA(cudaMalloc(&d_costmap_cmin,   (size_t)total_costmap * sizeof(int32_t)));
        V5_CUDA(cudaMalloc(&d_costmap_cmax,   (size_t)total_costmap * sizeof(int32_t)));
        V5_CUDA(cudaMemcpy(d_costmap_offset, topo.costmap_offset.data(),
                           (size_t)(N + 1) * sizeof(int32_t), cudaMemcpyHostToDevice));
        if (!topo.costmap_p.empty()) {
            V5_CUDA(cudaMemcpy(d_costmap_p, topo.costmap_p.data(),
                               (size_t)total_costmap * sizeof(int32_t), cudaMemcpyHostToDevice));
            V5_CUDA(cudaMemcpy(d_costmap_cmin, topo.costmap_cmin.data(),
                               (size_t)total_costmap * sizeof(int32_t), cudaMemcpyHostToDevice));
            V5_CUDA(cudaMemcpy(d_costmap_cmax, topo.costmap_cmax.data(),
                               (size_t)total_costmap * sizeof(int32_t), cudaMemcpyHostToDevice));
        }
        std::printf("Phase 3.3 moldable CSR: %d entries across %d jobs uploaded\n",
                    (int)topo.costmap_p.size(), N);
    }

    // BCRT / WCRT
    int32_t *d_BCRT = nullptr, *d_WCRT = nullptr;
    V5_CUDA(cudaMalloc(&d_BCRT, N * sizeof(int32_t)));
    V5_CUDA(cudaMalloc(&d_WCRT, N * sizeof(int32_t)));
    {
        std::vector<int32_t> h_BCRT(N, INT32_MAX);
        std::vector<int32_t> h_WCRT(N, 0);
        V5_CUDA(cudaMemcpy(d_BCRT, h_BCRT.data(), N * sizeof(int32_t), cudaMemcpyHostToDevice));
        V5_CUDA(cudaMemcpy(d_WCRT, h_WCRT.data(), N * sizeof(int32_t), cudaMemcpyHostToDevice));
    }

    // Per-layer scalar counters / flags. Packed into a single contiguous
    // device int array so the K1+K2 D2H readback (output/unsched/trunc) can
    // batch into one cudaMemcpyAsync (12 bytes) instead of three 4-byte
    // copies per wave. Layout indices (matching h_pin_scratch slot semantics
    // where useful):
    //   [0] = output_count  (K2 output count, also h_pin_scratch[0])
    //   [1] = unsched       (K2 unsched flag, also h_pin_scratch[1])
    //   [2] = trunc         (truncation flag, also h_pin_scratch[2])
    //   [3] = num_groups    (post-merge boundary count, h_pin_scratch[3])
    //   [4] = merge_count   (post-merge slot count)
    //   [5] = valid_count   (K1 valid pair count)
    //   [6] = curr_count    (num_states_in for K1)
    //   [7] = max_F_count   (diagnostic: max popcount(F_mask|X) observed)
    constexpr int N_COUNTERS = 8;
    int* d_counters = nullptr;
    V5_CUDA(cudaMalloc(&d_counters, N_COUNTERS * sizeof(int)));
    V5_CUDA(cudaMemset(d_counters, 0, N_COUNTERS * sizeof(int)));
    int* d_output_count = d_counters + 0;
    int* d_unsched      = d_counters + 1;
    int* d_trunc        = d_counters + 2;
    int* d_num_groups   = d_counters + 3;
    int* d_merge_count  = d_counters + 4;
    int* d_valid_count  = d_counters + 5;
    int* d_curr_count   = d_counters + 6;
    int* d_max_F_count  = d_counters + 7;

    // varwidth-spill: dedicated counter for per-chunk max F_count measurement.
    // Distinct from d_max_F_count (the K1/K2 sticky run-wide diagnostic) so
    // measuring a chunk doesn't corrupt the global stats.
    int* d_chunk_max_F = nullptr;
    V5_CUDA(cudaMalloc(&d_chunk_max_F, sizeof(int)));
    V5_CUDA(cudaMemset(d_chunk_max_F, 0, sizeof(int)));

    // POR (Partial-Order Reduction) Phase A scaffold (RTAS 2022 / RT Sys 2023):
    // SAG_V5_POR=1 enables the optional POR pre-pass that, in a future
    // session's Phase B+, will fuse multiple in-ready candidates per
    // state into a single multi-job dispatch (Algorithm 4 greedy
    // interferer absorption). Phase A: env var parse + status banner +
    // reserve d_por_dbg counters. No kernel-body change yet -- verdicts
    // are unchanged when SAG_V5_POR=0 (default) AND when =1 (predicate
    // not yet implemented). Memory v5_por_algorithm4_estimate.md +
    // iter150_por_basic_zero_fires.md document the algorithmic background.
    bool por_enabled = false;
    if (const char* p = std::getenv("SAG_V5_POR")) {
        por_enabled = (p[0] != '\0' && p[0] != '0');
    }
    if (por_enabled) {
        std::printf("V5 POR: SAG_V5_POR=1 -- Phase B observation enabled "
                    "(counts states with |in_ready|>=2 = P1 predicate). "
                    "Phase B-E full predicate + emission deferred. "
                    "See PHASE_POR_PLAN.md for the roadmap.\n");
        V5_set_por_observe(1);
        V5_clear_por_dbg();
    }

    // iter410: IJP detection toggle. Default comes from the variant tricks
    // table (g_tricks.ijp_default); SAG_V5_IJP overrides it as a back-door
    // for benchmarking / regression testing of the rtss24 variant. Other
    // variants always have ijp off (they short-circuited above in Phase 1
    // anyway; in Phases 2-4 the variant tricks table will be authoritative).
    bool ijp_enabled = g_tricks.ijp_default;
    if (const char* p = std::getenv("SAG_V5_IJP")) {
        bool overridden = (p[0] != '\0' && p[0] != '0');
        if (overridden != ijp_enabled) {
            std::printf("V5 IJP: SAG_V5_IJP back-door override -> %s "
                        "(variant default was %s)\n",
                        overridden ? "ENABLED" : "disabled",
                        ijp_enabled ? "ENABLED" : "disabled");
        }
        ijp_enabled = overridden;
    }
    IJPInfoV5* d_ijp_info = nullptr;
    if (ijp_enabled) {
        V5_CUDA(cudaMalloc(&d_ijp_info,
                          (size_t)cap.wave_states * sizeof(IJPInfoV5)));
        V5_CUDA(cudaMemset(d_ijp_info, 0,
                          (size_t)cap.wave_states * sizeof(IJPInfoV5)));
        std::printf("V5 IJP: ENABLED (IJPInfoV5 buf=%.2f MB)\n",
                    (double)cap.wave_states * sizeof(IJPInfoV5) / (1024.0*1024.0));
    } else {
        std::printf("V5 IJP: disabled\n");
    }

    // Merge scratch buffers (iter370).
    //
    // Sized to the maximum the layer-end merge will ever sort. Three cases:
    //   - No-spill path: input is d_layer_scratch[0..S), S <= scratch_capacity.
    //   - Spill path (single buffer): input is d_aggregator[0..total),
    //     total <= aggregator_capacity.
    //   - Spill path (dual source): input is d_aggregator (first agg_cap)
    //     + d_layer_scratch (overflow up to scratch_capacity); total <=
    //     aggregator_capacity + scratch_capacity.
    // iter390: triple-source consolidate adds d_curr (size = layer_capacity)
    //     as a 3rd input, so the sort must index up to
    //     aggregator_capacity + scratch_capacity + layer_capacity states.
    // Plus per-wave sort uses the same buffers (size <= wave_max_output).
    long long merge_scratch_max_ll =
        (long long)cap.scratch_capacity + (long long)cap.aggregator_capacity
        + (long long)cap.layer_capacity;
    if ((long long)cap.wave_max_output > merge_scratch_max_ll)
        merge_scratch_max_ll = cap.wave_max_output;
    if (merge_scratch_max_ll > (long long)INT_MAX) merge_scratch_max_ll = INT_MAX;
    int merge_scratch_max = (int)merge_scratch_max_ll;

    uint64_t* d_dkeys_in   = nullptr;
    uint64_t* d_dkeys_out  = nullptr;
    int*      d_idx_in     = nullptr;
    int*      d_idx_out    = nullptr;
    int*      d_is_start   = nullptr;
    int*      d_group_id   = nullptr;
    int*      d_group_starts = nullptr;
    V5_CUDA(cudaMalloc(&d_dkeys_in,    (size_t)merge_scratch_max * W * sizeof(uint64_t)));
    V5_CUDA(cudaMalloc(&d_dkeys_out,   (size_t)merge_scratch_max * W * sizeof(uint64_t)));
    V5_CUDA(cudaMalloc(&d_idx_in,      (size_t)merge_scratch_max * sizeof(int)));
    V5_CUDA(cudaMalloc(&d_idx_out,     (size_t)merge_scratch_max * sizeof(int)));
    V5_CUDA(cudaMalloc(&d_is_start,    (size_t)merge_scratch_max * sizeof(int)));
    V5_CUDA(cudaMalloc(&d_group_id,    (size_t)merge_scratch_max * sizeof(int)));
    V5_CUDA(cudaMalloc(&d_group_starts,(size_t)merge_scratch_max * sizeof(int)));

    // CUB temp storage size queries (max of sort + scan, sized to the largest
    // input we will ever sort, i.e. merge_scratch_max for the cross-wave global
    // merge pass).
    //
    // For W=1 we use cub::DeviceRadixSort::SortPairs(end_bit=N) -- the
    // free 2x speedup from memory iter121. For W>1 we fall back to
    // cub::DeviceMergeSort::SortKeys with a custom W-word lex-compare
    // (mirrors legacy at expand_test_main.cu:3910).
    size_t sort_temp_bytes = 0;
    size_t scan_temp_bytes = 0;
    if (W == 1) {
        cub::DeviceRadixSort::SortPairs(
            nullptr, sort_temp_bytes,
            d_dkeys_in, d_dkeys_out,
            d_idx_in,   d_idx_out,
            merge_scratch_max, 0, N);
    } else {
        const uint64_t* dk_q = d_dkeys_in;
        int             lw_q = W;
        cub::DeviceMergeSort::SortKeys(
            nullptr, sort_temp_bytes,
            d_idx_in, merge_scratch_max,
            [dk_q, lw_q] __device__ (const int& a, const int& b) {
                for (int w = 0; w < lw_q; w++) {
                    if (dk_q[a * lw_q + w] < dk_q[b * lw_q + w]) return true;
                    if (dk_q[a * lw_q + w] > dk_q[b * lw_q + w]) return false;
                }
                return false;
            });
    }
    cub::DeviceScan::ExclusiveSum(
        nullptr, scan_temp_bytes,
        d_is_start, d_group_id,
        merge_scratch_max);
    size_t cub_temp_bytes = (sort_temp_bytes > scan_temp_bytes)
                            ? sort_temp_bytes : scan_temp_bytes;
    void* d_cub_temp = nullptr;
    V5_CUDA(cudaMalloc(&d_cub_temp, cub_temp_bytes));

    int max_layers = N + 4;
    // Debug override: SAG_V5_MAX_LAYERS allows extending the layer cap to
    // diagnose convergence issues with mid-layer flush.
    if (const char* p = std::getenv("SAG_V5_MAX_LAYERS")) {
        int v = std::atoi(p);
        if (v > 0) max_layers = v;
    }

    // ----- Cooperative-launch grid sizing -----
    int dev_id = 0;
    V5_CUDA(cudaGetDevice(&dev_id));
    int coop_supported = 0;
    V5_CUDA(cudaDeviceGetAttribute(&coop_supported, cudaDevAttrCooperativeLaunch, dev_id));
    if (!coop_supported) {
        fprintf(stderr, "Error: cooperative launch unsupported on device %d\n", dev_id);
        return 4;
    }
    int num_sms = 0;
    V5_CUDA(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id));

    // L2 persisting cache (SM_80+; older SMs return 0 = no-op fallback). The
    // merge predicate's slot-state reads dominate the hot path on n>=24
    // workloads; pinning a slice of d_layer_scratch in L2 reduces re-fetch
    // latency. Set on the default stream (stream 0) since V5 uses
    // per-thread default stream for K1+K2+merge launches. d_layer_scratch
    // was allocated earlier (line ~1392) so we can pin a slice of it.
    {
        int persisting_max = 0;
        cudaDeviceGetAttribute(&persisting_max,
                               cudaDevAttrMaxPersistingL2CacheSize, dev_id);
        if (persisting_max > 0 && d_layer_scratch != nullptr) {
            cudaCtxResetPersistingL2Cache();
            size_t window_bytes = persisting_max;
            if (window_bytes > (size_t)16 * 1024 * 1024) {
                window_bytes = (size_t)16 * 1024 * 1024;
            }
            cudaStreamAttrValue attr = {};
            attr.accessPolicyWindow.base_ptr = d_layer_scratch;
            attr.accessPolicyWindow.num_bytes = window_bytes;
            attr.accessPolicyWindow.hitRatio = 1.0f;
            attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
            attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
            cudaError_t err = cudaStreamSetAttribute(
                /*stream=*/0,
                cudaStreamAttributeAccessPolicyWindow,
                &attr);
            if (err == cudaSuccess) {
                std::printf("V5 L2 persisting cache: device max=%d MiB; "
                            "%.1f MiB pinned on d_layer_scratch (stream 0)\n",
                            persisting_max / (1024 * 1024),
                            window_bytes / (1024.0 * 1024.0));
            }
        }
    }

    size_t k1k2_smem = V5_smem_bytes(N, W, M);
    int k1k2_block_size = V5_block_size();
    // Pick the kernel pointer based on (variant, IJP). Phase 1 only reaches
    // here for NPG_RTSS24 (others short-circuited at startup), so the
    // variant arms are dead in Phase 1 but pre-wired for Phase 2-4.
    // All variant kernels share the same smem layout (the IJP NPG24 variant
    // adds register-only state in lane 0 for the detect phase).
    const void* k1k2_kernel_ptr = nullptr;
    switch (g_variant) {
        case sag::v5::V5Variant::NPG_RTSS24:
            k1k2_kernel_ptr = ijp_enabled ? V5_k1k2_kernel_ijp_ptr()
                                          : V5_k1k2_kernel_ptr();
            break;
        case sag::v5::V5Variant::NP_UNI_17:
            k1k2_kernel_ptr = V5_k1k2_kernel_rtss17_ptr();
            break;
        case sag::v5::V5Variant::LP_DAG_19:
            k1k2_kernel_ptr = V5_k1k2_kernel_ecrts19_ptr();
            break;
        case sag::v5::V5Variant::GANG_22:
            // Route all-rigid-p=1 ECRTS22 to rtss24's kernel (math-equivalent,
            // skips ECRTS22-specific side-array overhead).
            k1k2_kernel_ptr = g_ecrts22_route_to_rtss24
                              ? V5_k1k2_kernel_ptr()
                              : V5_k1k2_kernel_ecrts22_ptr();
            break;
    }
    int k1k2_max_active = 0;
    V5_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &k1k2_max_active, k1k2_kernel_ptr, k1k2_block_size, k1k2_smem));
    if (k1k2_max_active < 1) {
        fprintf(stderr, "Error: V5K1K2Kernel cannot fit even one block per SM "
                "at block_size=%d, smem=%zu (variant=%s, IJP=%d).\n",
                k1k2_block_size, k1k2_smem,
                sag::v5::v5_variant_name(g_variant), (int)ijp_enabled);
        return 4;
    }
    int k1k2_grid = k1k2_max_active * num_sms;

    int merge_block_size = V5_merge_block_size();
    int merge_warps_per_block = V5_merge_warps_per_block();
    size_t merge_smem = V5_merge_smem_bytes();

    std::printf("V5 K1+K2 cooperative launch: grid=%d (max_active=%d/SM x %d SMs), "
                "block=%d, smem=%zu B\n",
                k1k2_grid, k1k2_max_active, num_sms, k1k2_block_size, k1k2_smem);
    std::printf("V5 merge launch: block=%d (%d warps/block), smem=%zu B\n",
                merge_block_size, merge_warps_per_block, merge_smem);

    // ----- Stage initial state -----
    // d_layerA[0] is already cudaMemset to all-zero.
    int curr_count_h = 1;

    // Pinned host scratch for status read-back (4 bytes per layer).
    int* h_pin_scratch = nullptr;
    V5_CUDA(cudaMallocHost(&h_pin_scratch, 4 * sizeof(int)));
    // h_pin_scratch[0] = output_count
    // h_pin_scratch[1] = unsched_flag
    // h_pin_scratch[2] = trunc_flag
    // h_pin_scratch[3] = num_groups (after merge)

    cudaEvent_t ev_start, ev_stop;
    V5_CUDA(cudaEventCreate(&ev_start));
    V5_CUDA(cudaEventCreate(&ev_stop));
    V5_CUDA(cudaEventRecord(ev_start, 0));

    int32_t status = (int32_t)V5Status::UNKNOWN;
    int     layers_done = 0;

    // Layer loop with INNER WAVE LOOP + MID-LAYER FLUSH (iter350):
    //
    // For each layer L -> L+1:
    //   1. Reset per-layer counters (cwm = cross_wave_merged so far).
    //   2. INNER WAVE LOOP -- expand parents:
    //      For each wave of cap.wave_states parents from d_curr:
    //        * cooperative-launch V5K1K2Kernel reading the wave's slice of
    //          d_curr, writing K2 output to d_wave_output.
    //        * Per-wave sort + group + merge into d_layer_scratch at offset
    //          total_pre_cw_merged.
    //        * total_pre_cw_merged += wave_merged.
    //        * If total_pre_cw_merged + projected_next_wave_output > scratch_cap:
    //            FLUSH: dual-source merge (scratch + d_next[0..cwm)) -> d_temp
    //                   swap d_next <-> d_temp; cwm = new merge count.
    //                   Reset total_pre_cw_merged = 0.
    //   3. After wave loop: FINAL FLUSH (same as mid-layer flush) consumes
    //      d_layer_scratch + d_next[0..cwm) -> d_temp, then swap.
    //      If cwm == 0 and total_pre_cw_merged > 0, this is just a single-source
    //      cross-wave merge (no dual source).
    //   4. Swap d_curr <-> d_next; curr_count_h := final cwm.
    //
    // Memory budget:
    //   d_layerA, d_layerB, d_layer_temp: 3 * layer_capacity * bps
    //   d_layer_scratch: scratch_capacity * bps  (small, default ~10% of avail)
    //
    // FAIL-LOUD: every overflow path printf's its context and sets V5Status::TRUNC.
    char* d_curr = d_layerA;
    char* d_next = d_layerB;
    // iter370: no d_temp -- merge writes directly to d_next.

    const char* verbose_env = std::getenv("SAG_V5_VERBOSE");
    bool verbose = (verbose_env && verbose_env[0] != '0');

    // iter370 layer-end merge helper (single-source).
    //
    // Sorts d_layer_scratch[0..S) by D-key, detects groups, runs V5MergeKernel
    // batched, with output spilling into h_layer_spill_next when n_groups
    // exceeds cap.layer_capacity. d_next ends up with the bottom layer_cap
    // states; h_layer_spill_next has the rest.
    //
    // This is the SINGLE-SOURCE variant used in:
    //   * Layer-end merge when no scratch ring spill happened (cwm == 0).
    //   * Layer-end merge when spill happened: invoked via the consolidation
    //     wrapper after staging chunks back into d_aggregator/d_layer_scratch.
    //
    // Inputs:
    //   d_pre_buf   = source state buffer (d_layer_scratch or d_aggregator)
    //   layer       (debug, current layer index)
    //   S           = states currently in d_pre_buf
    //   final       = whether this is the final flush of the layer (verbose only)
    //
    // Outputs (via reference):
    //   new_cwm     = the merge count after dedup (≤ layer_cap; spill held
    //                 separately in next_spill_count_h)
    auto do_flush = [&](int layer, char* d_pre_buf, int S, bool final,
                        int& new_cwm) -> bool {
        if (S <= 0) {
            new_cwm = 0;
            return true;
        }
        if ((long long)S > merge_scratch_max) {
            std::fprintf(stderr,
                "[V5 TRUNC] layer=%d %s flush: input %d exceeds "
                "merge_scratch_max=%d. FAIL LOUD.\n",
                layer, final ? "final" : "consolidate", S, merge_scratch_max);
            return false;
        }
        int combined = S;

        // Step 1: Extract D-keys + iota indices from d_pre_buf[0..combined).
        {
            int blocks = (combined + 255) / 256;
            if (blocks < 1) blocks = 1;
            V5_CUDA(cudaMemcpyAsync(d_curr_count, &combined, sizeof(int),
                                    cudaMemcpyHostToDevice));
            V5ExtractDKeysIotaKernel<<<blocks, 256>>>(
                d_pre_buf, layout, d_curr_count, W,
                d_dkeys_in, d_idx_in);
        }

        // Step 2: Sort by D-key. The sorted indices end up in d_idx_out.
        if (W == 1) {
            size_t tb = sort_temp_bytes;
            cub::DeviceRadixSort::SortPairs(
                d_cub_temp, tb,
                d_dkeys_in, d_dkeys_out,
                d_idx_in,   d_idx_out,
                combined, 0, N);
        } else {
            V5_CUDA(cudaMemcpyAsync(d_idx_out, d_idx_in,
                                    combined * sizeof(int),
                                    cudaMemcpyDeviceToDevice));
            const uint64_t* dk_q = d_dkeys_in;
            int             lw_q = W;
            size_t tb = sort_temp_bytes;
            cub::DeviceMergeSort::SortKeys(
                d_cub_temp, tb,
                d_idx_out, combined,
                [dk_q, lw_q] __device__ (const int& a, const int& b) {
                    for (int w = 0; w < lw_q; w++) {
                        if (dk_q[a * lw_q + w] < dk_q[b * lw_q + w]) return true;
                        if (dk_q[a * lw_q + w] > dk_q[b * lw_q + w]) return false;
                    }
                    return false;
                });
        }

        // Step 3: Detect group boundaries.
        V5_CUDA(cudaMemcpyAsync(d_curr_count, &combined, sizeof(int),
                                cudaMemcpyHostToDevice));
        {
            int blocks = (combined + 255) / 256;
            if (blocks < 1) blocks = 1;
            V5DetectBoundariesKernel<<<blocks, 256>>>(
                d_dkeys_in, d_idx_out, d_curr_count, W, d_is_start);
        }

        // Step 4: ExclusiveSum group ids.
        {
            size_t sb = scan_temp_bytes;
            cub::DeviceScan::ExclusiveSum(
                d_cub_temp, sb,
                d_is_start, d_group_id,
                combined);
        }

        // Step 5: Compact group_starts + d_num_groups.
        V5_CUDA(cudaMemsetAsync(d_num_groups, 0, sizeof(int)));
        {
            int blocks = (combined + 255) / 256;
            if (blocks < 1) blocks = 1;
            V5CompactGroupStartsKernel<<<blocks, 256>>>(
                d_is_start, d_group_id, d_curr_count,
                d_group_starts, d_num_groups);
        }

        V5_CUDA(cudaMemcpyAsync(&h_pin_scratch[3], d_num_groups,
                                sizeof(int), cudaMemcpyDeviceToHost));
        V5_CUDA(cudaStreamSynchronize(0));
        int n_groups = h_pin_scratch[3];
        if (n_groups < 1) {
            std::fprintf(stderr,
                "[V5 TRUNC] layer=%d %s flush: 0 groups despite combined=%d. "
                "FAIL LOUD.\n", layer, final ? "final" : "consolidate", combined);
            return false;
        }

        // Step 6: iter360 batched merge with DRAM spill.
        //
        // n_groups can exceed cap.layer_capacity for hard cases (n24/t000 hits
        // ~12.97M groups vs layer_cap=12.5M). The merge is split into batches.
        // Each batch processes a contiguous range of groups [start, start+sz)
        // from d_group_starts.
        //
        // Per-batch destination logic:
        //   * The bottom layer_cap states stay GPU-resident in d_temp.
        //   * Excess states go to host-pinned h_layer_spill_next via the
        //     d_spill_staging GPU buffer (DMA'd between batches).
        //
        // Batch sizing: each batch's worst-case output is bounded above by
        // batch_groups * MAX_SLOTS_PER_GROUP. For typical workloads the actual
        // emit count is much smaller (≤2 slots/group on average). To stay
        // safe, we pick batch_groups so worst-case output fits in the
        // destination buffer.
        //
        // d_temp is layer_capacity states; first batch uses up to layer_cap
        // groups. d_spill_staging is spill_chunk_states; subsequent batches
        // use up to spill_chunk_states groups.

        const int slots_per_group_safe_factor = 8;  // empirical cap << 64
        // For the d_temp destination, bound the batch by remaining d_temp room.
        // For d_spill_staging destination, bound by spill_chunk_states.

        // iter370: merge writes directly to d_next (no separate d_temp).
        // Optional debug: zero d_next before each merge.
        if (const char* p = std::getenv("SAG_V5_ZERO_TEMP")) {
            if (p[0] != '0') {
                V5_CUDA(cudaMemsetAsync(d_next, 0,
                                        (size_t)cap.layer_capacity * bps));
            }
        }

        long long total_merged_ll = 0;       // running merged count across batches
        long long resident_count_ll = 0;     // states placed in d_next
        long long spilled_count_ll = 0;      // states placed in h_layer_spill_next this flush
        // Initialize next_spill_count_h on the first call to do_flush in this
        // layer; further flushes accumulate. (Caller is responsible for
        // resetting next_spill_count_h = 0 at layer start.)

        int g_processed = 0;
        bool need_spill = ((long long)n_groups > cap.layer_capacity);
        if (need_spill) {
            if (!ensure_spill_initialized()) {
                std::fprintf(stderr,
                    "[V5 TRUNC] layer=%d %s flush: n_groups=%d > layer_cap=%d "
                    "but host spill buffer unavailable. Set "
                    "SAG_V5_HOST_SPILL_GB=N to enable. FAIL LOUD.\n",
                    layer, final ? "final" : "consolidate", n_groups,
                    cap.layer_capacity);
                return false;
            }
        }

        while (g_processed < n_groups) {
            // Decide destination + batch size.
            //
            // - Resident phase (d_temp not yet at layer_cap): write to
            //     d_temp + resident_count_ll * bps
            //   so the kernel's atomicAdd-based slots land at offset
            //   [resident_count_ll, resident_count_ll + batch_merged).
            // - Spill phase: write to d_spill_staging at offset 0; after the
            //   batch, async-DMA the bytes to h_layer_spill_next.
            char* dest_buf = nullptr;
            long long dest_capacity_states = 0;   // hard upper bound on this batch's output
            bool dest_is_spill = false;

            // Transition from resident to spill if the remaining d_next room
            // is less than slots_per_group_safe_factor (smallest safe batch).
            // Otherwise we'd risk a single-group batch overflowing d_next.
            if (resident_count_ll < cap.layer_capacity &&
                (long long)cap.layer_capacity - resident_count_ll
                    >= slots_per_group_safe_factor) {
                // d_next still has room.
                dest_buf = d_next + resident_count_ll * (long long)bps;
                dest_capacity_states = (long long)cap.layer_capacity - resident_count_ll;
            } else {
                // d_next full -- spill destination.
                dest_buf = d_spill_staging;
                dest_capacity_states = spill_chunk_states;
                dest_is_spill = true;
                if (!dest_buf) {
                    std::fprintf(stderr,
                        "[V5 TRUNC] layer=%d %s flush: spill destination not "
                        "allocated (alloc failed earlier). FAIL LOUD.\n",
                        layer, final ? "final" : "consolidate");
                    return false;
                }
                // Bound batch by remaining h_layer_spill_next room.
                // iter400: try to grow ping-pong if at allocation limit.
                long long remaining_host =
                    (long long)h_layer_spill_per_buf_states
                    - (long long)next_spill_count_h - spilled_count_ll;
                if (remaining_host < 1
                    && host_spill_per_buf_bytes_alloced
                       < host_spill_per_buf_bytes) {
                    long long min_needed =
                        ((long long)next_spill_count_h + spilled_count_ll
                         + 1LL) * (long long)bps;
                    long long curr_bytes_used =
                        (long long)curr_spill_count_h * (long long)bps;
                    long long next_bytes_used =
                        ((long long)next_spill_count_h + spilled_count_ll)
                            * (long long)bps;
                    V5_CUDA(cudaStreamSynchronize(spill_stream));
                    (void)grow_layer_spill_buffers(
                        min_needed, curr_bytes_used, next_bytes_used);
                    remaining_host =
                        (long long)h_layer_spill_per_buf_states
                        - (long long)next_spill_count_h - spilled_count_ll;
                }
                if (remaining_host < 1) {
                    std::fprintf(stderr,
                        "[V5 TRUNC] layer=%d %s flush: host spill buffer "
                        "exhausted (per-buf alloc %lld / cap %lld bytes; %d "
                        "states alloc, used %lld). "
                        "Increase SAG_V5_HOST_SPILL_GB. FAIL LOUD.\n",
                        layer, final ? "final" : "consolidate",
                        host_spill_per_buf_bytes_alloced,
                        host_spill_per_buf_bytes,
                        h_layer_spill_per_buf_states,
                        (long long)next_spill_count_h + spilled_count_ll);
                    return false;
                }
                if (remaining_host < dest_capacity_states)
                    dest_capacity_states = remaining_host;
            }

            // Batch size in groups: dest_capacity / slots_per_group_safe_factor.
            int batch_groups = (int)(dest_capacity_states /
                                     (long long)slots_per_group_safe_factor);
            if (batch_groups < 1) batch_groups = 1;
            if (batch_groups > n_groups - g_processed)
                batch_groups = n_groups - g_processed;

            // Wait for any prior spill DMA to finish reading d_spill_staging
            // before we overwrite it (only matters for the spill phase).
            if (dest_is_spill) {
                V5_CUDA(cudaStreamSynchronize(spill_stream));
            }

            // Run merge for this batch. Source is the buffer passed by the
            // caller (d_pre_buf -- d_layer_scratch for the no-spill path or
            // d_aggregator for the spill consolidation path).
            V5_CUDA(cudaMemsetAsync(d_merge_count, 0, sizeof(int)));
            {
                int merge_grid =
                    (batch_groups + merge_warps_per_block - 1)
                        / merge_warps_per_block;
                if (merge_grid < 1) merge_grid = 1;
                int dest_cap_int = (dest_capacity_states > (long long)INT_MAX)
                                   ? INT_MAX : (int)dest_capacity_states;
                V5_LAUNCH_MERGE(merge_grid, merge_block_size, merge_smem,
                    d_pre_buf, /*d_pre2=*/(const char*)nullptr,
                    /*pre1_count=*/0,
                    /*d_pre3=*/(const char*)nullptr,
                    /*pre2_end=*/0,
                    d_idx_out, d_group_starts,
                    g_processed, batch_groups, n_groups,
                    d_curr_count,
                    layout, N, M, W,
                    dest_buf, dest_cap_int,
                    d_merge_count, d_trunc, layer);
            }

            V5_CUDA(cudaMemcpyAsync(&h_pin_scratch[0], d_merge_count,
                                    sizeof(int), cudaMemcpyDeviceToHost));
            V5_CUDA(cudaMemcpyAsync(&h_pin_scratch[2], d_trunc,
                                    sizeof(int), cudaMemcpyDeviceToHost));
            V5_CUDA(cudaStreamSynchronize(0));
            int batch_merged = h_pin_scratch[0];
            int batch_tf     = h_pin_scratch[2];
            if (batch_merged < 0) batch_merged = 0;

            if (batch_tf != 0) {
                std::fprintf(stderr,
                    "[V5 TRUNC] layer=%d %s flush batch g=%d/%d: trunc flag "
                    "(F_count overflow likely). FAIL LOUD.\n",
                    layer, final ? "final" : "consolidate", g_processed, n_groups);
                return false;
            }
            if ((long long)batch_merged > dest_capacity_states) {
                std::fprintf(stderr,
                    "[V5 TRUNC] layer=%d %s flush batch g=%d sz=%d: emitted=%d "
                    "exceeds dest_capacity=%lld. Some group has > %d slots; "
                    "reduce slots_per_group_safe_factor. FAIL LOUD.\n",
                    layer, final ? "final" : "consolidate",
                    g_processed, batch_groups,
                    batch_merged, dest_capacity_states,
                    slots_per_group_safe_factor);
                return false;
            }

            // Move the batch's output to its final home.
            if (dest_is_spill) {
                // d_spill_staging[0..batch_merged) -> h_layer_spill_next at
                // (next_spill_count_h + spilled_count_ll) * bps.
                long long h_offset_states =
                    (long long)next_spill_count_h + spilled_count_ll;
                cudaError_t ce = cudaMemcpyAsync(
                    h_layer_spill_next + h_offset_states * bps,
                    d_spill_staging,
                    (size_t)batch_merged * bps,
                    cudaMemcpyDeviceToHost,
                    spill_stream);
                if (ce != cudaSuccess) {
                    std::fprintf(stderr,
                        "[V5 TRUNC] layer=%d %s flush spill D2H copy failed: "
                        "%s. FAIL LOUD.\n",
                        layer, final ? "final" : "consolidate",
                        cudaGetErrorString(ce));
                    return false;
                }
                spilled_count_ll += batch_merged;
                total_spill_bytes_xferd += (long long)batch_merged * bps;
            } else {
                resident_count_ll += batch_merged;
            }

            g_processed += batch_groups;
            total_merged_ll += batch_merged;
        }

        int total_merged = (int)total_merged_ll;
        int resident_merged = (int)resident_count_ll;
        // Wait for all spill copies to complete before we proceed.
        if (spill_stream != 0) {
            V5_CUDA(cudaStreamSynchronize(spill_stream));
        }

        // Update layer-level spill state.
        next_spill_count_h += (int)spilled_count_ll;
        total_spill_states_seen += spilled_count_ll;

        // iter370: merge wrote directly to d_next; no swap needed.

        if (verbose) {
            std::fprintf(stderr,
                "[layer %d] %s merge: S=%d combined=%d groups=%d "
                "total_merged=%d (resident=%d spill=%lld) (dedup %.2fx)\n",
                layer, final ? "FINAL" : "consolidate",
                S, combined, n_groups, total_merged,
                resident_merged, spilled_count_ll,
                (double)combined / (double)(total_merged > 0 ? total_merged : 1));
        }

        // new_cwm represents the RESIDENT (in d_next) merged count, bounded
        // above by layer_cap. The spill portion is held in next_spill_count_h.
        new_cwm = resident_merged;
        return true;
    };

    // -----------------------------------------------------------------
    // varwidth-spill helpers.
    //
    // measure_chunk_max_F: launch V5ChunkMaxFCountKernel on a GPU buffer
    //                      [src .. src + count*bps). Returns the chunk's
    //                      max F_count (host-side int).
    // d2h_pack_chunk     : pack `count` states from d_src (dense layout)
    //                      through d_spill_staging in batches; D2H to
    //                      h_dst at packed stride. d_src + count*bps must
    //                      not overlap d_spill_staging.
    // h2d_unpack_chunk   : H2D `count*packed_bps` bytes from h_src to
    //                      d_spill_staging in batches; unpack to d_dst at
    //                      dense stride.
    //
    // The "in batches" pattern is required because d_spill_staging is sized
    // to spill_chunk_states * bps (~512MB / bps states), which can be smaller
    // than full chunks (which can reach scratch_capacity states).
    // -----------------------------------------------------------------
    auto measure_chunk_max_F = [&](const char* d_src, int count) -> int {
        if (count <= 0) return 0;
        V5_CUDA(cudaMemsetAsync(d_chunk_max_F, 0, sizeof(int)));
        int blocks = (count + 255) / 256;
        if (blocks < 1) blocks = 1;
        V5ChunkMaxFCountKernel<<<blocks, 256>>>(
            d_src, layout, count, d_chunk_max_F);
        int h_max = 0;
        V5_CUDA(cudaMemcpyAsync(&h_max, d_chunk_max_F,
                                sizeof(int), cudaMemcpyDeviceToHost));
        V5_CUDA(cudaStreamSynchronize(0));
        if (h_max < 0) h_max = 0;
        if (h_max > sag::v5::F_MAX_PER_STATE) {
            // Should never happen (kernel sticky-checks F_count overflow).
            // Defensive clamp -- but this is a TRUNC path.
            h_max = sag::v5::F_MAX_PER_STATE;
        }
        if (verbose) {
            std::fprintf(stderr,
                "[varwidth] measure: count=%d max_F=%d packed_bps=%d "
                "(dense=%d savings=%.1f%%)\n",
                count, h_max, V5_packed_bps(W, M, h_max), bps,
                100.0 * (double)(bps - V5_packed_bps(W, M, h_max))
                       / (double)bps);
        }
        return h_max;
    };

    // d2h_pack_chunk: pack & D2H. Returns true on success.
    // Caller must ensure (h_dst .. h_dst + count*packed_bps) is reserved
    // and packed_bps already chosen via V5_packed_bps(W,M,max_F_used).
    //
    // Per-batch ordering:
    //   1. Wait for prior D2H (so we can overwrite d_spill_staging).
    //   2. Launch pack kernel on default stream (writes d_spill_staging).
    //   3. Sync stream 0 (so D2H sees pack output).
    //   4. D2H on spill_stream.
    // After loop, sync spill_stream so caller sees the host bytes.
    auto d2h_pack_chunk = [&](const char* d_src, int count,
                              int packed_bps_in, int max_F_used,
                              char* h_dst) -> bool {
        if (count <= 0) return true;
        int B_max = spill_chunk_states;
        for (int batch_start = 0; batch_start < count; batch_start += B_max) {
            int B = (batch_start + B_max <= count) ? B_max : (count - batch_start);
            // Wait for prior batch's D2H (it reads d_spill_staging).
            if (batch_start > 0) {
                V5_CUDA(cudaStreamSynchronize(spill_stream));
            }
            int warps_needed = B;
            int blocks =
                (warps_needed + VARWIDTH_WARPS_PER_BLOCK - 1)
                / VARWIDTH_WARPS_PER_BLOCK;
            if (blocks < 1) blocks = 1;
            const char* src_ptr = d_src + (long long)batch_start * bps;
            V5PackForSpillKernel<<<blocks, VARWIDTH_THREADS_PER_BLOCK>>>(
                src_ptr, layout,
                d_spill_staging, packed_bps_in, max_F_used,
                B, W, M);
            // Sync stream 0 so the pack kernel finishes before D2H reads
            // d_spill_staging.
            V5_CUDA(cudaStreamSynchronize(0));
            long long batch_bytes = (long long)B * (long long)packed_bps_in;
            long long h_off = (long long)batch_start * (long long)packed_bps_in;
            cudaError_t ce = cudaMemcpyAsync(
                h_dst + h_off, d_spill_staging,
                (size_t)batch_bytes, cudaMemcpyDeviceToHost,
                spill_stream);
            if (ce != cudaSuccess) {
                std::fprintf(stderr,
                    "[V5 TRUNC] d2h_pack_chunk: D2H batch %d (B=%d, %lld bytes) "
                    "failed: %s. FAIL LOUD.\n",
                    batch_start, B, batch_bytes, cudaGetErrorString(ce));
                return false;
            }
        }
        // Final sync so the host bytes are visible on caller's return.
        V5_CUDA(cudaStreamSynchronize(spill_stream));
        return true;
    };

    // h2d_unpack_chunk: H2D & unpack. Returns true on success.
    // For the dense path (packed_bps == bps), this is a non-blocking
    // cudaMemcpyAsync on spill_stream; caller must synchronize spill_stream
    // before reading d_dst on another stream.
    // For the packed path, the kernel launches synchronize spill_stream
    // and stream 0 internally because the unpack kernel must observe the
    // staging copy.
    auto h2d_unpack_chunk = [&](const char* h_src, int count,
                                int packed_bps_in, int max_F_used,
                                char* d_dst) -> bool {
        if (count <= 0) return true;
        if (packed_bps_in == bps) {
            // Dense path -- single H2D, NO sync (caller responsibility).
            cudaError_t ce = cudaMemcpyAsync(
                d_dst, h_src,
                (size_t)count * (long long)bps,
                cudaMemcpyHostToDevice,
                spill_stream);
            if (ce != cudaSuccess) {
                std::fprintf(stderr,
                    "[V5 TRUNC] h2d_unpack_chunk dense: H2D count=%d failed: %s. "
                    "FAIL LOUD.\n", count, cudaGetErrorString(ce));
                return false;
            }
            return true;
        }
        // Packed path: H2D each batch into d_spill_staging, then unpack to d_dst.
        // Per-batch ordering:
        //   1. cudaStreamSynchronize(0)         -- wait for prior batch's
        //                                          unpack kernel (which reads
        //                                          d_spill_staging) to finish
        //                                          BEFORE the next H2D writes
        //                                          to d_spill_staging.
        //   2. H2D on spill_stream
        //   3. cudaStreamSynchronize(spill_stream) -- wait for H2D to finish
        //                                             before the unpack kernel
        //                                             reads d_spill_staging.
        //   4. Launch unpack kernel on default stream.
        // After the loop, sync stream 0 so caller sees fully-unpacked d_dst.
        int B_max = spill_chunk_states;
        for (int batch_start = 0; batch_start < count; batch_start += B_max) {
            int B = (batch_start + B_max <= count) ? B_max : (count - batch_start);
            long long batch_bytes = (long long)B * (long long)packed_bps_in;
            long long h_off = (long long)batch_start * (long long)packed_bps_in;
            // Wait for prior batch's unpack kernel before reusing d_spill_staging.
            if (batch_start > 0) {
                V5_CUDA(cudaStreamSynchronize(0));
            }
            cudaError_t ce = cudaMemcpyAsync(
                d_spill_staging, h_src + h_off,
                (size_t)batch_bytes, cudaMemcpyHostToDevice,
                spill_stream);
            if (ce != cudaSuccess) {
                std::fprintf(stderr,
                    "[V5 TRUNC] h2d_unpack_chunk: H2D batch %d (B=%d, %lld bytes) "
                    "failed: %s. FAIL LOUD.\n",
                    batch_start, B, batch_bytes, cudaGetErrorString(ce));
                return false;
            }
            V5_CUDA(cudaStreamSynchronize(spill_stream));
            int warps_needed = B;
            int blocks =
                (warps_needed + VARWIDTH_WARPS_PER_BLOCK - 1)
                / VARWIDTH_WARPS_PER_BLOCK;
            if (blocks < 1) blocks = 1;
            char* dst_ptr = d_dst + (long long)batch_start * bps;
            V5UnpackFromSpillKernel<<<blocks, VARWIDTH_THREADS_PER_BLOCK>>>(
                d_spill_staging, packed_bps_in, max_F_used,
                dst_ptr, layout, B, W, M);
        }
        // Synchronize default stream (kernels) so caller sees unpacked state.
        V5_CUDA(cudaStreamSynchronize(0));
        return true;
    };

    // iter370: scratch-spill helper. Copies d_layer_scratch[0..S) into the
    // host-pinned scratch ring at scratch_ring_used_bytes; records the chunk
    // metadata in scratch_chunks. Caller should reset total_pre_cw_merged=0
    // afterwards. Returns false (TRUNC) if the ring is exhausted.
    //
    // varwidth-spill (iter+1): if varwidth_spill_enabled, first measure
    // max F_count across the chunk via V5ChunkMaxFCountKernel; then pack
    // through d_spill_staging in batches at packed_bps stride; record the
    // chunk's packed_bps + max_F_used in scratch_chunks for unpack on read-back.
    auto do_scratch_spill = [&](int layer, int S) -> bool {
        if (S <= 0) return true;
        if (!ensure_spill_initialized()) {
            std::fprintf(stderr,
                "[V5 TRUNC] layer=%d scratch_spill: scratch ring not "
                "available (need SAG_V5_HOST_SPILL_GB>0). FAIL LOUD.\n",
                layer);
            return false;
        }

        // Decide chunk's packed stride (varwidth) or fall back to dense.
        int chunk_max_F = 0;
        int chunk_packed_bps = bps;  // default = dense
        if (varwidth_spill_enabled) {
            chunk_max_F = measure_chunk_max_F(d_layer_scratch, S);
            chunk_packed_bps = V5_packed_bps(W, M, chunk_max_F);
            if (chunk_packed_bps > bps) chunk_packed_bps = bps;
            if (chunk_packed_bps < (int)sizeof(int32_t)) chunk_packed_bps = bps;
        }

        long long bytes_needed = (long long)S * (long long)chunk_packed_bps;
        long long min_needed = scratch_ring_used_bytes + bytes_needed;
        // iter400: try to grow ring to fit. grow_scratch_ring is a no-op if
        // already large enough; returns false only if growth hits the cap.
        if (min_needed > host_scratch_ring_bytes_alloced) {
            // Wait for any in-flight spill_stream writes BEFORE we realloc
            // (we'll memcpy the existing ring content; any pending DMAs need
            // to land first).
            V5_CUDA(cudaStreamSynchronize(spill_stream));
            (void)grow_scratch_ring(min_needed, scratch_ring_used_bytes);
        }
        if (min_needed > host_scratch_ring_bytes_alloced) {
            std::fprintf(stderr,
                "[V5 TRUNC] layer=%d scratch_spill: ring exhausted "
                "(used %lld + need %lld > alloc %lld / cap %lld bytes; states %d). "
                "Increase SAG_V5_HOST_SPILL_GB. FAIL LOUD.\n",
                layer,
                scratch_ring_used_bytes, bytes_needed,
                host_scratch_ring_bytes_alloced,
                host_scratch_ring_bytes, S);
            return false;
        }

        if (chunk_packed_bps == bps) {
            // Dense path -- D2H copy directly.
            cudaError_t ce = cudaMemcpyAsync(
                h_scratch_spill + scratch_ring_used_bytes,
                d_layer_scratch,
                (size_t)bytes_needed,
                cudaMemcpyDeviceToHost,
                spill_stream);
            if (ce != cudaSuccess) {
                std::fprintf(stderr,
                    "[V5 TRUNC] layer=%d scratch_spill: D2H copy %lld bytes "
                    "failed: %s. FAIL LOUD.\n",
                    layer, bytes_needed, cudaGetErrorString(ce));
                return false;
            }
        } else {
            // varwidth-spill: pack in batches.
            if (!d2h_pack_chunk(d_layer_scratch, S,
                                chunk_packed_bps, chunk_max_F,
                                h_scratch_spill + scratch_ring_used_bytes)) {
                std::fprintf(stderr,
                    "[V5 TRUNC] layer=%d scratch_spill varwidth: pack "
                    "failed.\n", layer);
                return false;
            }
            total_spill_states_packed += S;
        }
        total_spill_bytes_dense_eq += (long long)S * (long long)bps;

        scratch_chunks.push_back({scratch_ring_used_bytes, S,
                                  chunk_packed_bps, chunk_max_F});
        scratch_ring_used_bytes += bytes_needed;
        total_scratch_spill_states += S;
        total_scratch_spill_chunks += 1;
        total_spill_bytes_xferd += bytes_needed;

        // The next wave will overwrite d_layer_scratch[0..]; ensure the D2H
        // started reading first so we don't race. (In CUDA, host-issued ops
        // on the default stream serialize with stream-launched memcpys via
        // device-side dependency tracking on `d_layer_scratch`. To be safe,
        // synchronize spill_stream so the next K1+K2 writes are clean.)
        V5_CUDA(cudaStreamSynchronize(spill_stream));
        return true;
    };

    // iter380: pair merge of two host-spill chunks. Both chunks must be
    // sorted+deduped (via the pre-sort pass). Loads them as dual-source
    // input (a -> d_aggregator, b -> d_layer_scratch), runs sort+detect+merge,
    // and writes the merged output to NEW host-spill ring slots via
    // batched D2H through d_spill_staging.
    //
    // The output may be SPLIT across multiple chunks: each output chunk
    // is capped at `out_chunk_cap_states` to ensure it can be loaded into
    // d_aggregator (or d_layer_scratch) alone in subsequent rounds. When
    // a chunk fills, a new chunk is started at the next ring offset.
    // The list of output (offset, count) pairs is returned via out_chunks.
    //
    // Why new slots rather than overwriting chunk_a's slot?
    //   In-place reuse only works when chunks are contiguous AND the
    //   single output fits in [off_a, off_b + count_b*bps). With multiple
    //   output chunks the layout doesn't match. The post-round compact
    //   step keeps ring usage bounded.
    //
    // Returns false on any failure.
    //
    // Preconditions:
    //   * d_aggregator allocated, count_a fits.
    //   * d_layer_scratch ENTIRELY free, count_b fits.
    //   * d_spill_staging allocated.
    //   * h_scratch_spill allocated; ring has capacity for count_a+count_b
    //     more bytes (worst-case dedup grows nothing, so this is a safe
    //     upper bound).
    //
    // FAIL-LOUD: any sort/merge failure or D2H error printf's and returns false.
    auto do_pair_merge_chunks = [&](int layer,
                                    const ScratchChunk& chunk_a,
                                    const ScratchChunk& chunk_b,
                                    int out_chunk_cap_states,
                                    std::vector<ScratchChunk>& out_chunks)
                                    -> bool {
        long long off_a = chunk_a.host_offset;
        int       count_a = chunk_a.count;
        int       packed_bps_a = chunk_a.packed_bps;
        int       max_F_a = chunk_a.max_F_used;
        long long off_b = chunk_b.host_offset;
        int       count_b = chunk_b.count;
        int       packed_bps_b = chunk_b.packed_bps;
        int       max_F_b = chunk_b.max_F_used;
        out_chunks.clear();
        if (count_a <= 0 && count_b <= 0) {
            return true;
        }
        if (count_a <= 0) {
            // Pass-through chunk_b.
            out_chunks.push_back(chunk_b);
            return true;
        }
        if (count_b <= 0) {
            out_chunks.push_back(chunk_a);
            return true;
        }
        if ((long long)count_a > d_aggregator_capacity_states) {
            std::fprintf(stderr,
                "[V5 TRUNC] layer=%d pair_merge: chunk_a count=%d > "
                "aggregator cap %d. FAIL LOUD.\n",
                layer, count_a, d_aggregator_capacity_states);
            return false;
        }
        if ((long long)count_b > cap.scratch_capacity) {
            std::fprintf(stderr,
                "[V5 TRUNC] layer=%d pair_merge: chunk_b count=%d > "
                "scratch cap %d. FAIL LOUD.\n",
                layer, count_b, cap.scratch_capacity);
            return false;
        }
        long long combined = (long long)count_a + (long long)count_b;
        if (combined > merge_scratch_max) {
            std::fprintf(stderr,
                "[V5 TRUNC] layer=%d pair_merge: combined %lld > "
                "merge_scratch_max %d. FAIL LOUD.\n",
                layer, combined, merge_scratch_max);
            return false;
        }

        // Allocate destination at the ring tail. Outputs may be SPLIT
        // into multiple chunks each bounded by out_chunk_cap_states; the
        // post-round compaction reclaims unused tail bytes.
        if (out_chunk_cap_states <= 0) {
            std::fprintf(stderr,
                "[V5 TRUNC] layer=%d pair_merge: out_chunk_cap_states=%d "
                "must be positive. FAIL LOUD.\n",
                layer, out_chunk_cap_states);
            return false;
        }

        // ---- Load chunk_a into d_aggregator, chunk_b into d_layer_scratch ----
        // Both chunks may have different packed_bps; unpack to dense in
        // d_aggregator / d_layer_scratch.
        if (!h2d_unpack_chunk(h_scratch_spill + off_a, count_a,
                              packed_bps_a, max_F_a, d_aggregator)) {
            std::fprintf(stderr,
                "[V5 TRUNC] layer=%d pair_merge: H2D chunk_a unpack (off=%lld, "
                "count=%d, pbps=%d) failed.\n",
                layer, off_a, count_a, packed_bps_a);
            return false;
        }
        if (!h2d_unpack_chunk(h_scratch_spill + off_b, count_b,
                              packed_bps_b, max_F_b, d_layer_scratch)) {
            std::fprintf(stderr,
                "[V5 TRUNC] layer=%d pair_merge: H2D chunk_b unpack (off=%lld, "
                "count=%d, pbps=%d) failed.\n",
                layer, off_b, count_b, packed_bps_b);
            return false;
        }
        // Dense path of h2d_unpack_chunk does NOT synchronize spill_stream;
        // pair_merge launches kernels on default stream so we sync here.
        V5_CUDA(cudaStreamSynchronize(spill_stream));
        total_spill_bytes_xferd +=
            (long long)count_a * (long long)packed_bps_a +
            (long long)count_b * (long long)packed_bps_b;

        // ---- Extract D-keys + iota with dual-source layout ----
        // Indices [0, count_a) -> d_aggregator
        // Indices [count_a, combined) -> d_layer_scratch
        {
            int blocks_a = (count_a + 255) / 256;
            if (blocks_a < 1) blocks_a = 1;
            V5ExtractDKeysIotaOffsetKernel<<<blocks_a, 256>>>(
                d_aggregator, layout, count_a,
                /*out_offset=*/0, /*idx_offset=*/0, W,
                d_dkeys_in, d_idx_in);
        }
        {
            int blocks_b = (count_b + 255) / 256;
            if (blocks_b < 1) blocks_b = 1;
            V5ExtractDKeysIotaOffsetKernel<<<blocks_b, 256>>>(
                d_layer_scratch, layout, count_b,
                /*out_offset=*/count_a, /*idx_offset=*/count_a, W,
                d_dkeys_in, d_idx_in);
        }

        // ---- Sort by D-key globally ----
        int combined_int = (int)combined;
        V5_CUDA(cudaMemcpyAsync(d_curr_count, &combined_int, sizeof(int),
                                cudaMemcpyHostToDevice));
        if (W == 1) {
            size_t tb = sort_temp_bytes;
            cub::DeviceRadixSort::SortPairs(
                d_cub_temp, tb,
                d_dkeys_in, d_dkeys_out,
                d_idx_in,   d_idx_out,
                combined_int, 0, N);
        } else {
            V5_CUDA(cudaMemcpyAsync(d_idx_out, d_idx_in,
                                    combined_int * sizeof(int),
                                    cudaMemcpyDeviceToDevice));
            const uint64_t* dk_q = d_dkeys_in;
            int             lw_q = W;
            size_t tb = sort_temp_bytes;
            cub::DeviceMergeSort::SortKeys(
                d_cub_temp, tb,
                d_idx_out, combined_int,
                [dk_q, lw_q] __device__ (const int& a, const int& b) {
                    for (int w = 0; w < lw_q; w++) {
                        if (dk_q[a * lw_q + w] < dk_q[b * lw_q + w]) return true;
                        if (dk_q[a * lw_q + w] > dk_q[b * lw_q + w]) return false;
                    }
                    return false;
                });
        }

        // ---- Detect groups + scan ----
        {
            int blocks = (combined_int + 255) / 256;
            if (blocks < 1) blocks = 1;
            V5DetectBoundariesKernel<<<blocks, 256>>>(
                d_dkeys_in, d_idx_out, d_curr_count, W, d_is_start);
        }
        {
            size_t sb = scan_temp_bytes;
            cub::DeviceScan::ExclusiveSum(
                d_cub_temp, sb,
                d_is_start, d_group_id,
                combined_int);
        }
        V5_CUDA(cudaMemsetAsync(d_num_groups, 0, sizeof(int)));
        {
            int blocks = (combined_int + 255) / 256;
            if (blocks < 1) blocks = 1;
            V5CompactGroupStartsKernel<<<blocks, 256>>>(
                d_is_start, d_group_id, d_curr_count,
                d_group_starts, d_num_groups);
        }
        V5_CUDA(cudaMemcpyAsync(&h_pin_scratch[3], d_num_groups,
                                sizeof(int), cudaMemcpyDeviceToHost));
        V5_CUDA(cudaStreamSynchronize(0));
        int n_groups = h_pin_scratch[3];
        if (n_groups < 1) {
            std::fprintf(stderr,
                "[V5 TRUNC] layer=%d pair_merge: 0 groups despite "
                "combined=%d. FAIL LOUD.\n", layer, combined_int);
            return false;
        }

        // ---- Batched merge with output going to d_spill_staging,
        //      D2H'd to host ring slots. Output may be split across
        //      multiple chunks each ≤ out_chunk_cap_states. ----
        const int slots_per_group_safe_factor = 8;
        long long total_emitted_states = 0;
        int g_processed = 0;
        // Current output chunk being filled.
        long long cur_chunk_off = scratch_ring_used_bytes;
        long long cur_chunk_emit = 0;  // states emitted into current chunk

        // Pre-flight ring capacity check: worst-case emit = combined.
        // Rough upper bound. iter400: try to grow first.
        {
            long long min_needed = cur_chunk_off + combined * (long long)bps;
            if (min_needed > host_scratch_ring_bytes_alloced) {
                V5_CUDA(cudaStreamSynchronize(spill_stream));
                (void)grow_scratch_ring(min_needed, scratch_ring_used_bytes);
            }
            if (min_needed > host_scratch_ring_bytes_alloced) {
                std::fprintf(stderr,
                    "[V5 TRUNC] layer=%d pair_merge: ring tail %lld + combined "
                    "reserve %lld > ring alloc %lld / cap %lld. "
                    "Increase SAG_V5_HOST_SPILL_GB. FAIL LOUD.\n",
                    layer, cur_chunk_off, combined * (long long)bps,
                    host_scratch_ring_bytes_alloced, host_scratch_ring_bytes);
                return false;
            }
        }

        while (g_processed < n_groups) {
            // Bound batch by current chunk's remaining cap so we never
            // emit more states than out_chunk_cap_states into a single
            // host chunk. Also bounded by d_spill_staging's slot count.
            long long room_in_chunk =
                (long long)out_chunk_cap_states - cur_chunk_emit;
            if (room_in_chunk <= 0) {
                // Close current chunk; start a new one at the next ring
                // offset. pair_merge outputs use dense packing
                // (packed_bps==bps); subsequent reads via h2d_unpack_chunk
                // see this and take the dense path with no kernel work.
                out_chunks.push_back({cur_chunk_off, (int)cur_chunk_emit,
                                       /*packed_bps=*/bps,
                                       /*max_F_used=*/0});
                cur_chunk_off += cur_chunk_emit * (long long)bps;
                cur_chunk_emit = 0;
                room_in_chunk = (long long)out_chunk_cap_states;
                continue;
            }
            long long dest_capacity = (long long)spill_chunk_states;
            if (room_in_chunk < dest_capacity) dest_capacity = room_in_chunk;
            int batch_groups = (int)(dest_capacity /
                                     (long long)slots_per_group_safe_factor);
            if (batch_groups < 1) batch_groups = 1;
            if (batch_groups > n_groups - g_processed)
                batch_groups = n_groups - g_processed;

            // Wait for prior D2H so we can overwrite d_spill_staging.
            V5_CUDA(cudaStreamSynchronize(spill_stream));

            V5_CUDA(cudaMemsetAsync(d_merge_count, 0, sizeof(int)));
            {
                int merge_grid =
                    (batch_groups + merge_warps_per_block - 1)
                        / merge_warps_per_block;
                if (merge_grid < 1) merge_grid = 1;
                // Dual-source: d_pre = d_aggregator (count_a indices),
                //              d_pre2 = d_layer_scratch (count_b indices)
                int spill_dest_cap_int =
                    (dest_capacity > (long long)INT_MAX)
                        ? INT_MAX : (int)dest_capacity;
                V5_LAUNCH_MERGE(merge_grid, merge_block_size, merge_smem,
                    d_aggregator, d_layer_scratch, /*pre1_count=*/count_a,
                    /*d_pre3=*/(const char*)nullptr,
                    /*pre2_end=*/0,
                    d_idx_out, d_group_starts,
                    g_processed, batch_groups, n_groups,
                    d_curr_count,
                    layout, N, M, W,
                    d_spill_staging, spill_dest_cap_int,
                    d_merge_count, d_trunc, layer);
            }
            V5_CUDA(cudaMemcpyAsync(&h_pin_scratch[0], d_merge_count,
                                    sizeof(int), cudaMemcpyDeviceToHost));
            V5_CUDA(cudaMemcpyAsync(&h_pin_scratch[2], d_trunc,
                                    sizeof(int), cudaMemcpyDeviceToHost));
            V5_CUDA(cudaStreamSynchronize(0));
            int batch_merged = h_pin_scratch[0];
            int batch_tf     = h_pin_scratch[2];
            if (batch_merged < 0) batch_merged = 0;
            if (batch_tf != 0) {
                std::fprintf(stderr,
                    "[V5 TRUNC] layer=%d pair_merge batch g=%d/%d: trunc "
                    "flag (F_count overflow likely). FAIL LOUD.\n",
                    layer, g_processed, n_groups);
                return false;
            }
            if ((long long)batch_merged > dest_capacity) {
                std::fprintf(stderr,
                    "[V5 TRUNC] layer=%d pair_merge batch g=%d sz=%d: "
                    "emitted=%d > dest_capacity=%lld. FAIL LOUD.\n",
                    layer, g_processed, batch_groups,
                    batch_merged, dest_capacity);
                return false;
            }

            // D2H batch_merged states to current chunk slot.
            cudaError_t ce = cudaMemcpyAsync(
                h_scratch_spill + cur_chunk_off
                    + cur_chunk_emit * (long long)bps,
                d_spill_staging,
                (size_t)batch_merged * (long long)bps,
                cudaMemcpyDeviceToHost,
                spill_stream);
            if (ce != cudaSuccess) {
                std::fprintf(stderr,
                    "[V5 TRUNC] layer=%d pair_merge: D2H copy back to host "
                    "ring failed: %s. FAIL LOUD.\n",
                    layer, cudaGetErrorString(ce));
                return false;
            }
            total_spill_bytes_xferd += (long long)batch_merged * (long long)bps;

            cur_chunk_emit += batch_merged;
            total_emitted_states += batch_merged;
            g_processed += batch_groups;
        }
        V5_CUDA(cudaStreamSynchronize(spill_stream));

        if (total_emitted_states > combined) {
            std::fprintf(stderr,
                "[V5 INTERNAL] layer=%d pair_merge: merged %lld > "
                "combined %lld. BUG.\n",
                layer, total_emitted_states, combined);
            return false;
        }
        // Close the last chunk. pair_merge outputs are dense (see comment
        // above on output chunk packed_bps).
        if (cur_chunk_emit > 0) {
            out_chunks.push_back({cur_chunk_off, (int)cur_chunk_emit,
                                   /*packed_bps=*/bps,
                                   /*max_F_used=*/0});
            cur_chunk_off += cur_chunk_emit * (long long)bps;
        }
        // Advance ring tail to account for all emitted bytes.
        scratch_ring_used_bytes = cur_chunk_off;
        return true;
    };

    // iter380: pairwise round helper. Walks scratch_chunks pairwise,
    // merging each (chunks[2i], chunks[2i+1]) into one or more new chunks
    // appended at the ring tail. Each output chunk is bounded by
    // `out_chunk_cap` so that subsequent pair-merge rounds (which load
    // chunks into d_aggregator alone) don't exceed buffer capacity.
    //
    // After this round, chunks.size() may NOT halve cleanly: a pair
    // whose merged output exceeds out_chunk_cap becomes >=2 output chunks.
    // The total emitted states across all output chunks is <= combined
    // (since dedup never grows) so net chunk count still trends down.
    //
    // Ring management: source chunk slots become dead after each pair
    // merges. After each round we COMPACT the ring: copy all surviving
    // chunks to the START of the ring contiguously via host memcpy,
    // updating their offsets and resetting the ring tail.
    //
    // Returns false on TRUNC.
    auto do_pair_merge_round = [&](int layer,
                                   std::vector<ScratchChunk>& chunks_io,
                                   int out_chunk_cap) -> bool {
        std::vector<ScratchChunk> next_round;
        next_round.reserve(chunks_io.size());

        std::vector<ScratchChunk> pair_out;
        pair_out.reserve(8);

        for (size_t i = 0; i < chunks_io.size(); ) {
            ScratchChunk a = chunks_io[i];
            if (i + 1 >= chunks_io.size()) {
                // Odd last chunk: pass through.
                if (a.count > 0) next_round.push_back(a);
                i += 1;
                continue;
            }
            ScratchChunk b = chunks_io[i + 1];
            if (a.count <= 0 && b.count <= 0) {
                i += 2;
                continue;
            }
            if (a.count <= 0) {
                next_round.push_back(b);
                i += 2;
                continue;
            }
            if (b.count <= 0) {
                next_round.push_back(a);
                i += 2;
                continue;
            }
            pair_out.clear();
            if (!do_pair_merge_chunks(layer,
                                      a, b,
                                      out_chunk_cap, pair_out)) {
                return false;
            }
            for (auto& oc : pair_out) {
                if (oc.count > 0) next_round.push_back(oc);
            }
            i += 2;
        }

        // Compact: move surviving chunks to the front of the ring so the
        // ring tail reflects only live chunk bytes. Without this, dead
        // (post-merge) source slots accumulate and exhaust the ring.
        //
        // Order of compaction matters: each chunk is copied to a new
        // (smaller) offset. We must not overwrite a chunk before reading
        // it. Since chunks are listed in the order they were emitted
        // (round_emit by round_emit, monotonically increasing offsets),
        // and we copy each to a strictly smaller offset, processing in
        // order is safe.
        long long write_off = 0;
        for (auto& c : next_round) {
            // varwidth-spill: bytes per chunk is c.count * c.packed_bps
            // (packed_bps == bps for dense chunks).
            int chunk_bps = c.packed_bps;
            if (chunk_bps <= 0) chunk_bps = bps;
            if (c.host_offset == write_off) {
                // Already in place.
                write_off += (long long)c.count * (long long)chunk_bps;
                continue;
            }
            if (c.host_offset < write_off) {
                std::fprintf(stderr,
                    "[V5 INTERNAL] layer=%d pair_merge_round compact: "
                    "chunk off %lld < write_off %lld. BUG.\n",
                    layer, c.host_offset, write_off);
                return false;
            }
            // Source [c.host_offset, c.host_offset + bytes), dest
            // [write_off, write_off + bytes). bytes = count*chunk_bps.
            long long bytes = (long long)c.count * (long long)chunk_bps;
            // Memmove (overlapping safe) — but here dest < src so memcpy
            // is fine. Use memmove defensively.
            std::memmove(h_scratch_spill + write_off,
                         h_scratch_spill + c.host_offset,
                         (size_t)bytes);
            c.host_offset = write_off;
            write_off += bytes;
        }
        scratch_ring_used_bytes = write_off;
        chunks_io.swap(next_round);
        return true;
    };

    // iter370: layer-end consolidation when scratch spill happened. Pages
    // all spilled chunks back into d_aggregator (and overflow into
    // d_layer_scratch via dual-source); then runs sort+merge to dedupe
    // into d_next + h_layer_spill_next.
    //
    // iter380: when even after per-chunk pre-sort the combined still
    // exceeds (aggregator + scratch), run log2(K) rounds of pairwise merge
    // first to reduce the chunk count and total size.
    //
    // iter390: extend with d_curr as a 3rd input source during consolidate
    // (d_curr is the prior layer's resident states; consumed by the K1+K2
    // wave loop above, not touched during consolidate, ping-ponged AFTER
    // we write to d_next). This gives a triple-source merge with combined
    // capacity = aggregator + scratch + layer = ~53.5M states (vs ~44.3M
    // for dual). The pair-merge threshold also lifts to triple, so fewer
    // rounds run before the system can fit the consolidate.
    //
    // Dual-source layout for combined input:
    //   indices [0, agg_count)             -> d_aggregator[0 .. agg_count)
    //   indices [agg_count, total)         -> d_layer_scratch[0 .. residual_in_scratch)
    //
    // Triple-source layout for combined input:
    //   indices [0, agg_count)             -> d_aggregator[0 .. agg_count)
    //   indices [agg_count, scratch_used)  -> d_layer_scratch[0 .. scratch_used - agg_count)
    //   indices [scratch_used, total)      -> d_curr[0 .. total - scratch_used)
    //
    // Returns false (TRUNC) if combined total exceeds d_aggregator +
    // d_layer_scratch + d_curr combined even after all pair-merge rounds.
    auto do_layer_end_consolidate = [&](int layer, int S_residual,
                                        int& new_cwm) -> bool {
        // Wait for any in-flight spill_stream copies to land on the host.
        V5_CUDA(cudaStreamSynchronize(spill_stream));

        long long combined_total = (long long)S_residual;
        for (auto& c : scratch_chunks) combined_total += c.count;
        if (combined_total <= 0) {
            new_cwm = 0;
            return true;
        }
        // iter390: combined_capacity_dual is the existing aggregator + scratch
        // ceiling; triple adds d_curr (size = cap.layer_capacity) as a 3rd
        // input source. The consolidate writes to d_next; d_curr is unused
        // during consolidate and gets repurposed as input.
        long long combined_capacity_dual =
            (long long)d_aggregator_capacity_states +
            (long long)cap.scratch_capacity;
        long long combined_capacity_triple =
            combined_capacity_dual + (long long)cap.layer_capacity;
        // iter390: pair-merge runs only when total > triple cap (the larger
        // ceiling). Below triple cap, we can do a single triple-source merge
        // without the pair-rounds.
        long long combined_capacity = combined_capacity_triple;

        // ---- iter370 PRE-SORT PASS ----
        //
        // If combined_total > combined_capacity, try a per-chunk pre-sort
        // pass that sorts and dedupes each spilled chunk individually,
        // shrinking each chunk and (often) the total enough to fit in the
        // single-pass consolidate. Each chunk is loaded into d_aggregator,
        // sorted+merged into d_aggregator (in-place via the "merge writes
        // back into d_aggregator" pattern using d_layer_scratch as scratch),
        // then written back to host as a smaller chunk.
        //
        // Each chunk passes once: 1 H2D + 1 D2H = 2 PCIe touches per chunk
        // for the pre-sort. After pre-sort, chunks are sorted-by-D-key and
        // deduplicated.
        if (combined_total > combined_capacity) {
            if (verbose) {
                std::fprintf(stderr,
                    "[layer %d] consolidate: total %lld > combined cap %lld; "
                    "running per-chunk pre-sort pass on %zu chunks.\n",
                    layer, combined_total, combined_capacity,
                    scratch_chunks.size());
            }

            // Pre-sort each chunk in place.
            for (auto& c : scratch_chunks) {
                if (c.count <= 0) continue;
                if (c.count > d_aggregator_capacity_states) {
                    // Chunk doesn't fit aggregator alone; can't pre-sort
                    // this one with the simple in-aggregator path. Skip --
                    // the consolidate will then likely fail and TRUNC. (A
                    // real fix would split the chunk via dual-source pre-sort,
                    // but that's a significant refactor.)
                    if (verbose) {
                        std::fprintf(stderr,
                            "[layer %d] pre-sort: skipping chunk count=%d > "
                            "aggregator cap %d (would need split-pre-sort).\n",
                            layer, c.count, d_aggregator_capacity_states);
                    }
                    continue;
                }

                // Page chunk into d_aggregator. varwidth-spill: unpack
                // varying packed_bps to dense layout.
                int chunk_pbps_in = c.packed_bps;
                if (chunk_pbps_in <= 0) chunk_pbps_in = bps;
                if (!h2d_unpack_chunk(h_scratch_spill + c.host_offset,
                                      c.count, chunk_pbps_in, c.max_F_used,
                                      d_aggregator)) {
                    std::fprintf(stderr,
                        "[V5 TRUNC] layer=%d pre-sort: H2D unpack of chunk "
                        "(off=%lld, count=%d, pbps=%d) failed.\n",
                        layer, c.host_offset, c.count, chunk_pbps_in);
                    return false;
                }
                // Dense path of h2d_unpack_chunk does NOT synchronize
                // spill_stream; do_flush uses default stream so we must
                // sync here to observe the H2D in default-stream kernels.
                V5_CUDA(cudaStreamSynchronize(spill_stream));
                total_spill_bytes_xferd +=
                    (long long)c.count * (long long)chunk_pbps_in;

                // Run sort+detect+merge in d_aggregator. The merge writes to
                // d_layer_scratch (which is sized scratch_capacity, big
                // enough to hold the deduped chunk), then we copy back.
                int new_cwm_dummy = 0;
                if (!do_flush(layer, d_aggregator, c.count,
                              /*final=*/false, new_cwm_dummy)) {
                    std::fprintf(stderr,
                        "[V5 TRUNC] layer=%d pre-sort: do_flush on chunk "
                        "(count=%d) failed.\n", layer, c.count);
                    return false;
                }
                // do_flush wrote the deduped output to d_next (the merge dest
                // we use throughout). new_cwm_dummy is the deduped count
                // currently RESIDENT in d_next; any overflow lives in
                // h_layer_spill_next (count tracked by next_spill_count_h).
                //
                // Earlier this code asserted next_spill_count_h == 0 here on
                // the assumption that a single chunk's dedup always fits in
                // d_next (layer_capacity). On hard cases (e.g. n28/u50/t000)
                // a chunk's group count can exceed layer_capacity, so
                // do_flush legitimately spills. Treat this as a split:
                // deduped portion in d_next + spilled portion in
                // h_layer_spill_next. Both parts are written back into the
                // chunk's host slot (the slot is sized c.count*bps and
                // total_deduped <= c.count, so it fits).
                int deduped = new_cwm_dummy;
                if (deduped < 0) deduped = 0;
                int spilled = next_spill_count_h;
                if (spilled < 0) spilled = 0;
                long long total_deduped = (long long)deduped + (long long)spilled;
                if (total_deduped > (long long)c.count) {
                    std::fprintf(stderr,
                        "[V5 INTERNAL] layer=%d pre-sort: dedup grew chunk "
                        "from %d to %lld (resident=%d spill=%d). BUG.\n",
                        layer, c.count, total_deduped, deduped, spilled);
                    return false;
                }
                if (total_deduped == 0) {
                    // Empty after dedup -- mark chunk empty.
                    c.count = 0;
                    next_spill_count_h = 0;
                    continue;
                }

                // varwidth-spill: writeback with pack. Pre-sort produces
                // deduped states in d_next (resident) + h_layer_spill_next
                // (overflow). The original slot is sized
                // c.count * c.packed_bps bytes. After pre-sort the chunk
                // has total_deduped <= c.count states, but they may have
                // higher F_count after merge widening, so we cannot reuse
                // c.packed_bps if the new max_F is larger.
                //
                // Strategy: always re-measure max_F, choose new_packed_bps,
                // and write to the OLD slot if new bytes fit; else append
                // to ring tail.

                int new_max_F_d_next = 0;
                if (deduped > 0) {
                    new_max_F_d_next = measure_chunk_max_F(d_next, deduped);
                }
                // For spilled portion in h_layer_spill_next we don't have a
                // direct max_F (it was emitted by do_flush as dense bytes).
                // For safety, scan the host bytes and find max F_count.
                int new_max_F_spill = 0;
                if (spilled > 0) {
                    // Each state is bps-aligned in h_layer_spill_next (dense
                    // layout, since do_flush spills states in dense form).
                    // F_count is at byte offset header_bytes from each state.
                    int header_bytes = V5_layout_header_bytes(W, M);
                    for (int i = 0; i < spilled; i++) {
                        int fc = *(const int32_t*)(h_layer_spill_next
                                  + (long long)i * bps + header_bytes);
                        if (fc > new_max_F_spill) new_max_F_spill = fc;
                    }
                }
                int new_max_F = (new_max_F_d_next > new_max_F_spill)
                                ? new_max_F_d_next : new_max_F_spill;
                int new_packed_bps = bps;
                if (varwidth_spill_enabled) {
                    new_packed_bps = V5_packed_bps(W, M, new_max_F);
                    if (new_packed_bps > bps) new_packed_bps = bps;
                    if (new_packed_bps < (int)sizeof(int32_t)) new_packed_bps = bps;
                }

                long long new_bytes = (long long)total_deduped
                                    * (long long)new_packed_bps;
                long long old_slot_bytes = (long long)c.count
                                         * (long long)c.packed_bps;
                long long writeback_off = c.host_offset;
                bool      writeback_to_tail = (new_bytes > old_slot_bytes);
                if (writeback_to_tail) {
                    // Reserve at ring tail (with grow if needed).
                    long long tail_min = scratch_ring_used_bytes + new_bytes;
                    if (tail_min > host_scratch_ring_bytes_alloced) {
                        V5_CUDA(cudaStreamSynchronize(spill_stream));
                        (void)grow_scratch_ring(tail_min,
                                                scratch_ring_used_bytes);
                    }
                    if (tail_min > host_scratch_ring_bytes_alloced) {
                        std::fprintf(stderr,
                            "[V5 TRUNC] layer=%d pre-sort writeback: ring "
                            "tail %lld + needed %lld > alloc %lld / cap "
                            "%lld. FAIL LOUD.\n",
                            layer, scratch_ring_used_bytes, new_bytes,
                            host_scratch_ring_bytes_alloced,
                            host_scratch_ring_bytes);
                        return false;
                    }
                    writeback_off = scratch_ring_used_bytes;
                    scratch_ring_used_bytes += new_bytes;
                }

                // Write resident portion (d_next).
                if (deduped > 0) {
                    if (new_packed_bps == bps) {
                        cudaError_t ce = cudaMemcpyAsync(
                            h_scratch_spill + writeback_off,
                            d_next,
                            (size_t)deduped * (long long)bps,
                            cudaMemcpyDeviceToHost,
                            spill_stream);
                        if (ce != cudaSuccess) {
                            std::fprintf(stderr,
                                "[V5 TRUNC] layer=%d pre-sort: D2H dense "
                                "back failed: %s. FAIL LOUD.\n",
                                layer, cudaGetErrorString(ce));
                            return false;
                        }
                        V5_CUDA(cudaStreamSynchronize(spill_stream));
                        total_spill_bytes_xferd +=
                            (long long)deduped * (long long)bps;
                    } else {
                        if (!d2h_pack_chunk(d_next, deduped,
                                            new_packed_bps, new_max_F,
                                            h_scratch_spill + writeback_off)) {
                            std::fprintf(stderr,
                                "[V5 TRUNC] layer=%d pre-sort: pack D2H "
                                "back failed.\n", layer);
                            return false;
                        }
                        total_spill_bytes_xferd +=
                            (long long)deduped * (long long)new_packed_bps;
                        total_spill_states_packed += deduped;
                    }
                    total_spill_bytes_dense_eq +=
                        (long long)deduped * (long long)bps;
                }

                // Write spilled portion (h_layer_spill_next, dense in host)
                // immediately after the resident packed bytes. This is a
                // host-side operation: dense -> packed translation per state.
                if (spilled > 0) {
                    long long off_after_resident =
                        writeback_off + (long long)deduped
                                         * (long long)new_packed_bps;
                    if (new_packed_bps == bps) {
                        // Dense -> dense memcpy.
                        std::memcpy(
                            h_scratch_spill + off_after_resident,
                            h_layer_spill_next,
                            (size_t)spilled * (long long)bps);
                    } else {
                        // Per-state pack on host.
                        int header_bytes = V5_layout_header_bytes(W, M);
                        int entries_bytes =
                            new_max_F * (int)sizeof(sag::v5::FEntryV5);
                        // Dense ovf is at header_bytes
                        //   + F_MAX_PER_STATE*sizeof(FEntryV5)
                        // Packed ovf is at header_bytes + entries_bytes.
                        int dense_ovf_off  = header_bytes
                            + sag::v5::F_MAX_PER_STATE
                              * (int)sizeof(sag::v5::FEntryV5);
                        int packed_ovf_off = header_bytes + entries_bytes;
                        for (int i = 0; i < spilled; i++) {
                            const char* src = h_layer_spill_next
                                            + (long long)i * (long long)bps;
                            char*       dst = h_scratch_spill + off_after_resident
                                            + (long long)i
                                              * (long long)new_packed_bps;
                            // Copy header + max_F entries verbatim.
                            std::memcpy(dst, src,
                                        (size_t)header_bytes + entries_bytes);
                            // Copy ovf int32 from dense to packed offset.
                            *(int32_t*)(dst + packed_ovf_off) =
                                *(const int32_t*)(src + dense_ovf_off);
                        }
                        total_spill_states_packed += spilled;
                    }
                    total_spill_bytes_xferd +=
                        (long long)spilled * (long long)new_packed_bps;
                    total_spill_bytes_dense_eq +=
                        (long long)spilled * (long long)bps;
                    // The spilled states have been folded into this chunk;
                    // reset the layer-level spill accumulator so it doesn't
                    // double-count at layer end.
                    next_spill_count_h = 0;
                }
                c.count       = (int)total_deduped;
                c.host_offset = writeback_off;
                c.packed_bps  = new_packed_bps;
                c.max_F_used  = new_max_F;
            }

            // Recompute combined_total after pre-sort.
            combined_total = (long long)S_residual;
            for (auto& c : scratch_chunks) combined_total += c.count;

            if (verbose) {
                std::fprintf(stderr,
                    "[layer %d] pre-sort done: combined_total=%lld\n",
                    layer, combined_total);
            }

            // ---- iter380 PAIRWISE MERGE ROUNDS ----
            //
            // If even after per-chunk pre-sort the combined total still
            // exceeds combined_capacity, run log2(K) rounds of pairwise
            // merging that compact adjacent chunks two-at-a-time. Each
            // pair-merge: load two adjacent chunks into d_aggregator +
            // d_layer_scratch, dual-source sort+detect+merge, write the
            // dedup'd output back to the host ring at the FIRST chunk's
            // offset (using the combined input space which is always
            // >= the merged output).
            //
            // For pair_merge to be safe, we need d_layer_scratch to be
            // ENTIRELY free. S_residual currently occupies
            // d_layer_scratch[0..S_residual). If pair-merge is required,
            // we first spill S_residual as a final chunk, then run pair
            // rounds over the unified chunk list.
            if (combined_total > combined_capacity) {
                if (verbose) {
                    std::fprintf(stderr,
                        "[layer %d] consolidate: post-pre-sort total %lld "
                        "> combined cap %lld; entering pair-merge rounds "
                        "(chunks=%zu, S_residual=%d).\n",
                        layer, combined_total, combined_capacity,
                        scratch_chunks.size(), S_residual);
                }

                // Spill S_residual to the host ring as a final chunk so
                // that d_layer_scratch becomes free for pair-merge inputs.
                // After this, S_residual=0 effectively for the rest of
                // the consolidate (we use a local s_resid_local that is
                // either S_residual or 0).
                int s_resid_local = S_residual;
                if (s_resid_local > 0) {
                    int resid_max_F = 0;
                    int resid_packed_bps = bps;
                    if (varwidth_spill_enabled) {
                        resid_max_F = measure_chunk_max_F(d_layer_scratch,
                                                          s_resid_local);
                        resid_packed_bps = V5_packed_bps(W, M, resid_max_F);
                        if (resid_packed_bps > bps) resid_packed_bps = bps;
                        if (resid_packed_bps < (int)sizeof(int32_t))
                            resid_packed_bps = bps;
                    }
                    long long bytes_needed =
                        (long long)s_resid_local * (long long)resid_packed_bps;
                    long long min_needed =
                        scratch_ring_used_bytes + bytes_needed;
                    if (min_needed > host_scratch_ring_bytes_alloced) {
                        V5_CUDA(cudaStreamSynchronize(spill_stream));
                        (void)grow_scratch_ring(min_needed,
                                                scratch_ring_used_bytes);
                    }
                    if (min_needed > host_scratch_ring_bytes_alloced) {
                        std::fprintf(stderr,
                            "[V5 TRUNC] layer=%d consolidate pair-merge "
                            "prep: ring full (used %lld + need %lld > "
                            "alloc %lld / cap %lld). Increase SAG_V5_HOST_SPILL_GB. "
                            "FAIL LOUD.\n",
                            layer, scratch_ring_used_bytes, bytes_needed,
                            host_scratch_ring_bytes_alloced,
                            host_scratch_ring_bytes);
                        return false;
                    }
                    if (resid_packed_bps == bps) {
                        cudaError_t ce = cudaMemcpyAsync(
                            h_scratch_spill + scratch_ring_used_bytes,
                            d_layer_scratch,
                            (size_t)bytes_needed,
                            cudaMemcpyDeviceToHost,
                            spill_stream);
                        if (ce != cudaSuccess) {
                            std::fprintf(stderr,
                                "[V5 TRUNC] layer=%d consolidate pair-merge "
                                "prep: D2H dense of S_residual=%d failed: %s. "
                                "FAIL LOUD.\n",
                                layer, s_resid_local, cudaGetErrorString(ce));
                            return false;
                        }
                        V5_CUDA(cudaStreamSynchronize(spill_stream));
                    } else {
                        if (!d2h_pack_chunk(d_layer_scratch, s_resid_local,
                                            resid_packed_bps, resid_max_F,
                                            h_scratch_spill
                                              + scratch_ring_used_bytes)) {
                            std::fprintf(stderr,
                                "[V5 TRUNC] layer=%d consolidate pair-merge "
                                "prep: pack D2H of S_residual=%d failed.\n",
                                layer, s_resid_local);
                            return false;
                        }
                        total_spill_states_packed += s_resid_local;
                    }
                    total_spill_bytes_xferd += bytes_needed;
                    total_spill_bytes_dense_eq +=
                        (long long)s_resid_local * (long long)bps;
                    scratch_chunks.push_back(
                        {scratch_ring_used_bytes, s_resid_local,
                         resid_packed_bps, resid_max_F});
                    scratch_ring_used_bytes += bytes_needed;
                    total_scratch_spill_states += s_resid_local;
                    total_scratch_spill_chunks += 1;
                    s_resid_local = 0;
                    S_residual = 0;
                }

                // Cap the maximum size of any single output chunk at
                // `cap.aggregator_capacity` so that the next round can
                // load it into d_aggregator alone. Use the smaller of
                // (aggregator_capacity, scratch_capacity) since chunks
                // may end up in either input buffer of a pair_merge.
                int pair_chunk_cap =
                    std::min(d_aggregator_capacity_states, cap.scratch_capacity);
                if (pair_chunk_cap < 1) pair_chunk_cap = 1;

                // Run pair-merge rounds until combined_total fits.
                int round_idx = 0;
                while (combined_total > combined_capacity
                       && scratch_chunks.size() >= 2) {
                    if (!do_pair_merge_round(layer, scratch_chunks,
                                             pair_chunk_cap)) {
                        std::fprintf(stderr,
                            "[V5 TRUNC] layer=%d pair-merge round %d failed.\n",
                            layer, round_idx);
                        return false;
                    }
                    long long new_total = (long long)s_resid_local;
                    for (auto& c : scratch_chunks) new_total += c.count;
                    if (verbose) {
                        std::fprintf(stderr,
                            "[layer %d] pair-merge round %d: chunks=%zu "
                            "total=%lld (was %lld; reduction %.2fx)\n",
                            layer, round_idx, scratch_chunks.size(),
                            new_total, combined_total,
                            (combined_total > 0)
                              ? (double)combined_total / (double)(new_total > 0 ? new_total : 1)
                              : 0.0);
                    }
                    if (new_total >= combined_total
                        && scratch_chunks.size() >= 2) {
                        // No reduction this round + still > cap. Avoid infinite loop.
                        std::fprintf(stderr,
                            "[V5 TRUNC] layer=%d pair-merge round %d: no "
                            "reduction (was %lld, now %lld) and still > "
                            "combined cap %lld. FAIL LOUD.\n",
                            layer, round_idx, combined_total, new_total,
                            combined_capacity);
                        return false;
                    }
                    combined_total = new_total;
                    round_idx++;
                    if (round_idx > 32) {
                        std::fprintf(stderr,
                            "[V5 TRUNC] layer=%d pair-merge: too many rounds "
                            "(%d). FAIL LOUD.\n",
                            layer, round_idx);
                        return false;
                    }
                }

                if (combined_total > combined_capacity) {
                    std::fprintf(stderr,
                        "[V5 TRUNC] layer=%d consolidate: even after %d "
                        "pair-merge rounds, total %lld > combined cap %lld "
                        "(chunks=%zu). FAIL LOUD.\n",
                        layer, round_idx, combined_total,
                        combined_capacity, scratch_chunks.size());
                    return false;
                }
                if (verbose) {
                    std::fprintf(stderr,
                        "[layer %d] pair-merge done after %d rounds: "
                        "chunks=%zu total=%lld\n",
                        layer, round_idx, scratch_chunks.size(),
                        combined_total);
                }
            }
        }
        // Final guard (covers both the no-pre-sort and post-pre-sort paths).
        if (combined_total > combined_capacity) {
            std::fprintf(stderr,
                "[V5 TRUNC] layer=%d consolidate: total %lld states exceeds "
                "combined buffer capacity (aggregator %d + scratch %d + "
                "layer %d = %lld). FAIL LOUD.\n",
                layer, combined_total,
                d_aggregator_capacity_states, cap.scratch_capacity,
                cap.layer_capacity, combined_capacity);
            return false;
        }
        if (combined_total > merge_scratch_max) {
            std::fprintf(stderr,
                "[V5 TRUNC] layer=%d consolidate: total %lld states exceeds "
                "merge_scratch_max=%d. The merge_scratch buffers cannot index "
                "this many states. FAIL LOUD.\n",
                layer, combined_total, merge_scratch_max);
            return false;
        }

        // iter390: choose dual-source vs triple-source layout based on
        // combined_total. Dual is preferred (touches one fewer buffer);
        // triple unlocks the d_curr capacity (~9.5M extra states) when
        // dual cannot fit the combined input.
        //
        // Plan layout. We pack chunks first into d_aggregator; if the
        // residual scratch (already in d_layer_scratch[0..S_residual)) and
        // any chunk overflow won't fit in d_aggregator + d_layer_scratch,
        // additional overflow lands in d_curr (triple-source).
        //
        // Strategy: copy chunks INTO d_aggregator filling it. The residual
        // scratch is already in d_layer_scratch[0..S_residual); we APPEND
        // any chunk overflow AFTER the residual in d_layer_scratch. If
        // d_layer_scratch fills, remaining overflow goes into d_curr.
        //
        // Result (dual when overflow fits scratch):
        //   d_aggregator[0 .. agg_count)               : first agg_count chunk states
        //   d_layer_scratch[0 .. S_residual)           : residual (already there)
        //   d_layer_scratch[S_residual .. scratch_used) : chunk overflow
        //
        // Result (triple when scratch overflows):
        //   d_aggregator[0 .. agg_count)
        //   d_layer_scratch[0 .. S_residual)
        //   d_layer_scratch[S_residual .. cap.scratch_capacity)  : chunk overflow tail-1
        //   d_curr[0 .. third_source_count)            : chunk overflow tail-2
        //
        // The sort uses dual or triple-source indexing.
        long long total_chunk_states = combined_total - (long long)S_residual;
        long long agg_room = (long long)d_aggregator_capacity_states;
        long long agg_count = std::min(total_chunk_states, agg_room);
        long long overflow_count = total_chunk_states - agg_count;
        long long scratch_room_for_overflow =
            (long long)cap.scratch_capacity - (long long)S_residual;
        if (scratch_room_for_overflow < 0) scratch_room_for_overflow = 0;
        long long scratch_overflow_count =
            std::min(overflow_count, scratch_room_for_overflow);
        long long third_source_count =
            overflow_count - scratch_overflow_count;
        long long second_source_count =
            (long long)S_residual + scratch_overflow_count;

        // pre1_count = agg_count (boundary between d_aggregator and d_pre2),
        // passed directly to V5MergeKernel below.
        // pre2_end   = agg_count + second_source_count
        //              (boundary between d_pre2 (d_layer_scratch) and d_pre3 (d_curr)).
        long long pre2_end_ll   = agg_count + second_source_count;

        bool use_triple = (third_source_count > 0);

        // Sanity bounds.
        if (second_source_count > (long long)cap.scratch_capacity) {
            std::fprintf(stderr,
                "[V5 TRUNC] layer=%d consolidate: second-source size %lld "
                "exceeds d_layer_scratch %d. BUG (combined total check should "
                "have caught).\n",
                layer, second_source_count, cap.scratch_capacity);
            return false;
        }
        if (third_source_count > (long long)cap.layer_capacity) {
            std::fprintf(stderr,
                "[V5 TRUNC] layer=%d consolidate: third-source size %lld "
                "exceeds d_curr (layer_capacity) %d. BUG (combined total "
                "check should have caught).\n",
                layer, third_source_count, cap.layer_capacity);
            return false;
        }

        // Copy chunks into d_aggregator, then d_layer_scratch[S_residual..],
        // then d_curr[0..]. Chunks are copied in order; once a destination
        // buffer fills, the rest of the chunk goes to the next destination.
        long long chunks_processed = 0;
        long long agg_written = 0;
        long long scratch_appended = 0;
        long long curr_appended = 0;
        for (auto& c : scratch_chunks) {
            long long count_remaining = c.count;
            long long src_state_offset = 0;
            int chunk_pbps = c.packed_bps;
            if (chunk_pbps <= 0) chunk_pbps = bps;
            int chunk_max_F = c.max_F_used;
            // Fill d_aggregator first.
            if (agg_written < agg_count && count_remaining > 0) {
                long long take = std::min(count_remaining,
                                          agg_count - agg_written);
                if (!h2d_unpack_chunk(
                        h_scratch_spill + c.host_offset
                            + src_state_offset * (long long)chunk_pbps,
                        (int)take, chunk_pbps, chunk_max_F,
                        d_aggregator + agg_written * (long long)bps)) {
                    std::fprintf(stderr,
                        "[V5 TRUNC] layer=%d consolidate: H2D unpack into agg "
                        "(chunk_off=%lld, take=%lld, pbps=%d) failed.\n",
                        layer, c.host_offset, take, chunk_pbps);
                    return false;
                }
                total_spill_bytes_xferd += take * (long long)chunk_pbps;
                agg_written += take;
                src_state_offset += take;
                count_remaining -= take;
            }
            // Then overflow into d_layer_scratch[S_residual + scratch_appended ..]
            // up to scratch_overflow_count.
            if (count_remaining > 0
                && scratch_appended < scratch_overflow_count) {
                long long take = std::min(count_remaining,
                                          scratch_overflow_count - scratch_appended);
                if (!h2d_unpack_chunk(
                        h_scratch_spill + c.host_offset
                            + src_state_offset * (long long)chunk_pbps,
                        (int)take, chunk_pbps, chunk_max_F,
                        d_layer_scratch
                            + ((long long)S_residual + scratch_appended)
                                * (long long)bps)) {
                    std::fprintf(stderr,
                        "[V5 TRUNC] layer=%d consolidate: H2D unpack into scratch "
                        "(chunk_off=%lld, take=%lld, pbps=%d) failed.\n",
                        layer, c.host_offset, take, chunk_pbps);
                    return false;
                }
                total_spill_bytes_xferd += take * (long long)chunk_pbps;
                scratch_appended += take;
                src_state_offset += take;
                count_remaining -= take;
            }
            // iter390: remaining overflow goes into d_curr (triple-source).
            if (count_remaining > 0) {
                if (curr_appended + count_remaining > third_source_count) {
                    std::fprintf(stderr,
                        "[V5 INTERNAL] layer=%d consolidate: curr overflow "
                        "%lld + take %lld > third_source_count %lld. BUG.\n",
                        layer, curr_appended, count_remaining,
                        third_source_count);
                    return false;
                }
                if (!h2d_unpack_chunk(
                        h_scratch_spill + c.host_offset
                            + src_state_offset * (long long)chunk_pbps,
                        (int)count_remaining, chunk_pbps, chunk_max_F,
                        d_curr + curr_appended * (long long)bps)) {
                    std::fprintf(stderr,
                        "[V5 TRUNC] layer=%d consolidate: H2D unpack into curr "
                        "(chunk_off=%lld, take=%lld, pbps=%d) failed.\n",
                        layer, c.host_offset, count_remaining, chunk_pbps);
                    return false;
                }
                total_spill_bytes_xferd += count_remaining * (long long)chunk_pbps;
                curr_appended += count_remaining;
            }
            chunks_processed++;
        }
        V5_CUDA(cudaStreamSynchronize(spill_stream));

        if (agg_written != agg_count
            || scratch_appended != scratch_overflow_count
            || curr_appended != third_source_count) {
            std::fprintf(stderr,
                "[V5 INTERNAL] layer=%d consolidate: agg_written=%lld != %lld "
                "or scratch_appended=%lld != %lld or curr_appended=%lld != "
                "%lld. BUG.\n",
                layer, agg_written, agg_count,
                scratch_appended, scratch_overflow_count,
                curr_appended, third_source_count);
            return false;
        }

        if (verbose) {
            std::fprintf(stderr,
                "[layer %d] consolidate %s: total=%lld (chunks=%zu, agg=%lld, "
                "scratch_residual=%d, scratch_overflow=%lld, curr=%lld)\n",
                layer, use_triple ? "triple-source" : "dual-source",
                combined_total, scratch_chunks.size(),
                agg_count, S_residual, scratch_overflow_count,
                third_source_count);
        }

        // ---- Run the dual or triple-source sort + merge ----
        int combined = (int)combined_total;
        // Step 1a: Extract D-keys + indices from d_aggregator[0..agg_count).
        // Indices: [0..agg_count).
        if (agg_count > 0) {
            int blocks = ((int)agg_count + 255) / 256;
            if (blocks < 1) blocks = 1;
            V5ExtractDKeysIotaOffsetKernel<<<blocks, 256>>>(
                d_aggregator, layout, (int)agg_count,
                /*out_offset=*/0, /*idx_offset=*/0, W,
                d_dkeys_in, d_idx_in);
        }
        // Step 1b: Extract D-keys + indices from d_layer_scratch[0..second_source_count).
        // Indices in the global combined buffer are [agg_count .. agg_count + second_source_count).
        // Use idx_offset = agg_count so d_idx_in[agg_count + i] = agg_count + i.
        if (second_source_count > 0) {
            int blocks = ((int)second_source_count + 255) / 256;
            if (blocks < 1) blocks = 1;
            V5ExtractDKeysIotaOffsetKernel<<<blocks, 256>>>(
                d_layer_scratch, layout, (int)second_source_count,
                /*out_offset=*/(int)agg_count,
                /*idx_offset=*/(int)agg_count, W,
                d_dkeys_in, d_idx_in);
        }
        // iter390: Step 1c: Extract D-keys + indices from
        // d_curr[0..third_source_count). Indices in the global combined buffer
        // are [pre2_end_ll .. combined). idx_offset = pre2_end_ll.
        if (third_source_count > 0) {
            int blocks = ((int)third_source_count + 255) / 256;
            if (blocks < 1) blocks = 1;
            V5ExtractDKeysIotaOffsetKernel<<<blocks, 256>>>(
                d_curr, layout, (int)third_source_count,
                /*out_offset=*/(int)pre2_end_ll,
                /*idx_offset=*/(int)pre2_end_ll, W,
                d_dkeys_in, d_idx_in);
        }

        // Step 2: Sort by D-key globally over [0..combined).
        V5_CUDA(cudaMemcpyAsync(d_curr_count, &combined, sizeof(int),
                                cudaMemcpyHostToDevice));
        if (W == 1) {
            size_t tb = sort_temp_bytes;
            cub::DeviceRadixSort::SortPairs(
                d_cub_temp, tb,
                d_dkeys_in, d_dkeys_out,
                d_idx_in,   d_idx_out,
                combined, 0, N);
        } else {
            V5_CUDA(cudaMemcpyAsync(d_idx_out, d_idx_in,
                                    combined * sizeof(int),
                                    cudaMemcpyDeviceToDevice));
            const uint64_t* dk_q = d_dkeys_in;
            int             lw_q = W;
            size_t tb = sort_temp_bytes;
            cub::DeviceMergeSort::SortKeys(
                d_cub_temp, tb,
                d_idx_out, combined,
                [dk_q, lw_q] __device__ (const int& a, const int& b) {
                    for (int w = 0; w < lw_q; w++) {
                        if (dk_q[a * lw_q + w] < dk_q[b * lw_q + w]) return true;
                        if (dk_q[a * lw_q + w] > dk_q[b * lw_q + w]) return false;
                    }
                    return false;
                });
        }

        // Step 3: Detect group boundaries.
        {
            int blocks = (combined + 255) / 256;
            if (blocks < 1) blocks = 1;
            V5DetectBoundariesKernel<<<blocks, 256>>>(
                d_dkeys_in, d_idx_out, d_curr_count, W, d_is_start);
        }

        // Step 4: ExclusiveSum group ids.
        {
            size_t sb = scan_temp_bytes;
            cub::DeviceScan::ExclusiveSum(
                d_cub_temp, sb,
                d_is_start, d_group_id,
                combined);
        }

        // Step 5: Compact group_starts + d_num_groups.
        V5_CUDA(cudaMemsetAsync(d_num_groups, 0, sizeof(int)));
        {
            int blocks = (combined + 255) / 256;
            if (blocks < 1) blocks = 1;
            V5CompactGroupStartsKernel<<<blocks, 256>>>(
                d_is_start, d_group_id, d_curr_count,
                d_group_starts, d_num_groups);
        }

        V5_CUDA(cudaMemcpyAsync(&h_pin_scratch[3], d_num_groups,
                                sizeof(int), cudaMemcpyDeviceToHost));
        V5_CUDA(cudaStreamSynchronize(0));
        int n_groups = h_pin_scratch[3];
        if (n_groups < 1) {
            std::fprintf(stderr,
                "[V5 TRUNC] layer=%d consolidate: 0 groups despite "
                "combined=%d. FAIL LOUD.\n", layer, combined);
            return false;
        }

        // Step 6: batched merge with dual-source + DRAM output spill.
        // Same logic as do_flush() except input is d_aggregator (pre1) +
        // d_layer_scratch (pre2) using the dual-source mode of V5MergeKernel.
        const int slots_per_group_safe_factor = 8;

        if (const char* p = std::getenv("SAG_V5_ZERO_TEMP")) {
            if (p[0] != '0') {
                V5_CUDA(cudaMemsetAsync(d_next, 0,
                                        (size_t)cap.layer_capacity * bps));
            }
        }

        long long total_merged_ll = 0;
        long long resident_count_ll = 0;
        long long spilled_count_ll = 0;
        int g_processed = 0;
        bool need_spill = ((long long)n_groups > cap.layer_capacity);
        if (need_spill) {
            // ensure_spill_initialized was called when we did the scratch
            // spill earlier; if it succeeded we have h_layer_spill_next.
            if (!h_layer_spill_next || !d_spill_staging) {
                std::fprintf(stderr,
                    "[V5 TRUNC] layer=%d consolidate: n_groups=%d > layer_cap=%d "
                    "but layer-spill buffers unavailable (spill init failed?). "
                    "FAIL LOUD.\n",
                    layer, n_groups, cap.layer_capacity);
                return false;
            }
        }

        while (g_processed < n_groups) {
            char* dest_buf = nullptr;
            long long dest_capacity_states = 0;
            bool dest_is_spill = false;

            if (resident_count_ll < cap.layer_capacity &&
                (long long)cap.layer_capacity - resident_count_ll
                    >= slots_per_group_safe_factor) {
                dest_buf = d_next + resident_count_ll * (long long)bps;
                dest_capacity_states =
                    (long long)cap.layer_capacity - resident_count_ll;
            } else {
                dest_buf = d_spill_staging;
                dest_capacity_states = spill_chunk_states;
                dest_is_spill = true;
                // iter400: try to grow ping-pong if at allocation limit.
                long long remaining_host =
                    (long long)h_layer_spill_per_buf_states
                    - (long long)next_spill_count_h - spilled_count_ll;
                if (remaining_host < 1
                    && host_spill_per_buf_bytes_alloced
                       < host_spill_per_buf_bytes) {
                    long long min_needed =
                        ((long long)next_spill_count_h + spilled_count_ll
                         + 1LL) * (long long)bps;
                    long long curr_bytes_used =
                        (long long)curr_spill_count_h * (long long)bps;
                    long long next_bytes_used =
                        ((long long)next_spill_count_h + spilled_count_ll)
                            * (long long)bps;
                    V5_CUDA(cudaStreamSynchronize(spill_stream));
                    (void)grow_layer_spill_buffers(
                        min_needed, curr_bytes_used, next_bytes_used);
                    remaining_host =
                        (long long)h_layer_spill_per_buf_states
                        - (long long)next_spill_count_h - spilled_count_ll;
                }
                if (remaining_host < 1) {
                    std::fprintf(stderr,
                        "[V5 TRUNC] layer=%d consolidate: host layer-spill "
                        "buffer exhausted (per-buf alloc %lld / cap %lld bytes; "
                        "%d states alloc, used %lld). "
                        "Increase SAG_V5_HOST_SPILL_GB. FAIL LOUD.\n",
                        layer,
                        host_spill_per_buf_bytes_alloced,
                        host_spill_per_buf_bytes,
                        h_layer_spill_per_buf_states,
                        (long long)next_spill_count_h + spilled_count_ll);
                    return false;
                }
                if (remaining_host < dest_capacity_states)
                    dest_capacity_states = remaining_host;
            }

            int batch_groups = (int)(dest_capacity_states /
                                     (long long)slots_per_group_safe_factor);
            if (batch_groups < 1) batch_groups = 1;
            if (batch_groups > n_groups - g_processed)
                batch_groups = n_groups - g_processed;

            if (dest_is_spill) {
                V5_CUDA(cudaStreamSynchronize(spill_stream));
            }

            V5_CUDA(cudaMemsetAsync(d_merge_count, 0, sizeof(int)));
            {
                int merge_grid =
                    (batch_groups + merge_warps_per_block - 1)
                        / merge_warps_per_block;
                if (merge_grid < 1) merge_grid = 1;
                // iter390: triple-source merge when third_source_count > 0;
                // dual-source otherwise (d_pre3=nullptr).
                const char* d_pre3_arg = use_triple ? d_curr
                                                    : (const char*)nullptr;
                int dest_cap_int =
                    (dest_capacity_states > (long long)INT_MAX)
                        ? INT_MAX : (int)dest_capacity_states;
                V5_LAUNCH_MERGE(merge_grid, merge_block_size, merge_smem,
                    d_aggregator, d_layer_scratch, /*pre1_count=*/(int)agg_count,
                    d_pre3_arg, /*pre2_end=*/(int)pre2_end_ll,
                    d_idx_out, d_group_starts,
                    g_processed, batch_groups, n_groups,
                    d_curr_count,
                    layout, N, M, W,
                    dest_buf, dest_cap_int,
                    d_merge_count, d_trunc, layer);
            }

            V5_CUDA(cudaMemcpyAsync(&h_pin_scratch[0], d_merge_count,
                                    sizeof(int), cudaMemcpyDeviceToHost));
            V5_CUDA(cudaMemcpyAsync(&h_pin_scratch[2], d_trunc,
                                    sizeof(int), cudaMemcpyDeviceToHost));
            V5_CUDA(cudaStreamSynchronize(0));
            int batch_merged = h_pin_scratch[0];
            int batch_tf     = h_pin_scratch[2];
            if (batch_merged < 0) batch_merged = 0;

            if (batch_tf != 0) {
                std::fprintf(stderr,
                    "[V5 TRUNC] layer=%d consolidate batch g=%d/%d: trunc "
                    "flag (F_count overflow likely). FAIL LOUD.\n",
                    layer, g_processed, n_groups);
                return false;
            }
            if ((long long)batch_merged > dest_capacity_states) {
                std::fprintf(stderr,
                    "[V5 TRUNC] layer=%d consolidate batch g=%d sz=%d: "
                    "emitted=%d exceeds dest_capacity=%lld. FAIL LOUD.\n",
                    layer, g_processed, batch_groups,
                    batch_merged, dest_capacity_states);
                return false;
            }

            if (dest_is_spill) {
                long long h_offset_states =
                    (long long)next_spill_count_h + spilled_count_ll;
                cudaError_t ce = cudaMemcpyAsync(
                    h_layer_spill_next + h_offset_states * bps,
                    d_spill_staging,
                    (size_t)batch_merged * bps,
                    cudaMemcpyDeviceToHost,
                    spill_stream);
                if (ce != cudaSuccess) {
                    std::fprintf(stderr,
                        "[V5 TRUNC] layer=%d consolidate: D2H spill copy "
                        "failed: %s. FAIL LOUD.\n",
                        layer, cudaGetErrorString(ce));
                    return false;
                }
                spilled_count_ll += batch_merged;
                total_spill_bytes_xferd += (long long)batch_merged * bps;
            } else {
                resident_count_ll += batch_merged;
            }

            g_processed += batch_groups;
            total_merged_ll += batch_merged;
        }

        int total_merged = (int)total_merged_ll;
        int resident_merged = (int)resident_count_ll;
        if (spill_stream != 0) {
            V5_CUDA(cudaStreamSynchronize(spill_stream));
        }

        next_spill_count_h += (int)spilled_count_ll;
        total_spill_states_seen += spilled_count_ll;

        // iter370: merge wrote directly to d_next; no swap needed.

        if (verbose) {
            std::fprintf(stderr,
                "[layer %d] FINAL consolidate (%s): combined=%d "
                "agg=%lld scratch_2nd=%lld curr_3rd=%lld groups=%d "
                "total_merged=%d (resident=%d spill=%lld) (dedup %.2fx)\n",
                layer, use_triple ? "triple-source" : "dual-source",
                combined, agg_count, second_source_count, third_source_count,
                n_groups, total_merged, resident_merged, spilled_count_ll,
                (double)combined / (double)(total_merged > 0 ? total_merged : 1));
        }

        new_cwm = resident_merged;
        return true;
    };

    // iter360: the spill paged-input cache (d_paged_input) holds at most
    // spill_chunk_states states from h_layer_spill_curr. Track which slice
    // [spill_chunk_base_h, spill_chunk_base_h + spill_chunk_loaded) is
    // currently resident across wave launches so we can avoid re-paging.
    int spill_chunk_base_h    = -1;  // -1 = no chunk loaded
    int spill_chunk_loaded    = 0;

    for (int layer = 0; layer < max_layers; ++layer) {
        layers_done = layer + 1;

        // RAM safety: per-layer self-monitor of host RAM usage. Aborts the
        // run BEFORE host-OOM if usage exceeds SAG_HOST_RAM_LIMIT_PCT
        // (default 90%). See feedback_ram_safety_cutoffs.md.
        if (!v5_check_host_ram_or_fail(layer)) {
            status = (int32_t)V5Status::TRUNC;
            break;
        }

        // iter360: total parent count = resident in d_curr + spilled in
        // h_layer_spill_curr. The wave loop iterates over a global parent
        // index in [0, total_parents).
        long long total_parents_ll =
            (long long)curr_count_h + (long long)curr_spill_count_h;
        // Reset the page-cache tracker at layer start (h_layer_spill_curr's
        // state is the prior layer's spill output, just freshly swapped in).
        spill_chunk_base_h = -1;
        spill_chunk_loaded = 0;

        // Reset next-layer spill accumulator.
        next_spill_count_h = 0;

        // iter370: reset scratch ring metadata at layer start.
        scratch_chunks.clear();
        scratch_ring_used_bytes = 0;

        // Number of waves needed for this layer.
        int cur_num_waves = (int)((total_parents_ll + cap.wave_states - 1)
                                  / cap.wave_states);
        if (cur_num_waves < 1) cur_num_waves = 1;

        if (verbose) {
            std::fprintf(stderr,
                "[layer %d] curr_resident=%d curr_spill=%d (total=%lld), "
                "waves=%d (wave_states=%d)\n",
                layer, curr_count_h, curr_spill_count_h,
                total_parents_ll, cur_num_waves, cap.wave_states);
        }

        // Per-layer running offset into d_layer_scratch. iter370 no longer
        // tracks cwm (mid-layer flush replaced by scratch-spill ring).
        int total_pre_cw_merged = 0;
        // Per-layer aggregates (verbose only).
        long long layer_pre_merge = 0;

        bool layer_done = false;

        for (int wave = 0; wave < cur_num_waves; ++wave) {
            long long wave_start_ll = (long long)wave * cap.wave_states;
            long long wave_remain_ll = total_parents_ll - wave_start_ll;
            int wave_count = cap.wave_states;
            if ((long long)wave_count > wave_remain_ll)
                wave_count = (int)wave_remain_ll;
            if (wave_count <= 0) break;

            // Determine the parent source for this wave. iter360: parents in
            // [0, curr_count_h) live in d_curr (resident). Parents in
            // [curr_count_h, total) live in h_layer_spill_curr.
            //
            // For simplicity (and to avoid inefficient cross-source waves),
            // we cap wave_count to keep the wave entirely in one source.
            char* wave_input_ptr = nullptr;
            if (wave_start_ll < (long long)curr_count_h) {
                // Resident slice. Cap wave to stay in d_curr.
                if (wave_start_ll + wave_count > (long long)curr_count_h) {
                    wave_count = (int)((long long)curr_count_h - wave_start_ll);
                }
                wave_input_ptr = d_curr + (size_t)wave_start_ll * bps;
            } else {
                // Spilled slice. Spill was generated last layer, so it must
                // be initialized.
                if (!spill_initialized || !d_paged_input || !h_layer_spill_curr) {
                    std::fprintf(stderr,
                        "[V5 INTERNAL] layer=%d wave=%d/%d: parent in spill "
                        "but spill state uninitialized. BUG.\n",
                        layer, wave+1, cur_num_waves);
                    status = (int32_t)V5Status::TRUNC;
                    layer_done = true;
                    break;
                }
                long long spill_start_ll = wave_start_ll - (long long)curr_count_h;
                int spill_start = (int)spill_start_ll;

                // Cap wave to spill_chunk_states (we can't span beyond a chunk
                // without re-paging mid-kernel).
                if (wave_count > spill_chunk_states)
                    wave_count = spill_chunk_states;

                // Determine which chunk this wave's parent index falls into.
                // We use chunks aligned to wave boundaries (chunk_base = wave_start
                // in spill coords) to keep the logic simple, but ensure we
                // load enough to cover the wave.
                int chunk_base = spill_start;
                int chunk_size = spill_chunk_states;
                if (chunk_size > curr_spill_count_h - chunk_base)
                    chunk_size = curr_spill_count_h - chunk_base;
                if (wave_count > chunk_size) wave_count = chunk_size;

                // Page in if not already resident.
                if (spill_chunk_base_h != chunk_base ||
                    spill_chunk_loaded < chunk_size) {
                    cudaError_t pe = cudaMemcpyAsync(
                        d_paged_input,
                        h_layer_spill_curr + (size_t)chunk_base * bps,
                        (size_t)chunk_size * bps,
                        cudaMemcpyHostToDevice,
                        page_stream);
                    if (pe != cudaSuccess) {
                        std::fprintf(stderr,
                            "[V5 TRUNC] layer=%d wave=%d/%d: H2D paging copy "
                            "failed: %s. FAIL LOUD.\n",
                            layer, wave+1, cur_num_waves, cudaGetErrorString(pe));
                        status = (int32_t)V5Status::TRUNC;
                        layer_done = true;
                        break;
                    }
                    total_spill_bytes_xferd += (long long)chunk_size * bps;
                    // Wait for the paging copy before we issue the wave's K1K2
                    // (which reads d_paged_input).
                    V5_CUDA(cudaStreamSynchronize(page_stream));
                    spill_chunk_base_h = chunk_base;
                    spill_chunk_loaded = chunk_size;
                }

                int wave_offset_in_chunk = spill_start - spill_chunk_base_h;
                wave_input_ptr = d_paged_input
                               + (size_t)wave_offset_in_chunk * bps;
            }
            if (layer_done) break;

            // Reset per-wave counters. d_counters layout:
            //   [0] output_count, [1] unsched (sticky), [2] trunc (sticky),
            //   [3] num_groups,   [4] merge_count,      [5] valid_count.
            // Indices 3-5 are contiguous per-wave-reset items; collapse
            // their three separate cudaMemsetAsync calls into ONE 12-byte
            // memset. Saves ~20 us of per-wave host-launch overhead.
            V5_CUDA(cudaMemsetAsync(d_output_count, 0, sizeof(int)));
            V5_CUDA(cudaMemsetAsync(d_num_groups,   0, 3 * sizeof(int)));
            // iter410: zero the IJP-info buffer so stale info from a
            // prior wave can't leak into this wave's K2 dispatch.
            if (ijp_enabled && d_ijp_info != nullptr) {
                V5_CUDA(cudaMemsetAsync(d_ijp_info, 0,
                    (size_t)wave_count * sizeof(IJPInfoV5)));
            }

            // ----- Cooperative launch: V5K1K2Kernel for this wave -----
            int n_arg = N, m_arg = M, w_arg = W;
            // Phase 5.x per-layer candidate truncation: at layer L, only
            // candidates with topo_depth <= L+1 can possibly be eligible
            // (deeper jobs have a precedence chain that hasn't been
            // fully dispatched yet). Truncate K1's scan to those.
            int nc_arg;
            {
                int dp_idx = layer + 2;
                int dp_max = (int)topo.depth_prefix.size() - 1;
                if (dp_idx > dp_max) dp_idx = dp_max;
                if (dp_idx < 0) dp_idx = 0;
                nc_arg = topo.depth_prefix[dp_idx];
                if (nc_arg < 0) nc_arg = 0;
                if (nc_arg > num_candidates) nc_arg = num_candidates;
            }
            int wave_max_out_arg = cap.wave_max_output;
            int wave_max_pairs_arg = cap.wave_max_pairs;
            int wave_count_arg = wave_count;
            int layer_dbg_arg = layer;

            void* kargs_default[] = {
                (void*)&layout,
                (void*)&n_arg, (void*)&m_arg, (void*)&w_arg,
                (void*)&d_TC, (void*)&d_PO, (void*)&d_Pred, (void*)&d_Succ,
                (void*)&d_r_min, (void*)&d_r_max,
                (void*)&d_C_min, (void*)&d_C_max, (void*)&d_deadline,
                (void*)&d_priority, (void*)&d_prio_order,
                (void*)&d_sus_min, (void*)&d_sus_max,
                (void*)&d_candidates, (void*)&nc_arg,
                (void*)&wave_input_ptr,    // d_layer_curr (sub-slice of d_curr)
                (void*)&wave_count_arg,    // num_states_in
                (void*)&d_wave_output,     // d_layer_next (per-wave output scratch)
                (void*)&wave_max_out_arg,  // layer_capacity (= per-wave output cap)
                (void*)&d_valid_pairs, (void*)&wave_max_pairs_arg,
                (void*)&d_valid_count, (void*)&d_output_count,
                (void*)&d_unsched, (void*)&d_trunc,
                (void*)&d_BCRT, (void*)&d_WCRT,
                (void*)&d_max_F_count,
                (void*)&layer_dbg_arg,
            };
            // IJP variant has an extra d_ijp_info argument before layer_dbg.
            void* kargs_ijp[] = {
                (void*)&layout,
                (void*)&n_arg, (void*)&m_arg, (void*)&w_arg,
                (void*)&d_TC, (void*)&d_PO, (void*)&d_Pred, (void*)&d_Succ,
                (void*)&d_r_min, (void*)&d_r_max,
                (void*)&d_C_min, (void*)&d_C_max, (void*)&d_deadline,
                (void*)&d_priority, (void*)&d_prio_order,
                (void*)&d_sus_min, (void*)&d_sus_max,
                (void*)&d_candidates, (void*)&nc_arg,
                (void*)&wave_input_ptr,
                (void*)&wave_count_arg,
                (void*)&d_wave_output,
                (void*)&wave_max_out_arg,
                (void*)&d_valid_pairs, (void*)&wave_max_pairs_arg,
                (void*)&d_valid_count, (void*)&d_output_count,
                (void*)&d_unsched, (void*)&d_trunc,
                (void*)&d_BCRT, (void*)&d_WCRT,
                (void*)&d_max_F_count,
                (void*)&d_ijp_info,
                (void*)&layer_dbg_arg,
            };
            // ECRTS 2022 variant: extra d_p_max argument right after d_sus_max,
            // plus Phase 3.3.b CSR cost map + per-pair side arrays just before
            // layer_dbg_arg. Matches V5K1K2KernelECRTS22's signature in
            // controller.cu.
            void* kargs_ecrts22[] = {
                (void*)&layout,
                (void*)&n_arg, (void*)&m_arg, (void*)&w_arg,
                (void*)&d_TC, (void*)&d_PO, (void*)&d_Pred, (void*)&d_Succ,
                (void*)&d_r_min, (void*)&d_r_max,
                (void*)&d_C_min, (void*)&d_C_max, (void*)&d_deadline,
                (void*)&d_priority, (void*)&d_prio_order,
                (void*)&d_sus_min, (void*)&d_sus_max,
                (void*)&d_p_max,
                (void*)&d_p_min,
                (void*)&d_candidates, (void*)&nc_arg,
                (void*)&wave_input_ptr,
                (void*)&wave_count_arg,
                (void*)&d_wave_output,
                (void*)&wave_max_out_arg,
                (void*)&d_valid_pairs, (void*)&wave_max_pairs_arg,
                (void*)&d_valid_count, (void*)&d_output_count,
                (void*)&d_unsched, (void*)&d_trunc,
                (void*)&d_BCRT, (void*)&d_WCRT,
                (void*)&d_max_F_count,
                (void*)&d_costmap_offset, (void*)&d_costmap_p,
                (void*)&d_costmap_cmin, (void*)&d_costmap_cmax,
                (void*)&d_pair_p, (void*)&d_pair_cmin, (void*)&d_pair_cmax,
                (void*)&layer_dbg_arg,
            };
            void** kargs;
            if (g_variant == sag::v5::V5Variant::GANG_22 &&
                !g_ecrts22_route_to_rtss24) {
                kargs = kargs_ecrts22;
            } else if (ijp_enabled) {
                kargs = kargs_ijp;
            } else {
                // rtss24/rtss17/ecrts19 OR ecrts22-routed-to-rtss24
                kargs = kargs_default;
            }

            // Adaptive grid sizing: cap grid at the smallest power-of-2
            // that covers num_states_in / V5_BLOCK_WARPS warps. For early
            // layers with few states, this drastically reduces unused
            // warps and the grid_sync time. Cooperative launch only
            // requires the chosen grid fit simultaneously on the device,
            // which the smaller grid trivially satisfies (it's <= the
            // device-max we pre-computed via cudaOccupancyMaxActiveBPM).
            int needed_warps = (wave_count + 3) / 4;  // V5_BLOCK_WARPS == 4
            int needed_blocks = (needed_warps + 0) > 0 ? needed_warps : 1;
            int adaptive_grid = (needed_blocks < k1k2_grid)
                                ? needed_blocks : k1k2_grid;
            if (adaptive_grid < 1) adaptive_grid = 1;
            cudaError_t le = cudaLaunchCooperativeKernel(
                k1k2_kernel_ptr,
                dim3(adaptive_grid, 1, 1), dim3(k1k2_block_size, 1, 1),
                kargs, k1k2_smem, 0);
            if (le != cudaSuccess) {
                fprintf(stderr,
                    "cudaLaunchCooperativeKernel(V5K1K2) failed: %s\n",
                    cudaGetErrorString(le));
                return 5;
            }

            // Read post-K2 state D2H. unsched/trunc are sticky so we read on
            // every wave to detect mid-layer termination. d_counters[0..2] =
            // {output_count, unsched, trunc} are contiguous; batch into a
            // single 12-byte memcpy.
            V5_CUDA(cudaMemcpyAsync(&h_pin_scratch[0], d_counters,
                                    3 * sizeof(int), cudaMemcpyDeviceToHost));
            V5_CUDA(cudaStreamSynchronize(0));

            int out_count = h_pin_scratch[0];
            int uf        = h_pin_scratch[1];
            int tf        = h_pin_scratch[2];

            if (verbose) {
                int valid_count_h;
                V5_CUDA(cudaMemcpy(&valid_count_h, d_valid_count, sizeof(int),
                                   cudaMemcpyDeviceToHost));
                std::fprintf(stderr,
                    "  wave %d/%d: ws=%d valid=%d out=%d uf=%d tf=%d\n",
                    wave + 1, cur_num_waves, wave_count, valid_count_h,
                    out_count, uf, tf);
            }

            if (uf != 0) {
                status = (int32_t)V5Status::UNSCHED;
                layer_done = true;
                break;
            }
            if (tf != 0) {
                // Per-wave K2 output overflowed wave_max_output -- this is a
                // sizing failure (should not happen if wave_max_output ==
                // wave_states * N, but defensive).
                std::fprintf(stderr,
                    "[V5 TRUNC] layer=%d wave=%d/%d: K2 output overflow "
                    "(wave_max_output=%d). FAIL LOUD.\n",
                    layer, wave+1, cur_num_waves, cap.wave_max_output);
                status = (int32_t)V5Status::TRUNC;
                layer_done = true;
                break;
            }
            if (out_count <= 0) {
                // No successors from this wave (all candidates filtered).
                continue;
            }
            if (out_count > cap.wave_max_output) out_count = cap.wave_max_output;

            layer_pre_merge += out_count;

            // ----- iter370: Scratch-ring spill trigger -----
            //
            // (POST-K1K2, PRE-sort/groups). Replaces iter360's mid-layer
            // concat-and-merge flush. Trigger: per-wave merge output bound
            // is `out_count` (worst case if every wave-output state is
            // unique). If total_pre_cw_merged + out_count > scratch_cap,
            // spill the entire current scratch tail to the host ring buffer
            // (D2H copy, no merge work) and reset total_pre_cw_merged=0.
            //
            // The scratch ring chunks accumulate through the layer; the
            // FINAL layer-end consolidate pass pages them all back together
            // and runs a single sort+detect+merge.
            //
            // Each state spilled at MOST ONCE per layer: O(N) PCIe out + O(N)
            // PCIe back = 2N total per layer (vs iter360's O(N*K) per layer).
            if (total_pre_cw_merged > 0 &&
                (long long)total_pre_cw_merged + out_count > cap.scratch_capacity) {
                if (!do_scratch_spill(layer, total_pre_cw_merged)) {
                    status = (int32_t)V5Status::TRUNC;
                    layer_done = true;
                    break;
                }
                total_pre_cw_merged = 0;
            }
            if (layer_done) break;

            // ----- Per-wave sort + group + merge -----
            //
            // The wave's merged output goes into d_layer_scratch at the
            // running offset total_pre_cw_merged. The cross-wave merge will
            // later dedupe duplicate D-keys that leak between waves.
            V5_CUDA(cudaMemcpyAsync(d_curr_count, &out_count,
                                    sizeof(int), cudaMemcpyHostToDevice));

            {
                int blocks = (out_count + 255) / 256;
                if (blocks < 1) blocks = 1;
                V5ExtractDKeysIotaKernel<<<blocks, 256>>>(
                    d_wave_output, layout, d_curr_count, W,
                    d_dkeys_in, d_idx_in);
            }

            if (W == 1) {
                size_t tb = sort_temp_bytes;
                cub::DeviceRadixSort::SortPairs(
                    d_cub_temp, tb,
                    d_dkeys_in, d_dkeys_out,
                    d_idx_in,   d_idx_out,
                    out_count, 0, N);
            } else {
                V5_CUDA(cudaMemcpyAsync(d_idx_out, d_idx_in,
                                        out_count * sizeof(int),
                                        cudaMemcpyDeviceToDevice));
                const uint64_t* dk_q = d_dkeys_in;
                int             lw_q = W;
                size_t tb = sort_temp_bytes;
                cub::DeviceMergeSort::SortKeys(
                    d_cub_temp, tb,
                    d_idx_out, out_count,
                    [dk_q, lw_q] __device__ (const int& a, const int& b) {
                        for (int w = 0; w < lw_q; w++) {
                            if (dk_q[a * lw_q + w] < dk_q[b * lw_q + w]) return true;
                            if (dk_q[a * lw_q + w] > dk_q[b * lw_q + w]) return false;
                        }
                        return false;
                    });
            }

            {
                int blocks = (out_count + 255) / 256;
                if (blocks < 1) blocks = 1;
                V5DetectBoundariesKernel<<<blocks, 256>>>(
                    d_dkeys_in, d_idx_out, d_curr_count, W,
                    d_is_start);
            }

            {
                size_t sb = scan_temp_bytes;
                cub::DeviceScan::ExclusiveSum(
                    d_cub_temp, sb,
                    d_is_start, d_group_id,
                    out_count);
            }

            {
                int blocks = (out_count + 255) / 256;
                if (blocks < 1) blocks = 1;
                V5CompactGroupStartsKernel<<<blocks, 256>>>(
                    d_is_start, d_group_id, d_curr_count,
                    d_group_starts, d_num_groups);
            }

            V5_CUDA(cudaMemcpyAsync(&h_pin_scratch[3], d_num_groups,
                                    sizeof(int), cudaMemcpyDeviceToHost));
            V5_CUDA(cudaStreamSynchronize(0));
            int ng = h_pin_scratch[3];
            if (ng < 1) {
                // out_count > 0 should imply ng >= 1.
                std::fprintf(stderr,
                    "[V5 TRUNC] layer=%d wave=%d/%d: empty group set despite "
                    "out_count=%d. FAIL LOUD.\n",
                    layer, wave+1, cur_num_waves, out_count);
                status = (int32_t)V5Status::TRUNC;
                layer_done = true;
                break;
            }

            // Reset the per-wave merge_count atomic counter.
            V5_CUDA(cudaMemsetAsync(d_merge_count, 0, sizeof(int)));

            // V5MergeKernel emits per-wave merged states into
            // d_layer_scratch + total_pre_cw_merged*bps. The kernel uses
            // atomicAdd(d_merge_count, 1) for slot allocation, so passing a
            // base pointer with offset places the wave's merged slice into
            // [total_pre_cw_merged .. total_pre_cw_merged + wave_merged).
            char* d_post_target = d_layer_scratch + (size_t)total_pre_cw_merged * bps;
            {
                int merge_grid =
                    (ng + merge_warps_per_block - 1) / merge_warps_per_block;
                if (merge_grid < 1) merge_grid = 1;
                // Per-wave merge: single-source (d_pre2 = nullptr, pre1_count = 0),
                // single-batch (group_offset=0, num_groups_in_batch=ng).
                long long wave_dest_room =
                    (long long)cap.scratch_capacity - (long long)total_pre_cw_merged;
                if (wave_dest_room < 0) wave_dest_room = 0;
                int wave_dest_cap_int =
                    (wave_dest_room > (long long)INT_MAX)
                        ? INT_MAX : (int)wave_dest_room;
                V5_LAUNCH_MERGE(merge_grid, merge_block_size, merge_smem,
                    d_wave_output, /*d_pre2=*/(const char*)nullptr, /*pre1_count=*/0,
                    /*d_pre3=*/(const char*)nullptr, /*pre2_end=*/0,
                    d_idx_out, d_group_starts,
                    /*group_offset=*/0, /*num_groups_in_batch=*/ng,
                    /*num_groups_total=*/ng,
                    d_curr_count,
                    layout, N, M, W,
                    d_post_target, wave_dest_cap_int,
                    d_merge_count, d_trunc, layer);
            }

            V5_CUDA(cudaMemcpyAsync(&h_pin_scratch[0], d_merge_count,
                                    sizeof(int), cudaMemcpyDeviceToHost));
            V5_CUDA(cudaMemcpyAsync(&h_pin_scratch[2], d_trunc,
                                    sizeof(int), cudaMemcpyDeviceToHost));
            V5_CUDA(cudaStreamSynchronize(0));
            int wave_merged = h_pin_scratch[0];
            int merge_tf   = h_pin_scratch[2];
            if (wave_merged < 0) wave_merged = 0;

            if (merge_tf != 0) {
                std::fprintf(stderr,
                    "[V5 TRUNC] layer=%d wave=%d/%d: per-wave merge set "
                    "trunc flag (likely F_count overflow). FAIL LOUD.\n",
                    layer, wave+1, cur_num_waves);
                status = (int32_t)V5Status::TRUNC;
                layer_done = true;
                break;
            }

            // The merge wrote into d_layer_scratch[total_pre_cw_merged ..
            // total_pre_cw_merged + wave_merged). Post-launch sanity check:
            // confirm the actual write fit within scratch_cap. If not, we
            // had a worse-case-than-expected wave (ng underestimated
            // wave_merged due to non-overlapping A/F intervals creating
            // many slots within a group). FAIL LOUD -- mid-layer flush
            // cannot recover from this since the write has already corrupted
            // memory beyond scratch_cap.
            if ((long long)total_pre_cw_merged + wave_merged > cap.scratch_capacity) {
                std::fprintf(stderr,
                    "[V5 TRUNC] layer=%d wave=%d/%d: scratch overflow post-merge "
                    "(total=%d + wave=%d > scratch_cap=%d). The pre-launch flush "
                    "trigger underestimated wave_merged; bump SAG_V5_SCRATCH_FRAC.\n",
                    layer, wave+1, cur_num_waves,
                    total_pre_cw_merged, wave_merged, cap.scratch_capacity);
                status = (int32_t)V5Status::TRUNC;
                layer_done = true;
                break;
            }

            total_pre_cw_merged += wave_merged;

            if (verbose) {
                std::fprintf(stderr,
                    "    wave %d: pre_merge_groups=%d wave_merged=%d total=%d "
                    "scratch_chunks=%zu\n",
                    wave + 1, ng, wave_merged, total_pre_cw_merged,
                    scratch_chunks.size());
            }
        } // end wave loop

        if (layer_done) break;

        if (verbose) {
            std::fprintf(stderr,
                "[layer %d] pre_merge=%lld total_pre_cw_merged=%d "
                "scratch_chunks=%zu (waves=%d)\n",
                layer, layer_pre_merge, total_pre_cw_merged,
                scratch_chunks.size(), cur_num_waves);
        }

        if (total_pre_cw_merged <= 0 && scratch_chunks.empty()) {
            // No successors at all -> SCHED.
            status = (int32_t)V5Status::EMPTY;
            break;
        }

        // ----- iter370 LAYER-END MERGE -----
        //
        // After all waves complete, run ONE merge over either:
        //   (a) d_layer_scratch[0..S) directly (if no spill happened), OR
        //   (b) d_aggregator (if spill happened) -- pages all chunks back +
        //       residual scratch, then runs the same sort+detect+merge.
        //
        // Each spilled state crosses PCIe at most twice per layer (once out
        // during scratch-ring spill, once back during consolidate paging).
        // No quadratic re-touch.
        //
        // Output: d_temp gets the deduplicated layer (resident portion);
        // h_layer_spill_next gets any overflow (n_groups > layer_cap).
        // After do_flush returns, d_next <-> d_temp are swapped so d_next
        // holds the new layer.
        int cross_wave_merged = 0;
        if (scratch_chunks.empty()) {
            // No mid-layer scratch spill. Standard single-source merge from
            // d_layer_scratch[0..total_pre_cw_merged).
            int new_cwm = 0;
            if (!do_flush(layer, d_layer_scratch, total_pre_cw_merged,
                          /*final=*/true, new_cwm)) {
                status = (int32_t)V5Status::TRUNC;
                break;
            }
            cross_wave_merged = new_cwm;
        } else {
            // Mid-layer scratch spill happened. Consolidate all chunks +
            // residual through d_aggregator + single sort+detect+merge.
            int new_cwm = 0;
            if (!do_layer_end_consolidate(layer, total_pre_cw_merged,
                                          new_cwm)) {
                status = (int32_t)V5Status::TRUNC;
                break;
            }
            cross_wave_merged = new_cwm;
        }

        if (cross_wave_merged <= 0 && next_spill_count_h <= 0) {
            // After all merges nothing remains (no resident, no spill) -> SCHED.
            status = (int32_t)V5Status::EMPTY;
            break;
        }

        // Swap d_curr <-> d_next for the next layer. d_temp stays put (will be
        // reused next layer's flushes).
        {
            char* tmp = d_curr;
            d_curr = d_next;
            d_next = tmp;
        }

        // iter360: swap host spill ping-pong. h_layer_spill_next becomes
        // h_layer_spill_curr (input for next layer); h_layer_spill_next is
        // reset to overwrite-from-zero next layer.
        if (host_spill_per_buf_bytes > 0) {
            char* tmp_h = h_layer_spill_curr;
            h_layer_spill_curr = h_layer_spill_next;
            h_layer_spill_next = tmp_h;
        }

        curr_count_h = cross_wave_merged;
        curr_spill_count_h = next_spill_count_h;

        if (verbose) {
            std::fprintf(stderr,
                "[layer %d] swap: next-layer curr_count=%d curr_spill=%d "
                "(total %lld)\n",
                layer, curr_count_h, curr_spill_count_h,
                (long long)curr_count_h + (long long)curr_spill_count_h);
        }
    }

    if (status == (int32_t)V5Status::UNKNOWN) {
        // Fell through max_layers without termination -- conservative TRUNC.
        status = (int32_t)V5Status::MAX_LAYER_EXCEEDED;
    }

    V5_CUDA(cudaEventRecord(ev_stop, 0));
    V5_CUDA(cudaDeviceSynchronize());
    cudaError_t pe = cudaGetLastError();
    if (pe != cudaSuccess) {
        fprintf(stderr, "Kernel runtime error: %s\n", cudaGetErrorString(pe));
        return 5;
    }

    float ms = 0.0f;
    V5_CUDA(cudaEventElapsedTime(&ms, ev_start, ev_stop));

    // ----- Read back BCRT / WCRT -----
    std::vector<int32_t> h_BCRT(N), h_WCRT(N);
    V5_CUDA(cudaMemcpy(h_BCRT.data(), d_BCRT, N * sizeof(int32_t), cudaMemcpyDeviceToHost));
    V5_CUDA(cudaMemcpy(h_WCRT.data(), d_WCRT, N * sizeof(int32_t), cudaMemcpyDeviceToHost));

    int h_max_F_count = 0;
    V5_CUDA(cudaMemcpy(&h_max_F_count, d_max_F_count, sizeof(int), cudaMemcpyDeviceToHost));
    std::printf("V5 sparse: max popcount(F_mask|X) observed = %d "
                "(F_MAX_PER_STATE = %d, headroom %d)\n",
                h_max_F_count, F_MAX_PER_STATE,
                F_MAX_PER_STATE - h_max_F_count);

    // ----- Print results -----
    bool schedulable = false;
    bool truncated = false;
    const char* status_str = "UNKNOWN";
    switch ((V5Status)status) {
        case V5Status::SCHED:
        case V5Status::EMPTY:
            schedulable = true;
            status_str = "YES";
            break;
        case V5Status::UNSCHED:
            schedulable = false;
            status_str = "NO";
            break;
        case V5Status::TRUNC:
            truncated = true;
            status_str = "TRUNCATED";
            break;
        case V5Status::MAX_LAYER_EXCEEDED:
            truncated = true;
            status_str = "TRUNCATED";
            break;
        default:
            status_str = "UNKNOWN";
            break;
    }

    std::printf("\n========================================\n");
    std::printf("   GPU SAG (framework iter380)\n");
    std::printf("========================================\n");
    std::printf("Schedulable:            %s\n", status_str);
    std::printf("Total layers:           %d\n", layers_done);
    std::printf("Scratch spill chunks:   %lld (states %lld)\n",
                total_scratch_spill_chunks, total_scratch_spill_states);
    std::printf("Controller wall:        %.4f ms\n", ms);
    std::printf("Jobs (n): %d, Cores (m): %d, W: %d, bps: %d\n", N, M, W, bps);
    std::printf("Spill: initialized=%s pinned=%s; total_spilled=%lld states "
                "(%.2f GB), total_xferd=%.2f GB\n",
                spill_initialized ? "yes" : "no",
                spill_pinned ? "yes" : "no",
                total_spill_states_seen,
                (double)(total_spill_states_seen * (long long)bps)
                  / (1024.0*1024.0*1024.0),
                (double)total_spill_bytes_xferd / (1024.0*1024.0*1024.0));

    if (por_enabled) {
        int dbg[8] = {0};
        V5_get_por_dbg(dbg, 8);
        long long p1     = dbg[0];
        long long total  = dbg[1];
        long long p6_ok  = dbg[2];
        long long p6_no  = dbg[3];
        long long single = dbg[4];
        long long zero   = dbg[5];
        auto pct = [total](long long x) {
            return (total > 0) ? 100.0 * (double)x / (double)total : 0.0;
        };
        std::printf("POR Phase B observation across %lld K1 states:\n", total);
        std::printf("  |ES| == 0 (terminal):           %lld  (%.2f%%)\n",
                    zero, pct(zero));
        std::printf("  |ES| == 1 (vacuous):            %lld  (%.2f%%)\n",
                    single, pct(single));
        std::printf("  |ES| >= 2 (P1 fires):           %lld  (%.2f%%)\n",
                    p1, pct(p1));
        std::printf("    of which 2 <= |ES| <= m (P6 ok):  %lld  (%.2f%%)\n",
                    p6_ok, pct(p6_ok));
        std::printf("    of which |ES| > m (P6 blocks):    %lld  (%.2f%%)\n",
                    p6_no, pct(p6_no));
        std::printf("  -> Phase C (Algorithm 4 greedy absorption) could\n"
                    "     potentially absorb the %.2f%% of states where\n"
                    "     P1+P6 pass into a single multi-job dispatch,\n"
                    "     subject to P3' / P4 / P5 (deferred to Phase C+).\n",
                    pct(p6_ok));
    }

    if (schedulable) {
        std::printf("\n--- BCRT/WCRT per job ---\n");
        for (int j = 0; j < N; j++) {
            if (h_BCRT[j] < INT32_MAX) {
                std::printf("  Job %d: BCRT=%d, WCRT=%d\n", j, h_BCRT[j], h_WCRT[j]);
            }
        }
    }

    if (schedulable && !truncated) {
        std::string rta = std::string(jobs_path);
        auto dot = rta.rfind('.');
        if (dot != std::string::npos) rta = rta.substr(0, dot);
        rta += ".v5.rta.csv";
        FILE* frta = std::fopen(rta.c_str(), "w");
        if (frta) {
            std::fprintf(frta, "Task ID, Job ID, BCCT, WCCT, BCRT, WCRT\n");
            for (int j = 0; j < N; j++) {
                int bcrt = (h_BCRT[j] < INT32_MAX) ? h_BCRT[j] : 0;
                int wcrt = h_WCRT[j];
                int bcct = bcrt + topo.r_min[j];
                int wcct = wcrt + topo.r_min[j];
                int task_id = (j < (int)topo.task_id.size()) ? topo.task_id[j] : 0;
                std::fprintf(frta, "%d, %d, %d, %d, %d, %d\n",
                             task_id, j, bcct, wcct, bcrt, wcrt);
            }
            std::fclose(frta);
            std::printf("Wrote %s\n", rta.c_str());
        }
    }

    cudaFree(d_layerA); cudaFree(d_layerB);
    cudaFree(d_layer_scratch);
    cudaFree(d_wave_output);
    cudaFree(d_valid_pairs);
    cudaFree(d_TC); cudaFree(d_PO); cudaFree(d_Pred); cudaFree(d_Succ);
    cudaFree(d_r_min); cudaFree(d_r_max);
    cudaFree(d_C_min); cudaFree(d_C_max); cudaFree(d_deadline);
    cudaFree(d_priority); cudaFree(d_prio_order);
    cudaFree(d_sus_min); cudaFree(d_sus_max);
    cudaFree(d_candidates);
    cudaFree(d_p_max);
    cudaFree(d_p_min);
    cudaFree(d_costmap_offset); cudaFree(d_costmap_p);
    cudaFree(d_costmap_cmin);   cudaFree(d_costmap_cmax);
    if (d_pair_p)    cudaFree(d_pair_p);
    if (d_pair_cmin) cudaFree(d_pair_cmin);
    if (d_pair_cmax) cudaFree(d_pair_cmax);
    cudaFree(d_BCRT); cudaFree(d_WCRT);
    // Single combined free for d_counters (replaces 8 individual frees:
    // valid/output/unsched/trunc/curr/merge/num_groups/max_F_count).
    cudaFree(d_counters);
    cudaFree(d_dkeys_in); cudaFree(d_dkeys_out);
    cudaFree(d_idx_in); cudaFree(d_idx_out);
    cudaFree(d_is_start); cudaFree(d_group_id); cudaFree(d_group_starts);
    cudaFree(d_cub_temp);
    if (d_ijp_info != nullptr) cudaFree(d_ijp_info);
    cudaFreeHost(h_pin_scratch);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    // iter370: free spill resources.
    if (spill_initialized) {
        if (h_layer_spill_curr) {
            if (spill_pinned) v5_mh_free(h_layer_spill_curr);
            else              std::free(h_layer_spill_curr);
        }
        if (h_layer_spill_next) {
            if (spill_pinned) v5_mh_free(h_layer_spill_next);
            else              std::free(h_layer_spill_next);
        }
        if (h_scratch_spill) {
            if (spill_pinned) v5_mh_free(h_scratch_spill);
            else              std::free(h_scratch_spill);
        }
        if (d_paged_input)   cudaFree(d_paged_input);
        if (d_spill_staging) cudaFree(d_spill_staging);
        if (spill_stream != 0) cudaStreamDestroy(spill_stream);
        if (page_stream  != 0) cudaStreamDestroy(page_stream);
    }
    // iter370: d_aggregator was eagerly allocated outside spill_initialized.
    if (d_aggregator) cudaFree(d_aggregator);

    return schedulable ? 0 : (truncated ? 6 : 7);
}
