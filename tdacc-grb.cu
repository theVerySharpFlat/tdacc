#include "GraphBLAS.h"
#include "tdb-grb.h"
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/pair.h>

#define GALAXY_GRAPH_A_BLOCK_SIZE 8192
#define GALAXY_GRAPH_JA_BLOCK_SIZE GALAXY_GRAPH_A_BLOCK_SIZE
#define GALAXY_GRAPH_IA_BLOCK_SIZE 8192

#define SQ(x) ((x) * (x))

struct GalaxyGraph {
    std::vector<System> A;    // values
    std::vector<uint32_t> IA; // accumulated non-sparse counts
    std::vector<uint32_t> JA; // columns of values in A
};

struct MarketInfoGPU {
    System *systems;
    Station *stations;
    ItemPricing *listings;
};

struct Problem {
    GrB_Index startSystem;
    GrB_Index endSystem;
    GrB_Index solutionOffset;
};

void printVector(const std::vector<std::pair<long, long>> &arr, int32_t len) {
    using namespace std;
    for (int i = 0; i < len; i++)
        cerr << " " << arr[i].first;

    cerr << endl;
}

int64_t cpu_computeMaxProfitWithStationIndices(int64_t srcStationIndex,
                                               int64_t dstStationIndex,
                                               const MarketInfo &marketInfo,
                                               int64_t space) {
    int64_t total = 0;
    int64_t profit = 0;

    thread_local std::vector<std::pair<int64_t, int64_t>> pq(1024);
    int profitsIndex = 0;
    for (int64_t i = 0, j = 0;
         i < marketInfo.stations[srcStationIndex].nListings &&
         j < marketInfo.stations[dstStationIndex].nListings;) {

        const ItemPricing &srcPricing =
            marketInfo.listings
                [marketInfo.stations[srcStationIndex].listingStartIndex + i];
        const ItemPricing &dstPricing =
            marketInfo.listings
                [marketInfo.stations[dstStationIndex].listingStartIndex + j];

        if (srcPricing.itemID == dstPricing.itemID) {
            int64_t profitPer = dstPricing.demandPrice - srcPricing.supplyPrice;
            if (profitPer > 0) {
                pq[profitsIndex].first = profitPer;
                pq[profitsIndex].second = srcPricing.supplyQuantity;
                profitsIndex++;
            }

            i++;
            j++;
        } else if (srcPricing.itemID < dstPricing.itemID) {
            i++;
        } else {
            j++;
        }
    }

    for (int i = 0; i < profitsIndex && total < space; i++) {
        bool swapped = false;
        for (int j = 0; j < profitsIndex - i - 1; j++) {
            if (pq[j].first > pq[j + 1].first) {
                std::swap(pq[j], pq[j + 1]);

                swapped = true;
            }
        }

        auto [profitPer, nAvailable] = pq[profitsIndex - i - 1];

        nAvailable = std::min(nAvailable, space - total);

        total += nAvailable;
        profit += nAvailable * profitPer;

        if (!swapped) {
            break;
        }
    }

    return profit;
};

__device__ uint64_t computeMaxProfitWithStationIndices(
    uint64_t srcStationIndex, uint64_t dstStationIndex,
    const MarketInfoGPU &marketInfo, uint64_t space) {
    uint64_t total = 0;
    uint64_t profit = 0;

    struct Pair {
        uint64_t first;
        uint64_t second;
    };

    // thread_local std::vector<std::pair<uint64_t, uint64_t>> pq(1024);
    Pair pq[1024];
    int profitsIndex = 0;
    for (uint64_t i = 0, j = 0;
         i < marketInfo.stations[srcStationIndex].nListings &&
         j < marketInfo.stations[dstStationIndex].nListings;) {

        const ItemPricing &srcPricing =
            marketInfo.listings
                [marketInfo.stations[srcStationIndex].listingStartIndex + i];
        const ItemPricing &dstPricing =
            marketInfo.listings
                [marketInfo.stations[dstStationIndex].listingStartIndex + j];

        if (srcPricing.itemID == dstPricing.itemID) {
            int64_t profitPer = dstPricing.demandPrice - srcPricing.supplyPrice;
            if (profitPer > 0) {
                pq[profitsIndex].first = profitPer;
                pq[profitsIndex].second = srcPricing.supplyQuantity;
                profitsIndex++;
            }

            i++;
            j++;
        } else if (srcPricing.itemID < dstPricing.itemID) {
            i++;
        } else {
            j++;
        }
    }

    for (int i = 0; i < profitsIndex && total < space; i++) {
        bool swapped = false;
        for (int j = 0; j < profitsIndex - i - 1; j++) {
            if (pq[j].first > pq[j + 1].first) {
                auto temp = pq[j];
                pq[j] = pq[j + 1];
                pq[j + 1] = temp;
                swapped = true;
            }
        }

        auto [profitPer, nAvailable] = pq[profitsIndex - i - 1];

        nAvailable = min(nAvailable, space - total);

        total += nAvailable;
        profit += nAvailable * profitPer;

        if (!swapped) {
            break;
        }
    }

    return profit;
};

__device__ int traverse(const MarketInfoGPU marketInfo,
                        uint64_t startSystemIndex, uint64_t endSystemIndex,
                        uint64_t solutionsStart, int capacity,
                        uint64_t *gpuSolutionVals, uint64_t *gpuSolutionSrcs,
                        uint64_t *gpuSolutionDsts) {
    // BitRange visitedSet(marketInfo.systems.size());

    struct Node {
        uint64_t index;
        uint64_t depth;
    };
    uint64_t srcEndStationIndex =
        marketInfo.systems[startSystemIndex].stationStartIndex +
        marketInfo.systems[startSystemIndex].nStations;
    uint64_t srcStartStationIndex =
        marketInfo.systems[startSystemIndex].stationStartIndex;
    uint64_t dstStartStationIndex =
        marketInfo.systems[endSystemIndex].stationStartIndex;
    uint64_t dstEndStationIndex =
        marketInfo.systems[endSystemIndex].stationStartIndex +
        marketInfo.systems[endSystemIndex].nStations;

    // int i = blockDim.x * blockIdx.x + threadIdx.x;
    // gpuSolutionSrcs[i] = (srcEndStationIndex - srcStartStationIndex) *
    //                      (dstEndStationIndex - dstStartStationIndex);

    for (uint64_t srcStationIndex =
             marketInfo.systems[startSystemIndex].stationStartIndex;
         srcStationIndex < srcEndStationIndex; srcStationIndex++) {
        for (uint64_t dstStationIndex =
                 marketInfo.systems[endSystemIndex].stationStartIndex;
             dstStationIndex < dstEndStationIndex; dstStationIndex++) {
            uint64_t profit = computeMaxProfitWithStationIndices(
                srcStationIndex, dstStationIndex, marketInfo, capacity);

            uint64_t solutionIndex =
                solutionsStart +
                (dstEndStationIndex - dstStartStationIndex) *
                    (srcStationIndex - srcStartStationIndex) +
                (dstStationIndex - dstStartStationIndex);
            gpuSolutionVals[solutionIndex] = profit;
            gpuSolutionSrcs[solutionIndex] = srcStationIndex;
            gpuSolutionDsts[solutionIndex] = dstStationIndex;
        }
    }

    return 0;
}

inline uint64_t computeProblemSolutionCount(const MarketInfo &marketInfo,
                                            uint64_t startSystemIndex,
                                            uint64_t endSystemIndex) {
    // BitRange visitedSet(marketInfo.systems.size());

    struct Node {
        uint64_t index;
        uint64_t depth;
    };
    uint64_t srcStartStationIndex =
        marketInfo.systems[startSystemIndex].stationStartIndex;
    uint64_t srcEndStationIndex =
        srcStartStationIndex + marketInfo.systems[startSystemIndex].nStations;
    uint64_t dstStartStationIndex =
        marketInfo.systems[endSystemIndex].stationStartIndex;
    uint64_t dstEndStationIndex =
        dstStartStationIndex + marketInfo.systems[endSystemIndex].nStations;

    return (srcEndStationIndex - srcStartStationIndex) *
           (dstEndStationIndex - dstStartStationIndex);
}

__global__ void kernel(Problem *problems, MarketInfoGPU gpuData,
                       uint64_t *gpuSolutionVals, uint64_t *gpuSolutionSrcs,
                       uint64_t *gpuSolutionDsts, uint64_t nProblems) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= nProblems) {
        return;
    }

    traverse(gpuData, problems[i].startSystem, problems[i].endSystem,
             problems[i].solutionOffset, 712, gpuSolutionVals, gpuSolutionSrcs,
             gpuSolutionDsts);
}

typedef struct {
    int64_t k;
    uint64_t v;
} tuple_u64;
#define U64_K "typedef struct { int64_t k ; uint64_t v ; } tuple_u64 ;"
void make_tuple_u64(void *z, const void *x, GrB_Index ix, GrB_Index jx,
                    const void *y, GrB_Index iy, GrB_Index jy,
                    const void *theta) {
    fprintf(stdout, "considering (%zu, %zu) -> %zu, (%zu, %zu) -> %zu\n", ix,
            jx, *(uint64_t *)x, iy, jy, *(uint64_t *)y);
    ((tuple_u64 *)z)->k = (int64_t)jx;
    ((tuple_u64 *)z)->v = ((tuple_u64 *)x)->v + *(uint64_t *)y;
}

// #define TUPLE_U64_MAX
void max_tuple_u64(void *in_z, const void *in_x, const void *in_y) {
    tuple_u64 *z = (tuple_u64 *)in_z;
    tuple_u64 *x = (tuple_u64 *)in_x;
    tuple_u64 *y = (tuple_u64 *)in_y;

    if (x->v > y->v || (x->v == y->v && x->k < y->k)) {
        z->k = x->k;
        z->v = x->v;
    } else {
        z->k = y->k;
        z->v = y->v;
    }
}

void tupleZeros(void *z, const void *x, GrB_Index i, GrB_Index j,
                const void *y) {
    ((tuple_u64 *)z)->k = i;
    ((tuple_u64 *)z)->v = 0;
}

int main(int argc, char *argv[]) {
    double maxJump = std::stod(argv[1]);
    int nJumps = std::stoi(argv[2]);
    int nHops = std::stoi(argv[3]);
    int capacity = std::stoi(argv[4]);

    std::cout << "jump range: " << maxJump << std::endl;
    std::cout << "jumps: " << nJumps << std::endl;
    std::cout << "hops: " << nHops << std::endl;
    std::cout << "capacity: " << capacity << std::endl;

    MarketInfo marketInfo;
    TDB tdb("data/TradeDangerous.db");
    if (tdb.loadMarketInfo(&marketInfo)) {
        exit(EXIT_FAILURE);
    }

    int initErr;
    if ((initErr = GrB_init(GrB_BLOCKING)) != GrB_SUCCESS) {
        fprintf(stderr, "failed to start graphblas: %d\n", initErr);
        exit(EXIT_FAILURE);
    }

    GrB_Matrix galaxyGraph;
    GrB_Matrix_new(&galaxyGraph, GrB_BOOL, marketInfo.systems.size(),
                   marketInfo.systems.size());
    {
        std::vector<GrB_Index> rows;
        std::vector<GrB_Index> cols;
        std::vector<int32_t> vals;

        rows.reserve(marketInfo.systems.size() * 16);
        cols.reserve(marketInfo.systems.size() * 16);
        vals.reserve(marketInfo.systems.size() * 16);

        for (int i = 0; i < marketInfo.systems.size(); i++) {
            for (int j = i + 1; j < marketInfo.systems.size(); j++) {
                double sqDist =
                    SQ(marketInfo.systems[i].x - marketInfo.systems[j].x) +
                    SQ(marketInfo.systems[i].y - marketInfo.systems[j].y) +
                    SQ(marketInfo.systems[i].z - marketInfo.systems[j].z);

                if (sqDist <= SQ(maxJump)) {
                    rows.push_back(i);
                    cols.push_back(j);
                    vals.push_back(1);

                    rows.push_back(j);
                    cols.push_back(i);
                    vals.push_back(1);

                    // if ((marketInfo.systems[i].id == 3107576615642 ||
                    //      marketInfo.systems[j].id == 3107576615642) &&
                    //     ((marketInfo.systems[i].id == 1733119972058 ||
                    //       marketInfo.systems[j].id == 1733119972058))) {
                    //     fprintf(stderr, "bad!\n");
                    // }
                }
            }
        }

        GrB_Matrix_build_INT32(galaxyGraph, rows.data(), cols.data(),
                               vals.data(), vals.size(), GrB_SECOND_BOOL);
    }

    int32_t solIndex = -1;
    for (int32_t i = 0; i < marketInfo.systems.size(); i++) {
        if (marketInfo.systems[i].id == 10477373803) {
            solIndex = i;
            break;
        }
    }

    if (solIndex < 0) {
        fprintf(stderr, "failed to find sol!\n");
        return 1;
    } else {
        fprintf(stderr, "found sol at index %d\n", solIndex);
    }

    GrB_Vector q, v;
    GrB_Vector_new(&q, GrB_BOOL, marketInfo.systems.size());
    GrB_Vector_new(&v, GrB_INT32, marketInfo.systems.size());

    GrB_Vector_setElement_BOOL(q, true, solIndex);

    int32_t level = 0;
    GrB_Index ok = false;
    do {
        ++level;

        GrB_Vector_apply_BinaryOp2nd_INT32(
            v, GrB_NULL, GrB_PLUS_INT32, GrB_SECOND_INT32, q, level, GrB_NULL);
        GrB_vxm(q, v, GrB_NULL, GrB_LOR_LAND_SEMIRING_BOOL, q, galaxyGraph,
                GrB_DESC_RC);
        GrB_Vector_nvals(&ok, q);

        // GxB_Vector_fprint(q, "q", GxB_COMPLETE, stdout);

        fprintf(stderr, "level: %d\n", level);
    } while (ok && level < nHops * nJumps);

    GrB_Vector_apply_BinaryOp2nd_INT32(v, GrB_NULL, GrB_PLUS_INT32,
                                       GrB_SECOND_INT32, q, ++level, GrB_NULL);

    GrB_Descriptor desc;
    GrB_Descriptor_new(&desc);
    GrB_Descriptor_set_INT32(desc, GrB_OUTP, GrB_REPLACE);
    GrB_Descriptor_set_INT32(desc, GrB_INP1, GrB_TRAN);

    GrB_Matrix vMat;
    GrB_Matrix_new(&vMat, GrB_BOOL, marketInfo.systems.size(), 1);
    GrB_Info info =
        GrB_Col_assign(vMat, GrB_NULL, GrB_NULL, v, GrB_ALL, 0, 0, GrB_DESC_R);

    GrB_Matrix simplified;
    GrB_Matrix_new(&simplified, GrB_BOOL, marketInfo.systems.size(),
                   marketInfo.systems.size());
    info = GrB_mxm(simplified, galaxyGraph, GrB_NULL,
                   GrB_LAND_LOR_SEMIRING_BOOL, vMat, vMat, desc);
    fprintf(stderr, "info: %d\n", info);

    // GxB_Vector_fprint(v, "v", GxB_COMPLETE, stdout);
    // GxB_Matrix_fprint(vMat, "vMat", GxB_COMPLETE, stdout);
    GxB_Matrix_fprint(simplified, "simplified", GxB_SUMMARY, stdout);
    GxB_Matrix_fprint(galaxyGraph, "galaxyGraph", GxB_SUMMARY, stdout);

    GrB_Matrix_free(&galaxyGraph);
    GrB_Matrix_free(&vMat);

    galaxyGraph = simplified;

    GxB_Iterator iterator;
    GxB_Iterator_new(&iterator);

    info = GxB_Matrix_Iterator_attach(iterator, galaxyGraph, NULL);
    info = GxB_Matrix_Iterator_seek(iterator, 0);
    FILE *f = fopen("galaxy.dot", "w");
    fprintf(f, "graph galaxy {");
    while (info != GxB_EXHAUSTED) {
        // get the entry A(i,j)
        GrB_Index i, j;
        GxB_Matrix_Iterator_getIndex(iterator, &i, &j);
        bool aij = GxB_Iterator_get_BOOL(iterator);

        fprintf(f, "\t%zu -- %zu;\n", i, j);

        // move to the next entry in A
        info = GxB_Matrix_Iterator_next(iterator);
    }
    fprintf(f, "}");
    fclose(f);

    GrB_Matrix a;
    GrB_Matrix_dup(&a, galaxyGraph);

    GrB_Matrix b;
    GrB_Matrix_new(&b, GrB_BOOL, marketInfo.systems.size(),
                   marketInfo.systems.size());

    GrB_Matrix c;
    GrB_Matrix_dup(&c, galaxyGraph);

    for (int i = 0; i < nJumps - 1; i++) {
        GrB_Info info = GrB_mxm(b, GrB_NULL, GrB_NULL,
                                GrB_LOR_LAND_SEMIRING_BOOL, a, c, GrB_NULL);
        fprintf(stderr, "info: %d\n", info);

        GrB_Matrix_eWiseAdd_BinaryOp(galaxyGraph, GrB_NULL, GrB_SECOND_BOOL,
                                     GrB_LOR, galaxyGraph, b, GrB_NULL);

        std::swap(a, b);
        GrB_Matrix_clear(b);
    }

    GxB_Matrix_fprint(galaxyGraph, "galaxyGraph", GxB_SUMMARY, stdout);

    std::vector<Problem> hostProblems;
    hostProblems.reserve(1e9);
    info = GxB_Matrix_Iterator_attach(iterator, galaxyGraph, NULL);
    info = GxB_Matrix_Iterator_seek(iterator, 0);
    size_t solutionCount = 0;
    while (info != GxB_EXHAUSTED) {
        // get the entry A(i,j)
        GrB_Index i, j;
        GxB_Matrix_Iterator_getIndex(iterator, &i, &j);
        //
        hostProblems.push_back(Problem{i, j, solutionCount});
        //
        solutionCount += computeProblemSolutionCount(marketInfo, i, j);

        // move to the next entry in A
        info = GxB_Matrix_Iterator_next(iterator);
    }
    // GxB_Iterator_free(&iterator);

    fprintf(stderr, "nproblems: %d\n", hostProblems.size());

    MarketInfoGPU gpuData{};
// gpuData.listings = marketInfo.listings;
// gpuData.stations = marketInfo.stations;
// gpuData.systems = marketInfo.systems;
#undef NDEBUG
    // GPU LISTINGS
    cudaError err = cudaMalloc(&gpuData.listings, marketInfo.listings.size() *
                                                      sizeof(ItemPricing));
    assert(err == cudaSuccess);
    fprintf(stderr, "info: %d\n", err);

    err = cudaMemcpy(gpuData.listings, (void *)marketInfo.listings.data(),
                     marketInfo.listings.size() * sizeof(ItemPricing),
                     cudaMemcpyHostToDevice);
    assert(err == cudaSuccess);
    fprintf(stderr, "info: %d\n", err);

    // GPU STATIONS
    err = cudaMalloc(&gpuData.stations,
                     marketInfo.stations.size() * sizeof(Station));
    assert(err == cudaSuccess);
    fprintf(stderr, "info: %d\n", err);

    err = cudaMemcpy(gpuData.stations, (void *)marketInfo.stations.data(),
                     marketInfo.stations.size() * sizeof(Station),
                     cudaMemcpyHostToDevice);
    assert(err == cudaSuccess);
    fprintf(stderr, "info: %d\n", err);

    // GPU SYSTEMS
    err = cudaMalloc(&gpuData.systems,
                     marketInfo.systems.size() * sizeof(System));
    assert(err == cudaSuccess);
    fprintf(stderr, "info: %d\n", err);

    err = cudaMemcpy(gpuData.systems, (void *)marketInfo.systems.data(),
                     marketInfo.systems.size() * sizeof(System),
                     cudaMemcpyHostToDevice);
    assert(err == cudaSuccess);
    fprintf(stderr, "info: %d\n", err);

    fflush(stderr);

    Problem *problems;
    err = cudaMalloc(&problems, hostProblems.size() * sizeof(Problem));
    assert(err == cudaSuccess);
    fprintf(stderr, "info: %d\n", err);
    err = cudaMemcpy(problems, hostProblems.data(),
                     hostProblems.size() * sizeof(Problem),
                     cudaMemcpyHostToDevice);
    assert(err == cudaSuccess);
    fprintf(stderr, "info: %d\n", err);

    fprintf(stderr, "solutions: %zu\n", solutionCount);
    uint64_t *gpuSolutionVals;
    uint64_t *gpuSolutionSrcs;
    uint64_t *gpuSolutionDsts;
    err = cudaMalloc(&gpuSolutionVals, solutionCount * sizeof(uint64_t));
    err = cudaMalloc(&gpuSolutionSrcs, solutionCount * sizeof(uint64_t));
    err = cudaMalloc(&gpuSolutionDsts, solutionCount * sizeof(uint64_t));

    fprintf(stderr, "running cuda...\n");
    kernel<<<hostProblems.size() / 128 + 1, 128>>>(
        problems, gpuData, gpuSolutionVals, gpuSolutionSrcs, gpuSolutionDsts,
        hostProblems.size());
    fprintf(stderr, "cuda done\n");
    err = cudaGetLastError();
    fprintf(stderr, "info: %d\n", err);

    uint64_t *solutionVals;
    uint64_t *solutionSrcs;
    uint64_t *solutionDsts;
    solutionVals = (uint64_t *)calloc(1, solutionCount * sizeof(uint64_t));
    solutionSrcs = (uint64_t *)calloc(1, solutionCount * sizeof(uint64_t));
    solutionDsts = (uint64_t *)calloc(1, solutionCount * sizeof(uint64_t));

    cudaDeviceSynchronize();
    cudaMemcpy(solutionVals, gpuSolutionVals, solutionCount * sizeof(uint64_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(solutionSrcs, gpuSolutionSrcs, solutionCount * sizeof(uint64_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(solutionDsts, gpuSolutionDsts, solutionCount * sizeof(uint64_t),
               cudaMemcpyDeviceToHost);
    err = cudaGetLastError();
    fprintf(stderr, "memcpy info: %d\n", err);

    int nonzeroSolutions = 0;
    for (int i = 0; i < solutionCount; i++) {
        if (solutionVals[i]) {
            nonzeroSolutions++;
            // fprintf(stderr, "(%zu, %zu) -> %zu\n", solutionSrcs[i],
            //         solutionDsts[i], solutionVals[i]);
        }
    }

    fprintf(stderr, "nonzerosolution: %d\n", nonzeroSolutions);

    GrB_Matrix solutionMat;
    GrB_Matrix_new(&solutionMat, GrB_UINT64, marketInfo.stations.size(),
                   marketInfo.stations.size());
    GrB_Info grbErr = GrB_Matrix_build_UINT64(
        solutionMat, (GrB_Index *)solutionSrcs, (GrB_Index *)solutionDsts,
        (GrB_Index *)solutionVals, solutionCount, GrB_SECOND_UINT64);
    grbErr = GrB_Matrix_select_UINT64(solutionMat, GrB_NULL, GrB_NULL,
                                      GrB_VALUENE_UINT64, solutionMat, 0,
                                      GrB_DESC_R);
    fprintf(stderr, "extractTuples error: %d\n", grbErr);

    // GxB_Matrix_fprint(solutionMat, "solutionmat", GxB_COMPLETE, stdout);

    std::vector<uint64_t> solStationIndices;
    std::vector<uint64_t> solStationVals;
    for (uint64_t i = marketInfo.systems[solIndex].stationStartIndex;
         i < marketInfo.systems[solIndex].stationStartIndex +
                 marketInfo.systems[solIndex].nStations;
         i++) {
        solStationIndices.push_back(i);
        solStationVals.push_back(1);
    }

    GrB_Type Tuple;
    GxB_Type_new(&Tuple, sizeof(tuple_u64), "tuple_u64", U64_K);
    GxB_IndexBinaryOp Iop;
    GrB_BinaryOp Bop, MonOp;
    GrB_Scalar scalar;
    GrB_Scalar_new(&scalar, GrB_UINT64);
    GrB_Scalar_setElement_UINT64(scalar, 0);

    assert(GxB_IndexBinaryOp_new(&Iop, make_tuple_u64, Tuple, Tuple, GrB_UINT64,
                                 GrB_UINT64, "make_u64",
                                 GrB_NULL) == GrB_SUCCESS);

    assert(GxB_BinaryOp_new_IndexOp(&Bop, Iop, scalar) == GrB_SUCCESS);
    assert(GxB_BinaryOp_new(&MonOp, max_tuple_u64, Tuple, Tuple, Tuple,
                            GrB_NULL, GrB_NULL) == GrB_SUCCESS);
    GrB_Monoid Monoid;
    GrB_Semiring Semiring;

    tuple_u64 identity;
    identity.k = INT64_MAX;
    identity.v = 0;

    assert(GrB_Monoid_new_UDT(&Monoid, MonOp, &identity) == GrB_SUCCESS);
    assert(GrB_Semiring_new(&Semiring, Monoid, Bop) == GrB_SUCCESS);

    GrB_Vector positionVector;
    assert(GrB_Vector_new(&positionVector, GrB_UINT64,
                          marketInfo.stations.size()) == GrB_SUCCESS);
    assert(GrB_Vector_build_UINT64(
               positionVector, (GrB_Index *)solStationIndices.data(),
               (GrB_Index *)solStationVals.data(), solStationIndices.size(),
               GrB_NULL) == GrB_SUCCESS);

    GrB_Vector netProfitVector;
    GrB_Vector_new(&netProfitVector, Tuple, marketInfo.stations.size());

    GrB_Vector profitVector;
    GrB_Vector_new(&profitVector, Tuple, marketInfo.stations.size());

    GxB_Matrix_fprint(solutionMat, "solutionmat", GxB_SUMMARY, stdout);
    GxB_Vector_fprint(positionVector, "positionvector", GxB_SUMMARY, stdout);
    std::vector<GrB_Vector> netProfitVectors;

    GrB_IndexUnaryOp TupleZeros;
    GrB_IndexUnaryOp_new(&TupleZeros, tupleZeros, Tuple, GrB_UINT64,
                         GrB_UINT64);

    fprintf(stderr, "ret: %d\n",
            GrB_Vector_apply_IndexOp_UDT(profitVector, positionVector, GrB_NULL,
                                         TupleZeros, positionVector, &identity,
                                         GrB_DESC_R));

    std::vector<GrB_Vector> positionVectors;

    {
        GrB_Vector dupPositionVec;
        GrB_Vector_dup(&dupPositionVec, positionVector);
        positionVectors.push_back(dupPositionVec);
    }

    for (int i = 0; i < nHops; i++) {
        fprintf(stderr, "vxm: %d\n",
                GrB_vxm(profitVector, GrB_NULL, GrB_NULL, Semiring,
                        profitVector, solutionMat, GrB_DESC_R));
        GxB_Vector_fprint(profitVector, "profitvector", GxB_COMPLETE, stdout);

        GrB_Vector_apply_BinaryOp2nd_UINT64(positionVector, GrB_NULL, GrB_NULL,
                                            GrB_SECOND_UINT64, profitVector, 1,
                                            GrB_DESC_R);
        {
            fprintf(stdout, "profit vector pre\n");
            info = GxB_Vector_Iterator_attach(iterator, profitVector, NULL);
            info = GxB_Vector_Iterator_seek(iterator, 0);
            while (info != GxB_EXHAUSTED) {
                // get the entry A(i,j)
                GrB_Index i = GxB_Vector_Iterator_getIndex(iterator);
                // uint64_t val = GxB_Iterator_get_UINT64(iterator);
                tuple_u64 t;
                GxB_Iterator_get_UDT(iterator, &t);
                fprintf(stdout, "(%zu, %zu) -> (%zu, %zu)\n", i, 0UL, t.k, t.v);

                // move to the next entry in A
                info = GxB_Matrix_Iterator_next(iterator);
            }
        }
        GrB_Vector_eWiseAdd_BinaryOp(netProfitVector, GrB_NULL, GrB_NULL, MonOp,
                                     netProfitVector, profitVector, GrB_NULL);
        GxB_Vector_fprint(netProfitVector, "net profit vector", GxB_COMPLETE,
                          stdout);

        GrB_Vector dupNetVec;
        GrB_Vector_dup(&dupNetVec, netProfitVector);
        netProfitVectors.push_back(dupNetVec);

        GrB_Vector dupPositionVec;
        GrB_Vector_dup(&dupPositionVec, positionVector);
        positionVectors.push_back(dupPositionVec);
    }

    for (int i = 0; i < nHops - 1; i++) {
        GrB_Vector profitVector;
        GrB_Vector_new(&profitVector, GrB_UINT64, marketInfo.stations.size());
    }

    tuple_u64 maxProfit = {};
    GrB_Vector_reduce_UDT(&maxProfit, GrB_NULL, Monoid, netProfitVector,
                          GrB_NULL);

    fprintf(stderr, "NET PROFIT: %zu, %zu\n", maxProfit.k, maxProfit.v);

    info = GxB_Vector_Iterator_attach(iterator, netProfitVector, NULL);
    info = GxB_Vector_Iterator_seek(iterator, 0);
    int64_t nextIndex = 0;
    while (info != GxB_EXHAUSTED) {
        // get the entry A(i,j)
        GrB_Index i = GxB_Vector_Iterator_getIndex(iterator);
        tuple_u64 val;
        GxB_Iterator_get_UDT(iterator, &val);
        if (val.v == maxProfit.v) {
            fprintf(stderr, "solution at index %zu\n", i);

            fprintf(stderr, "station index: %zu\n", marketInfo.stations[i].id);
            fprintf(stderr, "\n");

            nextIndex = val.k;
            break;
        }

        // move to the next entry in A
        info = GxB_Matrix_Iterator_next(iterator);
    }

    for (int i = nHops - 2; i >= 0; i--) {
        info = GxB_Vector_Iterator_attach(iterator, netProfitVectors[i], NULL);
        info = GxB_Vector_Iterator_seek(iterator, 0);
        while (info != GxB_EXHAUSTED) {
            // get the entry A(i,j)
            GrB_Index i = GxB_Vector_Iterator_getIndex(iterator);
            tuple_u64 val;
            GxB_Iterator_get_UDT(iterator, &val);
            if (i == nextIndex) {
                fprintf(stderr, "from station at index %zu (profit=%zu)\n", i,
                        val.v);

                fprintf(stderr, "station index: %zu\n",
                        marketInfo.stations[i].id);
                fprintf(stderr, "\n");

                nextIndex = val.k;
                break;
            }

            // move to the next entry in A
            info = GxB_Matrix_Iterator_next(iterator);
        }
    }

    return 0;

    //
    // {
    //     GrB_Vector currentVec;
    //     GrB_Vector_new(&currentVec, GrB_UINT64, marketInfo.stations.size());
    //     GrB_Vector_select_UINT64(currentVec, GrB_NULL, GrB_NULL,
    //                              GrB_VALUEEQ_UINT64, netProfitVector,
    //                              maxProfit, GrB_DESC_R);
    //     GxB_Vector_fprint(currentVec, "currentVec", GxB_COMPLETE, stdout);
    //
    //     for (int i = nHops - 2; i >= 0; i--) {
    //         fprintf(stderr, "vxm: %d\n",
    //                 GrB_vxm(currentVec, netProfitVectors[i], GrB_NULL,
    //                         GrB_MAX_FIRST_SEMIRING_UINT64, currentVec,
    //                         solutionMat, GrB_DESC_R));
    //         uint64_t hopMax;
    //
    //         GrB_Vector_select_UINT64(currentVec, currentVec, GrB_NULL,
    //                                  GrB_VALUENE_UINT64, netProfitVectors[i],
    //                                  UINT64_MAX, GrB_DESC_R);
    //         GxB_Vector_fprint(currentVec, "currentVecPre", GxB_COMPLETE,
    //                           stdout);
    //         GrB_Vector_reduce_UINT64(&hopMax, GrB_NULL,
    //         GrB_MAX_MONOID_UINT64,
    //                                  currentVec, GrB_NULL);
    //         GrB_Vector_select_UINT64(currentVec, GrB_NULL, GrB_NULL,
    //                                  GrB_VALUEEQ_UINT64, currentVec, hopMax,
    //                                  GrB_DESC_R);
    //
    //         GxB_Vector_fprint(currentVec, "currentVec", GxB_COMPLETE,
    //         stdout);
    //
    //         info = GxB_Vector_Iterator_attach(iterator, currentVec, NULL);
    //         info = GxB_Vector_Iterator_seek(iterator, 0);
    //         while (info != GxB_EXHAUSTED) {
    //             // get the entry A(i,j)
    //             GrB_Index i = GxB_Vector_Iterator_getIndex(iterator);
    //             uint64_t val = GxB_Iterator_get_UINT64(iterator);
    //             if (val == hopMax) {
    //                 fprintf(stderr, "from station at index %zu\n", i);
    //
    //                 fprintf(stderr, "station id: %zu\n",
    //                         marketInfo.stations[i].id);
    //
    //                 fprintf(stderr, "profit: %zu\n", hopMax);
    //             }
    //
    //             // move to the next entry in A
    //             info = GxB_Matrix_Iterator_next(iterator);
    //         }
    //     }
    // }

    GrB_finalize();
}
