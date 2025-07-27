#include "GraphBLAS.h"
#include "tdb-grb.h"
#include <cassert>
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
                pq[j + 1] = pq[j];
                swapped = true;
            }
            if (!swapped) {
                break;
            }
        }

        auto [profitPer, nAvailable] = pq[profitsIndex - i - 1];

        nAvailable = min(nAvailable, space - total);

        total += nAvailable;
        profit += nAvailable * profitPer;
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
                       uint64_t *gpuSolutionDsts) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    traverse(gpuData, problems[i].startSystem, problems[i].endSystem,
             problems[i].solutionOffset, 712, gpuSolutionVals, gpuSolutionSrcs,
             gpuSolutionDsts);
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

    for (int i = 0; i < nHops; i++) {
        GrB_Info info = GrB_mxm(b, GrB_NULL, GrB_NULL,
                                GrB_LOR_LAND_SEMIRING_BOOL, a, c, GrB_NULL);
        fprintf(stderr, "info: %d\n", info);

        GrB_Matrix_eWiseAdd_BinaryOp(galaxyGraph, GrB_NULL, GrB_NULL, GrB_LOR,
                                     galaxyGraph, b, GrB_NULL);

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
    GxB_Iterator_free(&iterator);

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

    kernel<<<hostProblems.size() / 128, 128>>>(
        problems, gpuData, gpuSolutionVals, gpuSolutionSrcs, gpuSolutionDsts);
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
    fprintf(stderr, "extractTuples error: %d\n", grbErr);

    GxB_Matrix_fprint(solutionMat, "solutionmat", GxB_SUMMARY, stdout);
    // *nvals, const GrB_Matrix A)

    GrB_finalize();
}
