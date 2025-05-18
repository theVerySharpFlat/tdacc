#include "bitrange.h"
#include "minmax_and_dary_heap.hpp"
#include "tdb.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <optional>
#include <queue>
#include <utility>

#define SQ(x) ((x) * (x))

struct GalaxyGraph {
    std::vector<System> A;    // values
    std::vector<uint32_t> IA; // accumulated non-sparse counts
    std::vector<uint32_t> JA; // columns of values in A
};

struct Solution {
    int64_t srcSystemIndex;
    int64_t srcStationIndex;

    int64_t dstSystemIndex; // might be redundant
    int64_t dstStationIndex;

    int64_t totalProfit;
};

int64_t computeMaxProfitWithStationIndices(int64_t srcStationIndex,
                                           int64_t dstStationIndex,
                                           const MarketInfo &marketInfo,
                                           int64_t space) {
    int64_t total = 0;
    int64_t profit = 0;

    // profitPer, nAvailible
    // thread_local std::priority_queue<std::pair<int64_t, int64_t>> pq;
    // std::array<std::pair<int64_t, int64_t>, 1024> pq;
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
                pq[profitsIndex++] =
                    std::make_pair(profitPer, srcPricing.supplyQuantity);
                // std::push_heap(pq.begin(), pq.begin() + profitsIndex);
                // pq.emplace(profitPer, srcPricing.supplyQuantity);
                // push_dary_heap<4>(pq.begin(), pq.begin() + profitsIndex);
                // profitsIndex++;
            }

            i++;
            j++;
        } else if (srcPricing.itemID < dstPricing.itemID) {
            i++;
        } else {
            j++;
        }
    }
    // make_dary_heap<4>(pq.begin(), pq.begin() + profitsIndex,
    // std::greater<>());

    // std::sort(pq.begin(), pq.begin() + profitsIndex);
    for (int i = 0; i < profitsIndex - 1; i++) {
        bool swapped = false;
        for (int j = 0; j < profitsIndex - i - 1; j++) {
            if (pq[j].first > pq[j + 1].first) {
                std::swap(pq[j], pq[j + 1]);
                swapped = true;
            }
            if (!swapped) {
                break;
            }
        }
    }
    // std::make_heap(profits.begin(), profits.begin() + profitsIndex);
    // std::priority_queue<std::pair<int64_t, int64_t>> pq(
    //     profits.begin(), profits.begin() + profitsIndex);

    for (int i = 0; i < profitsIndex && total < space; i++) {
        // std::pop_heap(profits.begin(), profits.begin() + profitsIndex - i);
        // auto [profitPer, nAvailable] =
        //     pq.top(); // profits[profitsIndex - i - 1];
        // pq.pop();
        auto [profitPer, nAvailable] = pq[profitsIndex - i - 1];
        // std::pop_heap(pq.begin(), pq.begin() + profitsIndex - i);

        // pop_dary_heap<4>(pq.begin(), pq.begin() + profitsIndex - i);

        nAvailable = std::min(nAvailable, space - total);

        total += nAvailable;
        profit += nAvailable * profitPer;
    }

    return profit;
};

int traverse(const MarketInfo &marketInfo, const GalaxyGraph &graph,
             std::optional<const std::vector<Solution> *> previousSolutions,
             std::vector<Solution> &solutions,
             std::vector<omp_lock_t> &solutionLocks, BitRange &visitedSet,
             int64_t startSystemIndex, int64_t jumps, int capacity) {
    // BitRange visitedSet(marketInfo.systems.size());

    struct Node {
        int64_t index;
        int64_t depth;
    };
    std::queue<Node> queue;
    queue.push({startSystemIndex, 0});
    visitedSet.set(startSystemIndex);

    while (!queue.empty()) {
        Node curr = queue.front();
        queue.pop();

        uint32_t startSize = graph.IA[curr.index];
        uint32_t endSize = graph.IA[curr.index + 1];
        uint32_t nAdjacent = endSize - startSize;

        // printf("nAdjacent: %d\n", nAdjacent);

        // Compute solutions here
        // solutions[curr.index] =
        // Solution solution =
        //     computeSolution(startSystemIndex, curr.index, marketInfo, 712);
        int64_t srcEndStationIndex =
            marketInfo.systems[startSystemIndex].stationStartIndex +
            marketInfo.systems[startSystemIndex].nStations;
        int64_t dstEndStationIndex =
            marketInfo.systems[curr.index].stationStartIndex +
            marketInfo.systems[curr.index].nStations;
        for (int64_t srcStationIndex =
                 marketInfo.systems[startSystemIndex].stationStartIndex;
             srcStationIndex < srcEndStationIndex; srcStationIndex++) {
            for (int64_t dstStationIndex =
                     marketInfo.systems[curr.index].stationStartIndex;
                 dstStationIndex < dstEndStationIndex; dstStationIndex++) {
                int64_t profit = computeMaxProfitWithStationIndices(
                    srcStationIndex, dstStationIndex, marketInfo, capacity);

                if (previousSolutions) {
                    profit += (*(previousSolutions.value()))[srcStationIndex]
                                  .totalProfit;
                }

                // begin critical section
                omp_set_lock(&solutionLocks[dstStationIndex]);
                if (profit > solutions[dstStationIndex].totalProfit) {
                    solutions[dstStationIndex] = Solution{
                        .srcSystemIndex = startSystemIndex,
                        .srcStationIndex = srcStationIndex,
                        .dstSystemIndex = curr.index,
                        .dstStationIndex = dstStationIndex,
                        .totalProfit = profit,
                    };
                }
                omp_unset_lock(&solutionLocks[dstStationIndex]);
                // end critical section
            }
        }

        // if (solutions[curr.index].totalProfit > 20000000)
        //     printf("profitttt: %lu\n", solutions[curr.index].totalProfit);

        if (curr.depth < jumps) {
            for (uint32_t i = 0; i < nAdjacent; i++) {
                const System &system = graph.A[startSize + i];

                if (!visitedSet.get(system.index)) {
                    queue.push({
                        .index = system.index,
                        .depth = curr.depth + 1,
                    });

                    visitedSet.set(system.index);
                }
            }
        }
    }

    return 0;
}

int traverseNOP(const MarketInfo &marketInfo, const GalaxyGraph &graph,
                std::optional<const std::vector<Solution> *> previousSolutions,
                std::vector<Solution> &solutions,
                std::vector<omp_lock_t> &solutionLocks, BitRange &visitedSet,
                int64_t startSystemIndex, int64_t jumps, int capacity) {
    // BitRange visitedSet(marketInfo.systems.size());

    struct Node {
        int64_t index;
        int64_t depth;
    };
    std::queue<Node> queue;
    queue.push({startSystemIndex, 0});
    visitedSet.set(startSystemIndex);

    while (!queue.empty()) {
        Node curr = queue.front();
        queue.pop();

        uint32_t startSize = graph.IA[curr.index];
        uint32_t endSize = graph.IA[curr.index + 1];
        uint32_t nAdjacent = endSize - startSize;

        // printf("nAdjacent: %d\n", nAdjacent);

        // Compute solutions here
        // solutions[curr.index] =
        // Solution solution =
        //     computeSolution(startSystemIndex, curr.index, marketInfo, 712);
        // int64_t srcEndStationIndex =
        //     marketInfo.systems[startSystemIndex].stationStartIndex +
        //     marketInfo.systems[startSystemIndex].nStations;
        // int64_t dstEndStationIndex =
        //     marketInfo.systems[curr.index].stationStartIndex +
        //     marketInfo.systems[curr.index].nStations;
        // for (int64_t srcStationIndex =
        //          marketInfo.systems[startSystemIndex].stationStartIndex;
        //      srcStationIndex < srcEndStationIndex; srcStationIndex++) {
        //     for (int64_t dstStationIndex =
        //              marketInfo.systems[curr.index].stationStartIndex;
        //          dstStationIndex < dstEndStationIndex; dstStationIndex++) {
        // int64_t profit = computeMaxProfitWithStationIndices(
        //     srcStationIndex, dstStationIndex, marketInfo, capacity);
        //
        // if (previousSolutions) {
        //     profit += (*(previousSolutions.value()))[srcStationIndex]
        //                   .totalProfit;
        // }
        //
        // // begin critical section
        // omp_set_lock(&solutionLocks[dstStationIndex]);
        // if (profit > solutions[dstStationIndex].totalProfit) {
        //     solutions[dstStationIndex] = Solution{
        //         .srcSystemIndex = startSystemIndex,
        //         .srcStationIndex = srcStationIndex,
        //         .dstSystemIndex = curr.index,
        //         .dstStationIndex = dstStationIndex,
        //         .totalProfit = profit,
        //     };
        // }
        // omp_unset_lock(&solutionLocks[dstStationIndex]);
        // // end critical section
        //     }
        // }
        //
        // if (solutions[curr.index].totalProfit > 20000000)
        //     printf("profitttt: %lu\n", solutions[curr.index].totalProfit);

        if (curr.depth < jumps) {
            for (uint32_t i = 0; i < nAdjacent; i++) {
                const System &system = graph.A[startSize + i];

                if (!visitedSet.get(system.index)) {
                    queue.push({
                        .index = system.index,
                        .depth = curr.depth + 1,
                    });

                    visitedSet.set(system.index);
                }
            }
        }
    }

    return 0;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "please enter arguments in the following order: "
                        "jump_range, n_jumps, n_hops, capacity\n");
        return 0;
    }
    double maxJump = std::stod(argv[1]);
    int nJumps = std::stoi(argv[2]);
    int nHops = std::stoi(argv[3]);
    int capacity = std::stoi(argv[4]);

    MPI_Init(&argc, &argv);
    int worldSize;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        std::cout << "jump range: " << maxJump << std::endl;
        std::cout << "jumps: " << nJumps << std::endl;
        std::cout << "hops: " << nHops << std::endl;
        std::cout << "capacity: " << capacity << std::endl;
    }

    // std::vector<System> systems = tdb.loadSystems();
    MarketInfo marketInfo;

    if (rank == 0) {
        TDB tdb("data/TradeDangerous.db");
        if (tdb.loadMarketInfo(&marketInfo)) {
            exit(EXIT_FAILURE);
        }
    }

    if (worldSize > 1) {
        int stationsLen = marketInfo.stations.size();
        int systemsLen = marketInfo.systems.size();
        int listingsLen = marketInfo.listings.size();

        MPI_Bcast(&stationsLen, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&systemsLen, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&listingsLen, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank != 0) {
            marketInfo.stations.resize(stationsLen);
            marketInfo.systems.resize(systemsLen);
            marketInfo.listings.resize(listingsLen);
        }

        MPI_Bcast(marketInfo.stations.data(),
                  marketInfo.stations.size() * sizeof(Station), MPI_BYTE, 0,
                  MPI_COMM_WORLD);
        MPI_Bcast(marketInfo.systems.data(),
                  marketInfo.systems.size() * sizeof(System), MPI_BYTE, 0,
                  MPI_COMM_WORLD);
        MPI_Bcast(marketInfo.listings.data(),
                  marketInfo.listings.size() * sizeof(ItemPricing), MPI_BYTE, 0,
                  MPI_COMM_WORLD);
    }

    if (rank == 0) {
        std::cout << "system count: " << marketInfo.systems.size() << std::endl;
        std::cout << "station count: " << marketInfo.stations.size()
                  << std::endl;
        std::cout << "listing count: " << marketInfo.listings.size()
                  << std::endl;
    }

    GalaxyGraph graph;
    if (rank == 0) {
        graph.IA.push_back(0);

        uint32_t nnz = 0;
        for (int i = 0; i < marketInfo.systems.size(); i++) {
            for (int j = 0; j < marketInfo.systems.size(); j++) {
                if (i == j)
                    continue;

                double sqDist =
                    SQ(marketInfo.systems[i].x - marketInfo.systems[j].x) +
                    SQ(marketInfo.systems[i].y - marketInfo.systems[j].y) +
                    SQ(marketInfo.systems[i].z - marketInfo.systems[j].z);

                if (sqDist <= SQ(maxJump)) {
                    graph.A.push_back(marketInfo.systems[j]);
                    graph.JA.push_back(j);
                    nnz++;
                }
            }
            graph.IA.push_back(nnz);
        }
    }

    if (worldSize > 1) {
        int iaLen = graph.IA.size();
        int jaLen = graph.JA.size();
        int aLen = graph.A.size();

        MPI_Bcast(&iaLen, 1, MPI_INT32_T, 0, MPI_COMM_WORLD);
        MPI_Bcast(&jaLen, 1, MPI_INT32_T, 0, MPI_COMM_WORLD);
        MPI_Bcast(&aLen, 1, MPI_INT32_T, 0, MPI_COMM_WORLD);

        if (rank != 0) {
            graph.IA.resize(iaLen);
            graph.JA.resize(jaLen);
            graph.A.resize(aLen);
        }

        MPI_Bcast(graph.IA.data(), graph.IA.size() * sizeof(uint32_t), MPI_BYTE,
                  0, MPI_COMM_WORLD);
        MPI_Bcast(graph.JA.data(), graph.JA.size() * sizeof(uint32_t), MPI_BYTE,
                  0, MPI_COMM_WORLD);
        MPI_Bcast(graph.A.data(), graph.A.size() * sizeof(System), MPI_BYTE, 0,
                  MPI_COMM_WORLD);
    }

    // std::vector<Item> itemsMap = tdb.loadItemTypes();
    // printf("number of items: %zu\n", itemsMap.size());

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

    std::vector<Solution> solutions(marketInfo.stations.size());
    std::vector<omp_lock_t> solutionLocks(solutions.size());
    for (omp_lock_t &lock : solutionLocks) {
        omp_init_lock(&lock);
    }

    // double start = omp_get_wtime();
    std::cout << "begin initial traversal..." << std::endl;
    BitRange visitedSet(marketInfo.systems.size());
    traverse(marketInfo, graph, std::nullopt, solutions, solutionLocks,
             visitedSet, solIndex, nJumps, capacity);
    // std::cout << "initial traversal finished..."
    //           << (omp_get_wtime() - start) * 1e3 << "ms" << std::endl;
    // exit(0);

    Solution optimalSolution = solutions[0];
    for (Solution solution : solutions) {
        if (solution.totalProfit > optimalSolution.totalProfit) {
            optimalSolution = solution;
        }
    }

    size_t originCount = 0;
    for (size_t i = 0; i < visitedSet.size; i++) {
        if (visitedSet.get(i)) {
            originCount++;
        }
    }

    // printf("optimal solution to system id %ld with profit %ld\n from ",
    //        marketInfo.systems[optimalSolution.dstSystemIndex].id,
    //        optimalSolution.totalProfit);

    std::vector<std::vector<Solution>> previousSolutions = {solutions};
    BitRange previousVisitedSet(visitedSet.size);

    for (size_t i = 0; i < visitedSet.size; i++) {
        if (visitedSet.get(i)) {
            previousVisitedSet.set(i);
        }
    }
    // std::vector<Solution> &previousSolutions = solutions;

    for (int hopNum = 1; hopNum < nHops; hopNum++) {
        int startIndex = 0;
        int endIndex = previousVisitedSet.size;

        // compute start and end indices if we're using MPI
        if (worldSize > 1) {
            int count = 0;
            int originsPerRank = originCount / worldSize;
            int currentRank = 0;
            for (int32_t i = 0; i < previousVisitedSet.size; i++) {
                if (previousVisitedSet.get(i)) {
                    count++;
                }

                if (count == (originsPerRank * rank)) {
                    startIndex = i;
                    endIndex = startIndex + originsPerRank;
                }

                if (count == (originsPerRank * (rank + 1))) {
                    endIndex = i;
                    if (rank == worldSize - 1) {
                        endIndex = previousVisitedSet.size;
                    }
                }
            }
        }

        std::vector<Solution> hopSolutions(marketInfo.stations.size());
        BitRange hopVisitedSet(marketInfo.systems.size());
        int32_t nIterations = 0;
#pragma omp parallel for schedule(dynamic)
        for (int32_t i = startIndex; i < endIndex; i++) {
            BitRange hopVisitedSetTraverse(marketInfo.systems.size());
            if (previousVisitedSet.get(i)) {
                traverse(marketInfo, graph,
                         &previousSolutions[previousSolutions.size() - 1],
                         hopSolutions, solutionLocks, hopVisitedSetTraverse, i,
                         nJumps, capacity);
                nIterations++;
                if (nIterations % 100 == 0) {
                    printf("%d/%zu\n", nIterations, originCount);
                }
            }

            for (int32_t j = 0; j < hopVisitedSetTraverse.size; j++) {
                if (hopVisitedSetTraverse.get(j))
                    hopVisitedSet.set(j);
            }
        }

        if (worldSize > 1) {
            if (rank == 0) {
                BitRange tempRange(visitedSet.size);
                std::vector<Solution> tempHopSolutions(
                    marketInfo.stations.size());
                for (int i = 1; i < worldSize; i++) {
                    MPI_Recv(tempRange.buf, tempRange.nmemb, MPI_UINT64_T, i, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv((void *)&tempHopSolutions[0],
                             tempHopSolutions.size() * sizeof(Solution),
                             MPI_BYTE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    // combine visited set
                    for (int32_t j = 0; j < tempRange.size; j++) {
                        if (tempRange.get(j))
                            hopVisitedSet.set(j);
                    }

                    // combine solutions (I.E max of the two)
                    for (int32_t j = 0; j < tempHopSolutions.size(); j++) {
                        if (tempHopSolutions[j].totalProfit >
                            hopSolutions[j].totalProfit) {
                            hopSolutions[j] = tempHopSolutions[j];
                        }
                    }
                }
            } else {
                MPI_Send(hopVisitedSet.buf, hopVisitedSet.nmemb, MPI_UINT64_T,
                         0, 0, MPI_COMM_WORLD);
                MPI_Send(hopSolutions.data(),
                         hopSolutions.size() * sizeof(Solution), MPI_BYTE, 0, 0,
                         MPI_COMM_WORLD);
            }

            // sync
            MPI_Bcast(hopVisitedSet.buf, hopVisitedSet.nmemb, MPI_UINT64_T, 0,
                      MPI_COMM_WORLD);
            MPI_Bcast(hopSolutions.data(),
                      hopSolutions.size() * sizeof(Solution), MPI_BYTE, 0,
                      MPI_COMM_WORLD);
        }

        originCount = 0;
        for (size_t i = 0; i < hopVisitedSet.size; i++) {
            if (hopVisitedSet.get(i)) {
                originCount++;
            }
        }

        for (size_t i = 0; i < hopVisitedSet.size; i++) {
            if (hopVisitedSet.get(i)) {
                previousVisitedSet.set(i);
            }
        }
        previousSolutions.push_back(hopSolutions);
    }

    // for (size_t i = previousSolutions.size() - 1; i >= 0; i--) {
    if (rank != 0) {
        return 0;
    }

    int i = previousSolutions.size() - 1;
    optimalSolution = previousSolutions[i][0];
    for (int64_t j = 0; j < previousSolutions[i].size(); j++) {
        Solution &solution = previousSolutions[i][j];
        if (solution.totalProfit > optimalSolution.totalProfit) {
            optimalSolution = solution;
        }
    }

    printf("hop %d:\n", nHops);
    printf("\toptimal solution to system id %ld and station id %ld from system "
           "id %ld and station id %ld with profit "
           "%ld\n",
           marketInfo.systems[optimalSolution.dstSystemIndex].id,
           marketInfo.stations[optimalSolution.dstStationIndex].id,
           marketInfo.systems[optimalSolution.srcSystemIndex].id,
           marketInfo.stations[optimalSolution.srcStationIndex].id,
           optimalSolution.totalProfit);

    for (int j = i - 1; j >= 0; j--) {
        optimalSolution = previousSolutions[j][optimalSolution.srcStationIndex];
        printf("hop %d:\n", j + 1);
        printf("\toptimal solution to system id %ld and station id %ld from "
               "system "
               "id %ld and station id %ld with profit "
               "%ld\n",
               marketInfo.systems[optimalSolution.dstSystemIndex].id,
               marketInfo.stations[optimalSolution.dstStationIndex].id,
               marketInfo.systems[optimalSolution.srcSystemIndex].id,
               marketInfo.stations[optimalSolution.srcStationIndex].id,
               optimalSolution.totalProfit);
    }
    // }

    return 0;
}
