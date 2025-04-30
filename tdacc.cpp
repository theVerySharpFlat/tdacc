#include "bitrange.h"
#include "tdb.h"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
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
    std::priority_queue<std::pair<int64_t, int64_t>> pq;
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
                pq.push(std::make_pair(profitPer, srcPricing.supplyQuantity));
            }

            i++;
            j++;
        } else if (srcPricing.itemID < dstPricing.itemID) {
            i++;
        } else {
            j++;
        }
    }

    while (!pq.empty() && total < space) {
        auto [profitPer, nAvailable] = pq.top();
        pq.pop();

        nAvailable = std::min(nAvailable, space - total);

        total += nAvailable;
        profit += nAvailable * profitPer;
    }

    return profit;
};

Solution computeSolution(int64_t startSystemIndex, int64_t endSystemIndex,
                         const MarketInfo &marketInfo, int64_t space) {
    Solution solution = {
        .srcSystemIndex = startSystemIndex,
        .dstSystemIndex = endSystemIndex,

        .totalProfit = -1,
    };

    int64_t srcEndStationIndex =
        marketInfo.systems[startSystemIndex].stationStartIndex +
        marketInfo.systems[startSystemIndex].nStations;
    int64_t dstEndStationIndex =
        marketInfo.systems[endSystemIndex].stationStartIndex +
        marketInfo.systems[endSystemIndex].nStations;

    // printf("dst nStations: %d\n",
    // marketInfo.systems[endSystemIndex].nStations); printf("src nStations:
    // %d\n",
    //        marketInfo.systems[startSystemIndex].nStations);
    for (int64_t srcStationIndex =
             marketInfo.systems[startSystemIndex].stationStartIndex;
         srcStationIndex < srcEndStationIndex; srcStationIndex++) {
        for (int64_t dstStationIndex =
                 marketInfo.systems[endSystemIndex].stationStartIndex;
             dstStationIndex < dstEndStationIndex; dstStationIndex++) {
            int64_t profit = computeMaxProfitWithStationIndices(
                srcStationIndex, dstStationIndex, marketInfo, space);

            if (profit > solution.totalProfit) {
                solution.srcStationIndex = srcStationIndex;
                solution.dstStationIndex = dstStationIndex;
                solution.totalProfit = profit;
            }
        }
    }

    return solution;
}

int traverse(const MarketInfo &marketInfo, const GalaxyGraph &graph,
             std::optional<const std::vector<Solution> *> previousSolutions,
             std::vector<Solution> &solutions, BitRange &visitedSet,
             int64_t startSystemIndex, int64_t jumps) {
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
        Solution solution =
            computeSolution(startSystemIndex, curr.index, marketInfo, 712);

        if (previousSolutions) {
            solution.totalProfit +=
                (*(previousSolutions.value()))[solution.srcSystemIndex]
                    .totalProfit;
        }
        // begin critical section
        if (solutions[curr.index].totalProfit < solution.totalProfit) {
            solutions[curr.index] = solution;
        }
        // end critical section
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

int main() {
    TDB tdb("data/TradeDangerous.db");
    // std::vector<System> systems = tdb.loadSystems();
    MarketInfo marketInfo;
    if (tdb.loadMarketInfo(&marketInfo)) {
        exit(EXIT_FAILURE);
    }

    std::cout << "system count: " << marketInfo.systems.size() << std::endl;
    std::cout << "station count: " << marketInfo.stations.size() << std::endl;
    std::cout << "listing count: " << marketInfo.listings.size() << std::endl;

    GalaxyGraph graph;
    graph.IA.push_back(0);

    double maxJump = 20.0;
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

    std::vector<Item> itemsMap = tdb.loadItemTypes();

    printf("number of items: %zu\n", itemsMap.size());

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

    std::vector<Solution> solutions(marketInfo.systems.size());
    BitRange visitedSet(marketInfo.systems.size());
    traverse(marketInfo, graph, std::nullopt, solutions, visitedSet, solIndex,
             10);

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

    printf("optimal solution to system id %ld with profit %ld\n from ",
           marketInfo.systems[optimalSolution.dstSystemIndex].id,
           optimalSolution.totalProfit);

    std::vector<Solution> hop2Solutions(marketInfo.systems.size());
    BitRange hop2VisitedSet(marketInfo.systems.size());
    int32_t nIterations = 0;
    for (int32_t i = 0; i < visitedSet.size; i++) {
        BitRange hop2VisitedSetTraverse(marketInfo.systems.size());
        if (visitedSet.get(i)) {
            traverse(marketInfo, graph, &solutions, hop2Solutions,
                     hop2VisitedSetTraverse, i, 10);
            nIterations++;
            printf("%d/%zu\n", nIterations, originCount);
        }

        for (int32_t j = 0; j < hop2VisitedSetTraverse.size; j++) {
            if (hop2VisitedSetTraverse.get(j))
                hop2VisitedSet.set(j);
        }
    }

    optimalSolution = hop2Solutions[0];
    for (int64_t i = 0; i < hop2Solutions.size(); i++) {
        Solution &solution = hop2Solutions[i];
        if (hop2VisitedSet.get(i)) {
            if (solution.totalProfit > optimalSolution.totalProfit) {
                optimalSolution = solution;
            }
        }
    }
    printf("optimal solution to system id %ld and station id %ld from system id %ld and station id %ld with profit "
           "%ld\n",
           marketInfo.systems[optimalSolution.dstSystemIndex].id,
           marketInfo.stations[optimalSolution.dstStationIndex].id,
           marketInfo.systems[optimalSolution.srcSystemIndex].id,
           marketInfo.stations[optimalSolution.srcStationIndex].id,
           optimalSolution.totalProfit);

    return 0;
}
