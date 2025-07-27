#pragma once

#include "GraphBLAS.h"
#include <cstdint>
#include <sqlite3.h>
#include <vector>

//
// Commodoty pricing layout idea:
//
// System:
// int64 systemid
// int64 stationcount
//
//     Station[stationCount]:
//     int64 commodityCount
//
//         Commodity[commodityCount]:
//         int32 demandPrice
//         int32 demandCount
//         int32 supplyPrice
//         int32 supplyCount
//
// Per-System solution idea
//
// Solution:
// int64 systemID
// int64 stationID
// int64 fromSystemID
// int64 fromStationID
//
// int32 totalProfit

struct ItemPricing {
    int64_t itemID;
    int32_t demandPrice;
    int32_t demandQuantity;
    int32_t supplyPrice;
    int32_t supplyQuantity;
};

struct Station {
    int64_t id;
    int64_t systemID;
    // int32_t lsFromStar;

    int32_t nListings;
    int32_t listingStartIndex;
};

struct System {
    int64_t id;
    int64_t index;
    double x, y, z;

    int32_t nStations;
    int32_t stationStartIndex;
};

struct Item {
    int64_t itemID;
};

struct MarketInfo {
    std::vector<System> systems;
    std::vector<Station> stations;
    std::vector<ItemPricing> listings;
};

class TDB {
  public:
    TDB(const char *path);

    std::vector<System> loadSystems();

    int loadMarketInfo(MarketInfo *info);

    /*
     * Item IDs in the SQL database are non-contiguous, so we produce a map here
     */
    GrB_Vector loadItemTypes();

  private:
    sqlite3 *conn;
};
