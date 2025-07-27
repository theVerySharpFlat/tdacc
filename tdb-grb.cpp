#include "tdb-grb.h"
#include "GraphBLAS.h"
#include <cstdio>
#include <sqlite3.h>
#include <vector>

#define SYSTEM_VECTOR_BLOCK_SIZE 8192
#define STATION_VECTOR_BLOCK_SIZE SYSTEM_VECTOR_BLOCK_SIZE * 16
#define LISTING_VECTOR_BLOCK_SIZE STATION_VECTOR_BLOCK_SIZE * 64

TDB::TDB(const char *path) {
    int err = sqlite3_open_v2(
        path, &conn, SQLITE_OPEN_READONLY | SQLITE_OPEN_NOMUTEX, nullptr);

    if (err != SQLITE_OK) {
        fprintf(stderr, "failed to open %s: %s\n", path, sqlite3_errmsg(conn));
    }
}

std::vector<System> TDB::loadSystems() {
    const char *sqlStmt = "SELECT system_id,pos_x,pos_y,pos_z FROM System;";
    sqlite3_stmt *stmt;
    int err = sqlite3_prepare_v2(conn, sqlStmt, -1, &stmt, nullptr);

    if (err != SQLITE_OK) {
        fprintf(stderr, "failed to compile loadSystems sql: %s\n",
                sqlite3_errmsg(conn));
        return std::vector<System>();
    }

    std::vector<System> systems;
    systems.reserve(SYSTEM_VECTOR_BLOCK_SIZE);

    while (true) {
        int ret = sqlite3_step(stmt);
        if (ret == SQLITE_DONE) {
            break;
        } else if (ret != SQLITE_ROW) {
            fprintf(stderr, "sqlite step error: %s\n", sqlite3_errmsg(conn));
            return std::vector<System>();
        }

        System system;
        system.id = sqlite3_column_int(stmt, 1);
        system.x = sqlite3_column_double(stmt, 2);
        system.y = sqlite3_column_double(stmt, 3);
        system.z = sqlite3_column_double(stmt, 4);

        systems.push_back(system);

        if (systems.capacity() <= systems.size()) {
            systems.reserve(systems.size() + SYSTEM_VECTOR_BLOCK_SIZE);
        }
    }

    err = sqlite3_finalize(stmt);

    if (err != SQLITE_OK) {
        fprintf(stderr, "failed to finalize loadsystems sql: %s\n",
                sqlite3_errmsg(conn));
    }

    return systems;
}

int TDB::loadMarketInfo(MarketInfo *info) {
    info->systems.clear();
    info->stations.clear();
    info->listings.clear();

    const char *systemLoadSQL =
        "SELECT system_id,pos_x,pos_y,pos_z FROM System;";
    sqlite3_stmt *stmt;
    int err = sqlite3_prepare_v2(conn, systemLoadSQL, -1, &stmt, nullptr);

    if (err != SQLITE_OK) {
        fprintf(stderr, "failed to compile loadSystems sql: %s\n",
                sqlite3_errmsg(conn));
        return 1;
    }

    info->systems.reserve(SYSTEM_VECTOR_BLOCK_SIZE);
    info->stations.reserve(STATION_VECTOR_BLOCK_SIZE);
    info->listings.reserve(LISTING_VECTOR_BLOCK_SIZE);

    while (true) {
        int ret = sqlite3_step(stmt);
        if (ret == SQLITE_DONE) {
            break;
        } else if (ret != SQLITE_ROW) {
            fprintf(stderr, "sqlite step error: %s\n", sqlite3_errmsg(conn));
            return 1;
        }

        System system;
        system.id = sqlite3_column_int64(stmt, 0);
        system.index = info->systems.size();
        system.x = sqlite3_column_double(stmt, 1);
        system.y = sqlite3_column_double(stmt, 2);
        system.z = sqlite3_column_double(stmt, 3);

        info->systems.push_back(system);

        if (info->systems.capacity() <= info->systems.size()) {
            info->systems.reserve(info->systems.size() +
                                  SYSTEM_VECTOR_BLOCK_SIZE);
        }
    }

    err = sqlite3_finalize(stmt);

    if (err != SQLITE_OK) {
        fprintf(stderr, "failed to finalize loadsystems sql: %s\n",
                sqlite3_errmsg(conn));
        return 1;
    }

    const char *stationloadSQL =
        "SELECT station_id FROM station WHERE type_id != 24 AND type_id != 0 "
        "AND system_id = ?;";
    err = sqlite3_prepare_v2(conn, stationloadSQL, -1, &stmt, nullptr);

    if (err != SQLITE_OK) {
        fprintf(stderr, "failed to compile stationLoad sql: %s\n",
                sqlite3_errmsg(conn));
        return 1;
    }
    for (auto &system : info->systems) {
        sqlite3_reset(stmt);
        sqlite3_bind_int64(stmt, 1, system.id);
        // printf("systemID: %ld\n", system.id);

        system.stationStartIndex = info->stations.size();
        while (true) {
            int ret = sqlite3_step(stmt);
            if (ret == SQLITE_DONE) {
                break;
            } else if (ret != SQLITE_ROW) {
                fprintf(stderr, "stationload sqlite step error: %s\n",
                        sqlite3_errmsg(conn));
                return 1;
            }

            Station station;
            station.systemID = system.id;
            station.id = sqlite3_column_int64(stmt, 0);

            info->stations.push_back(station);

            if (info->stations.capacity() <= info->stations.size()) {
                info->stations.reserve(info->stations.size() +
                                       STATION_VECTOR_BLOCK_SIZE);
            }
        }
        system.nStations = info->stations.size() - system.stationStartIndex;
    }

    sqlite3_finalize(stmt);

    const char *listingLoadSQL =
        "SELECT item_id,demand_price,demand_units,supply_price,supply_units "
        "FROM "
        "StationItem WHERE station_id=? ORDER BY item_id ASC;";
    err = sqlite3_prepare_v2(conn, listingLoadSQL, -1, &stmt, nullptr);

    if (err != SQLITE_OK) {
        fprintf(stderr, "failed to compile listingLoadSQL sql: %s\n",
                sqlite3_errmsg(conn));
        return 1;
    }

    for (auto &station : info->stations) {
        sqlite3_reset(stmt);
        sqlite3_bind_int64(stmt, 1, station.id);

        station.listingStartIndex = info->listings.size();
        while (true) {
            int ret = sqlite3_step(stmt);
            if (ret == SQLITE_DONE) {
                break;
            } else if (ret != SQLITE_ROW) {
                fprintf(stderr, "listingload sqlite step error: %s\n",
                        sqlite3_errmsg(conn));
                return 1;
            }

            ItemPricing listing;
            listing.itemID = sqlite3_column_int64(stmt, 0);
            listing.demandPrice = sqlite3_column_int64(stmt, 1);
            listing.demandQuantity = sqlite3_column_int64(stmt, 2);
            listing.supplyPrice = sqlite3_column_int64(stmt, 3);
            listing.supplyQuantity = sqlite3_column_int64(stmt, 4);

            info->listings.push_back(listing);

            if (info->listings.capacity() <= info->listings.size()) {
                info->listings.reserve(info->listings.size() +
                                       LISTING_VECTOR_BLOCK_SIZE);
            }
        }
        station.nListings = info->listings.size() - station.listingStartIndex;
    }
    sqlite3_finalize(stmt);

    return 0;
}

GrB_Vector TDB::loadItemTypes() {
    std::vector<GrB_Index> items;
    std::vector<int32_t> indices;

    const char *querySQL = "select item_id from Item;";
    sqlite3_stmt *stmt;
    if (sqlite3_prepare_v2(conn, querySQL, -1, &stmt, NULL) != SQLITE_OK) {
        fprintf(stderr, "failed to compile loadItemTypes() query: %s\n",
                sqlite3_errmsg(conn));
        return GrB_NULL;
    }

    GrB_Vector vec;

    int err;
    int index = 0;
    do {
        err = sqlite3_step(stmt);
        if (err == SQLITE_ROW) {
            items.push_back(sqlite3_column_int64(stmt, 1));
            indices.push_back(index++);
        }
    } while (err == SQLITE_ROW);

    if (err != SQLITE_DONE) {
        fprintf(stderr, "TDB::loadItemTypes sqlite3 error: %s\n",
                sqlite3_errmsg(conn));
        sqlite3_finalize(stmt);
        return GrB_NULL;
    }

    GrB_Vector_new(&vec, GrB_INT32, items.back());
    GrB_Vector_build_INT32(vec, items.data(), indices.data(), index, GrB_NULL);

    sqlite3_finalize(stmt);
    return vec;
}
