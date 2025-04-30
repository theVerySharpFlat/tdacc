cmake_minimum_required(VERSION 3.16)

project(tdacc)

find_package(SQLite3 REQUIRED)

add_executable(tdacc tdacc.cpp tdb.cpp)
target_link_libraries(tdacc PUBLIC SQLite::SQLite3)
