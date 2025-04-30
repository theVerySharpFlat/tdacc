cmake_minimum_required(VERSION 3.16)

project(tdacc)

find_package(SQLite3 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

add_executable(tdacc tdacc.cpp tdb.cpp)
target_link_libraries(tdacc PUBLIC SQLite::SQLite3 OpenMP::OpenMP_CXX MPI::MPI_CXX)
