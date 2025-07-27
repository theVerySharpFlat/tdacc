cmake_minimum_required(VERSION 3.16)

project(tdacc LANGUAGES CXX CUDA)

find_package(SQLite3 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
# find_package(GraphBLAS REQUIRED)

add_subdirectory(graphblas)

add_executable(tdacc tdacc.cpp tdb.cpp)
target_link_libraries(tdacc PUBLIC SQLite::SQLite3 OpenMP::OpenMP_CXX MPI::MPI_CXX)

add_executable(tdacc-grb tdacc-grb.cu tdb-grb.cpp)
target_link_libraries(tdacc-grb PUBLIC SQLite::SQLite3 GraphBLAS OpenMP::OpenMP_CXX MPI::MPI_CXX)
