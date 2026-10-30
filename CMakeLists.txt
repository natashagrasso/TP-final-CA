cmake_minimum_required(VERSION 3.16)
project(cartoon_final LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(generate_test_image src/generate_test_image.cpp src/bmp.cpp)
add_executable(cartoon_sequential src/cartoon_sequential.cpp src/cartoon_common.cpp src/bmp.cpp)
add_executable(cartoon_openmp src/cartoon_openmp.cpp src/cartoon_common.cpp src/bmp.cpp)

find_package(MPI REQUIRED)
add_executable(cartoon_mpi src/cartoon_mpi.cpp src/cartoon_common.cpp src/bmp.cpp)
target_link_libraries(cartoon_mpi PRIVATE MPI::MPI_CXX)

add_executable(cartoon_hybrid src/cartoon_hybrid.cpp src/cartoon_common.cpp src/bmp.cpp)
target_link_libraries(cartoon_hybrid PRIVATE MPI::MPI_CXX)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cartoon_openmp PRIVATE -fopenmp)
    target_compile_options(cartoon_hybrid PRIVATE -fopenmp)
endif()
