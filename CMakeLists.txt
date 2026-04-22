cmake_minimum_required(VERSION 3.18)
project(mpi_mlp VERSION 1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CUDA_ARCHITECTURES "80")

# Build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE  "-O3 -march=native")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math")

# Allow passing extra CUDA flags (e.g. -lineinfo for profiling)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC")

find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(SOURCES
    src/main.cpp
    src/data/data_loader.cpp
    src/mlp/mlp.cu
)

add_executable(mlp_train ${SOURCES})

target_include_directories(mlp_train PRIVATE src ${MPI_CXX_INCLUDE_DIRS})

target_link_libraries(mlp_train PRIVATE
    MPI::MPI_CXX
    CUDA::cudart
    CUDA::cublas
)

set_target_properties(mlp_train PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# grad_check target added when src/grad_check.cpp exists (Phase 1 validation)
if(EXISTS "${CMAKE_SOURCE_DIR}/src/grad_check.cpp")
    add_executable(grad_check src/grad_check.cpp src/data/data_loader.cpp)
    target_include_directories(grad_check PRIVATE src)
    target_link_libraries(grad_check PRIVATE MPI::MPI_CXX)
endif()
