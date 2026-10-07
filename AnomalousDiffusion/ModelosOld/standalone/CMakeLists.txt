# Standalone build of the legacy library and the cases used in Sections 3 and 8
# of BMSE-D-26-01905 (ModelosOld/programs/cases).
#
#   cmake -S AnomalousDiffusion/ModelosOld/standalone -B build-legacy -DCMAKE_BUILD_TYPE=Release
#   cmake --build build-legacy -j
#
# Executables are written to build-legacy/exec/.
cmake_minimum_required(VERSION 3.25)
project(BGCLegacy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/exec")

find_package(MPI REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(PETSC REQUIRED IMPORTED_TARGET PETSc)
find_package(yaml-cpp REQUIRED)

set(LEGACY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
add_subdirectory(${LEGACY_DIR} legacy_lib)
add_subdirectory(${LEGACY_DIR}/programs legacy_programs)
