cmake_minimum_required(VERSION 3.25)

project(AnomalousDiffusion
    VERSION 2.0
    DESCRIPTION "One-dimensional anomalous diffusion models"
    LANGUAGES CXX
)

# -----------------------------------------------------------------------------
# Project options
# -----------------------------------------------------------------------------

option(ANOMALOUS_BUILD_PROGRAMS
    "Build executable programs, tests, cases, and paper drivers"
    ON
)

option(ANOMALOUS_BUILD_TESTS
    "Build GTest-based unit tests"
    OFF
)

option(ANOMALOUS_ENABLE_WARNINGS
    "Enable project warning flags"
    ON
)

option(ANOMALOUS_ENABLE_LTO
    "Enable interprocedural/link-time optimization in optimized builds"
    ON
)

option(ANOMALOUS_ENABLE_NATIVE_ARCH
    "Optimize for the native CPU architecture"
    ON
)

option(ANOMALOUS_ENABLE_FAST_MATH
    "Enable aggressive floating-point optimizations; may change numerical results"
    OFF
)

set(ANOMALOUS_DEFAULT_MPI_RANKS 12 CACHE STRING
    "Default MPI ranks used by run targets"
)

# -----------------------------------------------------------------------------
# C++ standard and build type
# -----------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        Debug
        Release
        RelWithDebInfo
        MinSizeRel
    )
endif()

# -----------------------------------------------------------------------------
# Output directories
# -----------------------------------------------------------------------------

include(GNUInstallDirs)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY
    "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}"
)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY
    "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}"
)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
    "${CMAKE_BINARY_DIR}/bin"
)

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------

find_package(MPI REQUIRED)
find_package(PkgConfig REQUIRED)

if(DEFINED ENV{PETSC_DIR})
    if(DEFINED ENV{PETSC_ARCH})
        set(ENV{PKG_CONFIG_PATH}
            "$ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}"
        )
    endif()

    set(ENV{PKG_CONFIG_PATH}
        "$ENV{PETSC_DIR}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}"
    )
endif()

pkg_check_modules(PETSC REQUIRED IMPORTED_TARGET PETSc)

if(ANOMALOUS_BUILD_PROGRAMS)
    find_package(yaml-cpp REQUIRED)
endif()

if(ANOMALOUS_BUILD_TESTS)
    enable_testing()
    find_package(GTest REQUIRED)
endif()

message(STATUS "MPI CXX compiler   : ${MPI_CXX_COMPILER}")
message(STATUS "MPIEXEC executable : ${MPIEXEC_EXECUTABLE}")
message(STATUS "PETSc version      : ${PETSC_VERSION}")
message(STATUS "Build type         : ${CMAKE_BUILD_TYPE}")

# -----------------------------------------------------------------------------
# Shared compile and link settings
# -----------------------------------------------------------------------------

add_library(anomalous_project_options INTERFACE)
target_compile_features(anomalous_project_options INTERFACE cxx_std_23)

target_compile_options(anomalous_project_options INTERFACE
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Release>:-DNDEBUG>
    $<$<CONFIG:RelWithDebInfo>:-O3>
)

if(ANOMALOUS_ENABLE_NATIVE_ARCH)
    target_compile_options(anomalous_project_options INTERFACE
        $<$<CONFIG:Release>:-march=native>
        $<$<CONFIG:Release>:-mtune=native>
        $<$<CONFIG:RelWithDebInfo>:-march=native>
        $<$<CONFIG:RelWithDebInfo>:-mtune=native>
    )
endif()

if(ANOMALOUS_ENABLE_FAST_MATH)
    target_compile_options(anomalous_project_options INTERFACE
        $<$<CONFIG:Release>:-ffast-math>
        $<$<CONFIG:Release>:-fno-math-errno>
        $<$<CONFIG:RelWithDebInfo>:-ffast-math>
        $<$<CONFIG:RelWithDebInfo>:-fno-math-errno>
    )
endif()

add_library(anomalous_project_warnings INTERFACE)

if(ANOMALOUS_ENABLE_WARNINGS)
    target_compile_options(anomalous_project_warnings INTERFACE
        -Wall
        -Wextra
        -Wpedantic
        -Wshadow
        -Wconversion
        -Wsign-conversion
    )
endif()

if(ANOMALOUS_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)

    if(ipo_supported)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ON)
    else()
        message(WARNING "IPO/LTO not supported: ${ipo_error}")
    endif()
endif()

# -----------------------------------------------------------------------------
# Library
# -----------------------------------------------------------------------------

add_subdirectory(bgclib)

# -----------------------------------------------------------------------------
# Programs
# -----------------------------------------------------------------------------

if(ANOMALOUS_BUILD_PROGRAMS)
    add_subdirectory(programs)
endif()

if(ANOMALOUS_BUILD_PROGRAMS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/paper/CMakeLists.txt")
    add_subdirectory(paper)
endif()

if(ANOMALOUS_BUILD_PROGRAMS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/TSOM/CMakeLists.txt")
    add_subdirectory(tests/TSOM)
endif()

if(ANOMALOUS_BUILD_TESTS)
    add_subdirectory(tests)
endif()

# -----------------------------------------------------------------------------
# Generic run target
# -----------------------------------------------------------------------------

if(ANOMALOUS_BUILD_PROGRAMS)
    add_custom_target(run
        COMMAND ${MPIEXEC_EXECUTABLE}
                ${MPIEXEC_NUMPROC_FLAG}
                $<IF:$<BOOL:$ENV{NP}>,$ENV{NP},${ANOMALOUS_DEFAULT_MPI_RANKS}>
                $<IF:$<BOOL:$ENV{EXEC}>,
                    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$ENV{EXEC},
                    NOT_FOUND
                >
        DEPENDS programs
        USES_TERMINAL
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Run a program with: cmake --build build --target run -- EXEC=name NP=ranks"
    )
endif()

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

message(STATUS "Configuration complete")
message(STATUS "Runtime output directory: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "Build programs          : ${ANOMALOUS_BUILD_PROGRAMS}")
message(STATUS "Default MPI ranks       : ${ANOMALOUS_DEFAULT_MPI_RANKS}")
message(STATUS "Native architecture     : ${ANOMALOUS_ENABLE_NATIVE_ARCH}")
message(STATUS "LTO/IPO                 : ${ANOMALOUS_ENABLE_LTO}")
message(STATUS "Fast math               : ${ANOMALOUS_ENABLE_FAST_MATH}")
