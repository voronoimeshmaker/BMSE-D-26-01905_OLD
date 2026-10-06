# -----------------------------------------------------------------------------
# bgclib/CMakeLists.txt
#
# Static library for one-dimensional anomalous diffusion models.
# This file is included from the project root and must not call project().
#
# Design rule:
#   bgclib is the numerical library. It should depend on PETSc and MPI.
# -----------------------------------------------------------------------------

file(GLOB_RECURSE BGCLIB_SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

if(NOT BGCLIB_SOURCES)
    message(FATAL_ERROR
        "No bgclib source files found in ${CMAKE_CURRENT_SOURCE_DIR}/src"
    )
endif()

add_library(bgclib STATIC ${BGCLIB_SOURCES})
add_library(Anomalous::bgclib ALIAS bgclib)

target_include_directories(bgclib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(bgclib PUBLIC cxx_std_23)

target_link_libraries(bgclib
    PUBLIC
        anomalous_project_options
        MPI::MPI_CXX
        PkgConfig::PETSC
    PRIVATE
        anomalous_project_warnings
)

source_group(
    TREE "${CMAKE_CURRENT_SOURCE_DIR}/src"
    PREFIX "Source Files"
    FILES ${BGCLIB_SOURCES}
)
