# programs/CMakeLists.txt
cmake_minimum_required(VERSION 3.25)

# -----------------------------------------------------------------------------
# Inherited configuration
# -----------------------------------------------------------------------------

# Directory in which program executables are written.
# Preserve a value defined by the parent project, when available.
if(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    set(
        CMAKE_RUNTIME_OUTPUT_DIRECTORY
        "${CMAKE_SOURCE_DIR}/exec"
    )
endif()

# Default number of MPI ranks used by the run_<program> targets.
# Preserve a value defined by the parent project.
if(NOT DEFINED PARALLEL_PROCESSES)
    set(
        PARALLEL_PROCESSES
        2
        CACHE STRING
        "MPI ranks used by run targets"
    )
endif()

# Optional environment override.
#
# NP is evaluated when CMake is configured, not when Ninja executes.
# This avoids embedding $ENV{...} expressions in build.ninja.
#
# Examples:
#
#   NP=1 cmake -S . -B build
#
# or:
#
#   cmake -S . -B build -DPARALLEL_PROCESSES=1
#
if(DEFINED ENV{NP} AND NOT "$ENV{NP}" STREQUAL "")
    set(PROGRAM_MPI_PROCESSES "$ENV{NP}")
else()
    set(PROGRAM_MPI_PROCESSES "${PARALLEL_PROCESSES}")
endif()

message(
    STATUS
    "MPI ranks used by program run targets: ${PROGRAM_MPI_PROCESSES}"
)

# -----------------------------------------------------------------------------
# Aggregate targets
# -----------------------------------------------------------------------------

# Build every program detected below.
add_custom_target(
    programs ALL
    COMMENT "Building all programs"
)

# Execute every program through its corresponding run_<program> target.
# This target is not part of ALL and therefore runs only when explicitly
# requested.
add_custom_target(
    run_all
    COMMENT "Running all programs"
)

# -----------------------------------------------------------------------------
# Discover program directories
#
# A program directory is any directory below programs/ containing at least one
# .cpp file DIRECTLY in that directory.
#
# Each such directory defines one executable.
#
# IMPORTANT:
#
# Source files in nested directories do NOT belong to the executable of their
# parent directory. If a nested directory contains its own .cpp file, it is
# treated as an independent program.
#
# Example:
#
#   programs/
#       newpaper/
#           NewPaperPsiV.cpp
#           pulse_runner/
#               pulse_runner.cpp
#
# produces:
#
#   executable: newpaper
#       source: newpaper/NewPaperPsiV.cpp
#
#   executable: pulse_runner
#       source: newpaper/pulse_runner/pulse_runner.cpp
#
# This prevents independent main() functions from being linked into the same
# executable.
# -----------------------------------------------------------------------------

file(
    GLOB_RECURSE all_cpp_files
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

if(NOT all_cpp_files)
    message(
        WARNING
        "No program .cpp files found under ${CMAKE_CURRENT_SOURCE_DIR}"
    )
    return()
endif()

# -----------------------------------------------------------------------------
# Determine directories containing program source files
# -----------------------------------------------------------------------------

set(program_dirs "")

foreach(cpp_file IN LISTS all_cpp_files)

    get_filename_component(
        cpp_dir
        "${cpp_file}"
        DIRECTORY
    )

    list(
        APPEND
        program_dirs
        "${cpp_dir}"
    )

endforeach()

list(REMOVE_DUPLICATES program_dirs)
list(SORT program_dirs)

# -----------------------------------------------------------------------------
# Create program targets
# -----------------------------------------------------------------------------

set(program_target_names "")

foreach(prog_dir_rel IN LISTS program_dirs)

    # Ignore a .cpp located directly in programs/ itself.
    # Program executables are expected to live in their own directories.
    if(prog_dir_rel STREQUAL "")
        continue()
    endif()

    set(
        prog_dir_abs
        "${CMAKE_CURRENT_SOURCE_DIR}/${prog_dir_rel}"
    )

    # -------------------------------------------------------------------------
    # Collect only the .cpp files directly inside this program directory.
    #
    # DO NOT use GLOB_RECURSE here.
    #
    # Nested directories are independent programs and will be processed
    # separately by this loop.
    # -------------------------------------------------------------------------

    file(
        GLOB prog_srcs
        CONFIGURE_DEPENDS
        "${prog_dir_abs}/*.cpp"
    )

    if(NOT prog_srcs)
        continue()
    endif()

    # Executable name = leaf directory name.
    get_filename_component(
        prog_name
        "${prog_dir_abs}"
        NAME
    )

    # -------------------------------------------------------------------------
    # Prevent target-name collisions.
    #
    # For example:
    #
    #   cases/A/test/test.cpp
    #   cases/B/test/test.cpp
    #
    # would otherwise both try to create an executable named "test".
    # -------------------------------------------------------------------------

    if(prog_name IN_LIST program_target_names)
        message(
            FATAL_ERROR
            "Duplicate program target name detected: '${prog_name}'. "
            "Each program directory must have a unique leaf-directory name. "
            "Conflicting directory: ${prog_dir_rel}"
        )
    endif()

    list(
        APPEND
        program_target_names
        "${prog_name}"
    )

    # -------------------------------------------------------------------------
    # Executable
    # -------------------------------------------------------------------------

    add_executable(
        ${prog_name}
        ${prog_srcs}
    )

    # -------------------------------------------------------------------------
    # Project library
    #
    # Prefer the namespaced target when available, while retaining compatibility
    # with projects exposing bgclib directly.
    # -------------------------------------------------------------------------

    if(TARGET Anomalous::bgclib)

        target_link_libraries(
            ${prog_name}
            PRIVATE
                Anomalous::bgclib
        )

    elseif(TARGET bgclib)

        target_link_libraries(
            ${prog_name}
            PRIVATE
                bgclib
        )

    endif()

    # -------------------------------------------------------------------------
    # External dependencies
    # -------------------------------------------------------------------------

    target_link_libraries(
        ${prog_name}
        PRIVATE
            MPI::MPI_CXX
            yaml-cpp::yaml-cpp
            PkgConfig::PETSC
    )

    # -------------------------------------------------------------------------
    # Common project options
    # -------------------------------------------------------------------------

    if(TARGET anomalous_project_options)

        target_link_libraries(
            ${prog_name}
            PRIVATE
                anomalous_project_options
        )

    endif()

    if(TARGET anomalous_project_warnings)

        target_link_libraries(
            ${prog_name}
            PRIVATE
                anomalous_project_warnings
        )

    endif()

    # -------------------------------------------------------------------------
    # Aggregate build target
    # -------------------------------------------------------------------------

    add_dependencies(
        programs
        ${prog_name}
    )

    # -------------------------------------------------------------------------
    # Individual run target
    #
    # Example:
    #
    #   cmake --build build --target run_pulse_runner
    #
    # The executable runs from its own program directory so that relative input
    # and output paths retain their expected behaviour.
    # -------------------------------------------------------------------------

    add_custom_target(
        run_${prog_name}

        DEPENDS
            ${prog_name}

        COMMAND
            ${MPIEXEC_EXECUTABLE}
            ${MPIEXEC_NUMPROC_FLAG}
            ${PROGRAM_MPI_PROCESSES}
            $<TARGET_FILE:${prog_name}>

        WORKING_DIRECTORY
            "${prog_dir_abs}"

        USES_TERMINAL

        COMMENT
            "Running ${prog_name} with ${PROGRAM_MPI_PROCESSES} MPI rank(s)"
    )

    # -------------------------------------------------------------------------
    # Aggregate run target
    # -------------------------------------------------------------------------

    add_dependencies(
        run_all
        run_${prog_name}
    )

endforeach()

# -----------------------------------------------------------------------------
# Configuration summary
# -----------------------------------------------------------------------------

message(
    STATUS
    "Program targets detected: ${program_target_names}"
)

message(
    STATUS
    "Program runtime output directory: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
)

message(
    STATUS
    "Program run-target MPI ranks: ${PROGRAM_MPI_PROCESSES}"
)