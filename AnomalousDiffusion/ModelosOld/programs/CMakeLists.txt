# programs/CMakeLists.txt
cmake_minimum_required(VERSION 3.25)

# ---------------------------------------------------------------------------
# Inherited configuration
# ---------------------------------------------------------------------------
if(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/exec")
endif()

if(NOT DEFINED PARALLEL_PROCESSES)
    set(PARALLEL_PROCESSES 2 CACHE STRING "MPI ranks used by 'make run'")
endif()

# ---------------------------------------------------------------------------
# Aggregate targets
# ---------------------------------------------------------------------------
add_custom_target(programs COMMENT "Building all programs")
add_custom_target(run_all COMMENT "Running all programs...")

# ---------------------------------------------------------------------------
# Discover program directories recursively
#
# Rule:
#   - a program directory is any directory under programs/ that contains
#     at least one .cpp file directly in that directory;
#   - the executable name is the name of that directory;
#   - all .cpp files under that directory (recursively) belong to that target.
# ---------------------------------------------------------------------------

# Collect every .cpp file under programs/
file(GLOB_RECURSE all_cpp_files
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

# Extract the directories that directly contain .cpp files
set(program_dirs "")
foreach(cpp_file IN LISTS all_cpp_files)
    get_filename_component(cpp_dir "${cpp_file}" DIRECTORY)
    list(APPEND program_dirs "${cpp_dir}")
endforeach()
list(REMOVE_DUPLICATES program_dirs)

# Keep track of target names to avoid duplicates
set(program_target_names "")

foreach(prog_dir_rel IN LISTS program_dirs)

    # Ignore the root itself if a .cpp is ever placed directly in programs/
    if(prog_dir_rel STREQUAL "")
        continue()
    endif()

    set(prog_dir_abs "${CMAKE_CURRENT_SOURCE_DIR}/${prog_dir_rel}")

    # Sources for this program: all .cpp files under the program directory
    file(GLOB_RECURSE prog_srcs
        CONFIGURE_DEPENDS
        "${prog_dir_abs}/*.cpp"
    )

    if(NOT prog_srcs)
        continue()
    endif()

    # Executable name = leaf directory name
    get_filename_component(prog_name "${prog_dir_abs}" NAME)

    # Guard against duplicated leaf names in different branches
    if(prog_name IN_LIST program_target_names)
        message(FATAL_ERROR
            "Duplicate program target name detected: '${prog_name}'. "
            "Please ensure each program directory has a unique leaf name."
        )
    endif()
    list(APPEND program_target_names "${prog_name}")

    # Create executable
    add_executable(${prog_name} ${prog_srcs})

    # Link with bgclib if present
    if(TARGET bgclib)
        target_link_libraries(${prog_name} PRIVATE bgclib)
    endif()

    # Link global dependencies
    target_link_libraries(${prog_name} PRIVATE
        MPI::MPI_CXX
        yaml-cpp::yaml-cpp
        PkgConfig::PETSC
    )

    # Aggregate build target
    add_dependencies(programs ${prog_name})

    # Run target
    # Working directory is the program folder itself, so ./Dados works naturally.
    add_custom_target(run_${prog_name}
        DEPENDS ${prog_name}
        COMMAND ${MPIEXEC_EXECUTABLE}
                ${MPIEXEC_NUMPROC_FLAG}
                $<IF:$<BOOL:$ENV{NP}>,$ENV{NP},${PARALLEL_PROCESSES}>
                $<TARGET_FILE:${prog_name}>
        WORKING_DIRECTORY ${prog_dir_abs}
        COMMENT "Running ${prog_name}"
    )
    add_dependencies(run_all run_${prog_name})

endforeach()

# ---------------------------------------------------------------------------
# Final message
# ---------------------------------------------------------------------------
message(STATUS
    "Programs detected recursively from directories that directly contain .cpp files"
)