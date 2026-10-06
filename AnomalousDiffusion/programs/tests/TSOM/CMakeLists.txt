# tests/TSOM/CMakeLists.txt
#
# Standalone TSOM MMS-style drivers. These are intentionally not GTest tests:
# each leaf directory with .cpp files is built as an independent executable and
# writes its own CSV output under that directory.

file(GLOB_RECURSE tsom_test_cpp_files
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

set(tsom_test_dirs "")
foreach(cpp_file IN LISTS tsom_test_cpp_files)
    get_filename_component(cpp_dir "${cpp_file}" DIRECTORY)
    if(NOT cpp_dir STREQUAL "")
        list(APPEND tsom_test_dirs "${cpp_dir}")
    endif()
endforeach()

list(REMOVE_DUPLICATES tsom_test_dirs)
list(SORT tsom_test_dirs)

add_custom_target(tsom_mms_tests COMMENT "Building standalone TSOM MMS test drivers")
add_custom_target(run_tsom_mms_tests COMMENT "Running standalone TSOM MMS test drivers")

set(tsom_test_target_names "")
foreach(test_dir_rel IN LISTS tsom_test_dirs)
    set(test_dir_abs "${CMAKE_CURRENT_SOURCE_DIR}/${test_dir_rel}")

    file(GLOB_RECURSE test_srcs
        CONFIGURE_DEPENDS
        "${test_dir_abs}/*.cpp"
    )

    if(NOT test_srcs)
        continue()
    endif()

    get_filename_component(test_name "${test_dir_abs}" NAME)
    set(target_name "TSOM_${test_name}")

    if(target_name IN_LIST tsom_test_target_names)
        message(FATAL_ERROR "Duplicate TSOM test target detected: ${target_name}")
    endif()
    list(APPEND tsom_test_target_names "${target_name}")

    add_executable(${target_name} ${test_srcs})
    target_link_libraries(${target_name}
        PRIVATE
            bgclib
            MPI::MPI_CXX
            PkgConfig::PETSC
    )

    if(TARGET anomalous_project_options)
        target_link_libraries(${target_name} PRIVATE anomalous_project_options)
    endif()
    if(TARGET anomalous_project_warnings)
        target_link_libraries(${target_name} PRIVATE anomalous_project_warnings)
    endif()

    add_dependencies(tsom_mms_tests ${target_name})

    add_custom_target(run_${target_name}
        DEPENDS ${target_name}
        COMMAND ${MPIEXEC_EXECUTABLE}
                ${MPIEXEC_NUMPROC_FLAG}
                $<IF:$<BOOL:$ENV{NP}>,$ENV{NP},1>
                $<TARGET_FILE:${target_name}>
        WORKING_DIRECTORY "${test_dir_abs}"
        USES_TERMINAL
        COMMENT "Running ${target_name}"
    )
    add_dependencies(run_tsom_mms_tests run_${target_name})
endforeach()

message(STATUS "Standalone TSOM MMS test targets: ${tsom_test_target_names}")
