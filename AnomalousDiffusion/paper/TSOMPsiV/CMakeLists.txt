cmake_minimum_required(VERSION 3.25)

set(TSOMPSIV_PAPER_TARGETS
    R1TSOMPsiV
    R2TSOMPsiV
    R3TSOMPsiV
    R4TSOMPsiV
    R5TSOMPsiV
    R6TSOMPsiV
    R7TSOMPsiV
)

foreach(target IN LISTS TSOMPSIV_PAPER_TARGETS)
    add_executable(${target} ${target}/${target}.cpp)
    target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/common")
    target_link_libraries(${target}
        PRIVATE
            Anomalous::bgclib
            MPI::MPI_CXX
            PkgConfig::PETSC
    )
    if(TARGET anomalous_project_options)
        target_link_libraries(${target} PRIVATE anomalous_project_options)
    endif()
endforeach()
