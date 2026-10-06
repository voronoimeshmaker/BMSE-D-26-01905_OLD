cmake_minimum_required(VERSION 3.20)
project(MMS1PsiVDirichlet LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(PETSc REQUIRED)

add_executable(run_mms1_psiv_dirichlet
    mms1_psiv_dirichlet_driver.cpp
)

target_link_libraries(run_mms1_psiv_dirichlet PRIVATE PETSc::PETSc)
