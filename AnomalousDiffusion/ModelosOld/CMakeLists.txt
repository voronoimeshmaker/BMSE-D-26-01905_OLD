# ---------------------------------------------------------------------------
#  bgclib/CMakeLists.txt
#
#  Biblioteca estática do simulador de difusão anômala.
#  Não deve ter project() — este arquivo é um subdirectório do projeto raiz.
# ---------------------------------------------------------------------------

# Coleta todos os .cpp dentro de src/ (incluindo subpastas)
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

add_library(bgclib STATIC ${SOURCES})

target_include_directories(bgclib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# Vincula dependências externas
target_link_libraries(bgclib PUBLIC
    MPI::MPI_CXX
    yaml-cpp::yaml-cpp
    PkgConfig::PETSC
)

target_compile_features(bgclib PUBLIC cxx_std_23)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/lib")