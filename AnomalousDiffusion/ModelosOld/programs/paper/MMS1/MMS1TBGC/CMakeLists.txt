# ------------------------------------------------------------
# programs/CMakeLists.txt
# - Um executável por subpasta que contenha pelo menos um .cpp
# - Todos os .cpp da subpasta são compilados juntos
# - Linka com bgclib (se existir) e dependências globais (MPI, PETSc, yaml-cpp)
# - Mantém alvos 'programs', 'run_all_programs' e 'run_<programa>'
# ------------------------------------------------------------
cmake_minimum_required(VERSION 3.25)

# Políticas úteis
if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

# ------------------------------------------------------------
# Configurações herdadas (se o CMakeLists.txt principal já as definiu)
# ------------------------------------------------------------
# A menos que este arquivo seja chamado standalone, assumimos que
# o pai já configurou as dependências e flags.
# Se for standalone (para testes), podemos definir algumas defaults.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  message(WARNING "Este arquivo deve ser incluído pelo CMakeLists.txt raiz.")
  # Fallback mínimo
  set(CMAKE_CXX_STANDARD 23)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  find_package(MPI REQUIRED)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(PETSC REQUIRED IMPORTED_TARGET PETSc)
  find_package(yaml-cpp REQUIRED)
  set(BGCLIB_TARGET bgclib)  # supondo que a biblioteca esteja disponível
endif()

# Diretório de saída dos executáveis (herdado do pai, ou definido)
if(NOT DEFINED CMAKE_RUNTIME_OUTPUT_DIRECTORY)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/exec")
endif()

# ------------------------------------------------------------
# Alvos agregados
# ------------------------------------------------------------
add_custom_target(programs COMMENT "Compilando todos os programas")
add_custom_target(run_all_programs COMMENT "Executando todos os programas...")

# ------------------------------------------------------------
# Varre todas as subpastas imediatas que contenham pelo menos um .cpp
# ------------------------------------------------------------
file(GLOB SUBDIRS ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(subdir ${SUBDIRS})
  if(IS_DIRECTORY ${subdir})
    # Coleta todos os .cpp dentro da subpasta (recursivamente)
    file(GLOB_RECURSE SRC_LIST CONFIGURE_DEPENDS ${subdir}/*.cpp)
    if(SRC_LIST)
      # Nome do alvo = nome da pasta
      get_filename_component(prog_name ${subdir} NAME)

      # Evita conflito de nomes (embora improvável)
      if(TARGET ${prog_name})
        string(MD5 hash ${subdir})
        string(SUBSTRING ${hash} 0 8 short)
        set(prog_name "${prog_name}__${short}")
        message(WARNING "Alvo ${prog_name} já existe; renomeado para ${prog_name}__${short}")
      endif()

      message(STATUS "Programa detectado: ${prog_name} (pasta ${subdir})")

      # Cria executável com todos os fontes encontrados
      add_executable(${prog_name} ${SRC_LIST})

      # Link com bgclib (se existir) e dependências globais
      if(TARGET bgclib)
        target_link_libraries(${prog_name} PRIVATE bgclib)
      else()
        message(WARNING "Alvo 'bgclib' não encontrado. Pode ser que a biblioteca não tenha sido incluída antes.")
      endif()
      target_link_libraries(${prog_name} PRIVATE
        MPI::MPI_CXX
        yaml-cpp::yaml-cpp
        PkgConfig::PETSC
      )

      # Agrega ao alvo 'programs'
      add_dependencies(programs ${prog_name})

      # Alvo run_<prog_name>
      add_custom_target(run_${prog_name}
        DEPENDS ${prog_name}
        COMMAND ${MPIEXEC_EXECUTABLE}
                ${MPIEXEC_NUMPROC_FLAG}
                $<IF:$<BOOL:$ENV{NP}>,$ENV{NP},2>
                $<TARGET_FILE:${prog_name}>
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Executando programa ${prog_name}"
      )
      add_dependencies(run_all_programs run_${prog_name})
    endif()
  endif()
endforeach()

# Se nenhum programa foi encontrado, avisa
if(NOT TARGET programs)
  message(STATUS "Nenhum programa encontrado em ${CMAKE_CURRENT_SOURCE_DIR}")
endif()