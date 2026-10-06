#pragma once

// ---------------------------------------------------------------------------
//  BGCLib.hpp
//
//  Header de conveniência da biblioteca bgclib.
//
//  Inclui toda a interface pública em uma única linha:
//
//      #include <bgclib/BGCLib.hpp>
//
//  Módulos disponíveis:
//
//      Types.hpp                       —  tipos fundamentais, constantes,
//                                         enumerações e utilitários
//
//      SimConfig.hpp                   —  parâmetros de configuração da
//                                         simulação (struct plana, sem PETSc)
//
//      SimState.hpp                    —  objetos PETSc alocados em tempo
//                                         de execução (vetores, matrizes, KSP)
//
//      Core/BoundaryCondition.hpp      —  condição de contorno linear genérica
//                                         alpha*Phi + beta*dPhi/dxi = gamma(t)
//                                         com suporte a Dirichlet, Neumann e Robin
//
//      Coefficients/Coefficients.hpp   —  estruturas de dados do estêncil
//                                         (StencilRow, StencilCoefficients,
//                                         RHSCoefficients, VolumeRegion)
//                                         e helpers comuns a todos os modelos
//
//      Coefficients/BGC.hpp            —  cálculo dos coeficientes do modelo BGC
//                                         (computeCoefficientsBGC, computeRHSBGC)
//
//      Coefficients/TBGC.hpp           —  cálculo dos coeficientes do modelo TBGC
//                                         (computeCoefficientsTBGC, computeRHSTBGC)
//
//      Assembly/BGC.hpp                —  montagem da matriz e RHS do BGC
//                                         em objetos PETSc
//
//      Assembly/TBGC.hpp               —  montagem da matriz e RHS do TBGC
//                                         em objetos PETSc
//
//      Solver/Solver.hpp               —  resolução do sistema linear A x = b
//                                         com KSP/PC configuráveis via YAML
//                                         ou linha de comando
//
//      Physics/Analytics.hpp           —  avaliação da solução analítica e
//                                         do termo fonte fabricado
//
//      Physics/PostProcess.hpp         —  normas de erro (L1, L2, Linf) e
//                                         verificação da consistência
//                                         termodinâmica
//
//      Transient/Transient.hpp         —  loop transiente principal;
//                                         orquestra todos os módulos acima
//
//  Inclusão seletiva:
//      Cada módulo pode ser incluído individualmente se necessário.
//      Este header não impede o uso direto de um subconjunto da lib.
//
//  Dependências externas:
//      PETSc    —  obrigatória (todos os módulos com objetos PETSc)
//      yaml-cpp —  NÃO é dependência da lib; pertence a cada programa
// ---------------------------------------------------------------------------

#include <bgclib/Misc/Types.hpp>
#include <bgclib/SimConfig.hpp>
#include <bgclib/SimState.hpp>
#include <bgclib/Core/BoundaryCondition.hpp>
#include <bgclib/Core/Coefficients.hpp>
// #include <bgclib/Coefficients/BGC.hpp>
// #include <bgclib/Coefficients/TBGC.hpp>
// #include <bgclib/Models/TSPV/CoeffTSPV.hpp>
// #include <bgclib/Assembly/BGC.hpp>
// #include <bgclib/Assembly/TBGC.hpp>
// #include <bgclib/Models/TSPV/AssemblyTSPV.hpp>
// #include <bgclib/Solver/Solver.hpp>
// #include <bgclib/Physics/Analytics.hpp>
// #include <bgclib/Physics/PostProcess.hpp>
// #include <bgclib/Transient/Transient.hpp>