#pragma once

// ---------------------------------------------------------------------------
//  Solver.hpp
//
//  Configuração do solver linear e resolução de A x = b.
//
//  Separação de responsabilidades:
//
//      configureSolver   —  Configura KSP e PC de acordo com o modelo.
//                           Deve ser chamada UMA ÚNICA VEZ após a montagem
//                           completa da matriz (MatAssembly*).
//                           Chama KSPSetFromOptions ao final, permitindo que
//                           opções de linha de comando sobrescrevam os
//                           valores padrão.
//
//      solveLinearSystem —  Resolve A x = b.  Pressupõe que configureSolver
//                           já foi chamada.  Não reconfigura o KSP.
//
//  Estratégia por modelo:
//
//      BGC   (matriz AIJ única):
//          useDirectSolver = true  →  PREONLY + LU (MUMPS)
//          useDirectSolver = false →  FGMRES + ILU
//
//      TBGC  (MatNest 2×2):
//          useDirectSolver = true  →  PREONLY + LU (MUMPS)
//                                     PETSc converte MatNest→AIJ internamente.
//          useDirectSolver = false →  FGMRES + FIELDSPLIT (IS de phi e mu)
//                                     com PREONLY+LU em cada sub-bloco.
//                                     A22 = I → inverse trivial;
//                                     Schur = A11 - A12 * A21 resolvido
//                                     via PREONLY+LU.
// ---------------------------------------------------------------------------

#include <petsc.h>

namespace bgc {

struct SimConfig;
struct SimState;

/// Configura KSP e PC de acordo com cfg.model e cfg.useDirectSolver.
/// Deve ser chamada após MatAssemblyEnd da matriz final.
PetscErrorCode configureSolver(const SimConfig& cfg, SimState& st);

/// Resolve A x = b.  Pressupõe configureSolver já chamada.
PetscErrorCode solveLinearSystem(const SimConfig& cfg, SimState& st);

} // namespace bgc