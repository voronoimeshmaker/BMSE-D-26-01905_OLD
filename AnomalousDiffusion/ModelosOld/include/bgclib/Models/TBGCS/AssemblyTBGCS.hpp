#pragma once

// ---------------------------------------------------------------------------
//  Models/TBGCS/AssemblyTBGCS.hpp
//
//  Montagem dos blocos da matriz e dos vetores RHS do modelo TBGCS.
//
//  Fluxo de uso no loop transiente:
//      const auto tc = computeCoefficientsTBGCS(cfg);
//      PetscCall(assembleMatrixTBGCS(cfg, st, tc));
//
//      const auto rhs = computeRHSTBGCS(cfg, t);
//      PetscCall(assembleRHSTBGCS(cfg, st, rhs));
// ---------------------------------------------------------------------------

#include <petsc.h>
#include <bgclib/Models/TBGCS/CoeffTBGCS.hpp>

namespace bgc {

struct SimConfig;
struct SimState;

PetscErrorCode assembleMatrixTBGCS(const SimConfig&        cfg,
                                   SimState&               st,
                                   const TBGCSCoefficients& tc);

PetscErrorCode assembleRHSTBGCS(const SimConfig&           cfg,
                                  SimState&                  st,
                                  const TBGCSRHSCoefficients& rhs);

} // namespace bgc