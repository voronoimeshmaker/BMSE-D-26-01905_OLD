#pragma once

// ---------------------------------------------------------------------------
//  Models/TBGC/AssemblyTBGC.hpp
//
//  Montagem dos blocos da matriz e dos vetores RHS do modelo TBGC.
//
//  Fluxo de uso no loop transiente:
//      const auto tc = computeCoefficientsTBGC(cfg);
//      PetscCall(assembleMatrixTBGC(cfg, st, tc));
//
//      const auto rhs = computeRHSTBGC(cfg, t);
//      PetscCall(assembleRHSTBGC(cfg, st, rhs));
// ---------------------------------------------------------------------------

#include <petsc.h>
#include <bgclib/Models/TBGC/CoeffTBGC.hpp>

namespace bgc {

struct SimConfig;
struct SimState;

PetscErrorCode assembleMatrixTBGC(const SimConfig&        cfg,
                                   SimState&               st,
                                   const TBGCCoefficients& tc);

PetscErrorCode assembleRHSTBGC(const SimConfig&           cfg,
                                SimState&                  st,
                                const TBGCRHSCoefficients& rhs);

} // namespace bgc