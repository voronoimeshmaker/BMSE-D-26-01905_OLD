#pragma once

// ---------------------------------------------------------------------------
//  Models/BGC/AssemblyBGC.hpp
//
//  Montagem da matriz e do vetor RHS do modelo BGC em objetos PETSc.
//
//  Os coeficientes são calculados por CoeffBGC e passados aqui como
//  argumento — sem recálculo interno.
//
//  Fluxo de uso no loop transiente:
//
//      // Uma vez antes do loop
//      const auto sc = computeCoefficientsBGC(cfg);
//      PetscCall(assembleMatrixBGC(cfg, st, sc));
//      PetscCall(MatShift(st.A, h5/dt));   // termo temporal
//
//      // A cada passo temporal
//      const auto rhs = computeRHSBGC(cfg, t);
//      PetscCall(assembleRHSBGC(cfg, st, rhs));
// ---------------------------------------------------------------------------

#include <petsc.h>
#include <bgclib/Core/Coefficients.hpp>

namespace bgc {

struct SimConfig;
struct SimState;

PetscErrorCode assembleMatrixCBGC(const SimConfig&           cfg,
                                  SimState&                  st,
                                  const StencilCoefficients& sc);

PetscErrorCode assembleRHSCBGC(const SimConfig&       cfg,
                               SimState&              st,
                               const RHSCoefficients& rhs);

} // namespace bgc