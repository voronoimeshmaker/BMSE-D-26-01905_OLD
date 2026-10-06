#pragma once

#include <petsc.h>

#include <bgclib/Misc/Types.hpp>
#include <bgclib/Models/TSOM/CoeffTSOM.hpp>

namespace bgc { 

struct SimConfig;
struct SimState;

PetscErrorCode assembleMatrixTSOM(const SimConfig&        cfg,
                                  SimState&               st,
                                  const TSOMCoefficients& tc);

PetscErrorCode assembleRHSTSOM(const SimConfig&           cfg,
                               SimState&                  st,
                               const TSOMRHSCoefficients& rhs);

// Assembles the full RHS for TSOM:
//   - boundary terms returned by computeRHSTSOM
//   - backward-Euler mass term using the previous-step fields
//     (phi_0 = U^n, mu_0 = V^n)
//   - manufactured-source terms in the interior
PetscErrorCode assembleFullRHSTSOM(const SimConfig&           cfg,
                                   SimState&                  st,
                                   const TSOMRHSCoefficients& rhs,
                                   PetscReal                  t,
                                   const FieldFn&             sourceUFn,
                                   const FieldFn&             sourceVFn);

} // namespace bgc
