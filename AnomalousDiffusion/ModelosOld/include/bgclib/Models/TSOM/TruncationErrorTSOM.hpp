#pragma once

#include <petsc.h>

#include <bgclib/Misc/Types.hpp>
#include <bgclib/SimConfig.hpp>

#include <vector>
 
namespace bgc {

struct TSOMTruncationError {
    PetscReal l2U    {0.0};  ///< Discrete L2 norm of the LTE in the U equation.
    PetscReal l2V    {0.0};  ///< Discrete L2 norm of the LTE in the V equation.
    PetscReal linfU  {0.0};  ///< L-infinity norm of the LTE in the U equation.
    PetscReal linfV  {0.0};  ///< L-infinity norm of the LTE in the V equation.
};

// Computes the local truncation error of the TSOM backward-Euler finite-volume
// scheme by substituting the exact solution into the fully discrete equations.
[[nodiscard]]
PetscErrorCode computeTruncationErrorTSOM(const SimConfig&            cfg,
                                          const FieldFn&              exactUFn,
                                          const FieldFn&              exactVFn,
                                          const FieldFn&              sourceUFn,
                                          const FieldFn&              sourceVFn,
                                          PetscReal                   tPrev,
                                          PetscReal                   tNext,
                                          TSOMTruncationError&        te,
                                          std::vector<PetscReal>*     tauU = nullptr,
                                          std::vector<PetscReal>*     tauV = nullptr);

} // namespace bgc
