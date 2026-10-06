#pragma once

// Generic local truncation-error and discrete-residual analysis.
//
// Models provide a DiscreteOperator plus sampled analytical/source fields. This
// layer uses the generic PETSc assembly modules to evaluate:
//
//   tau = A * u_exact - source
//
// The same routine applies to BGC, TBGC, TSOM, and future models once
// they expose their model-specific operator in the generic DiscreteOperator
// format.

#include <vector>
#include <filesystem>

#include <petsc.h>

#include <bgclib/Numerics/DiscreteOperator.hpp>

namespace bgc {

// Computes the discrete residual of an exact manufactured field.
//
// exactValues and sourceValues must have the same size as the operator domain
// and range used by the model. tauValues is resized to the resulting PETSc Vec.
PetscErrorCode computeLocalTruncationError(
    const DiscreteOperator& op,
    const std::vector<PetscReal>& exactValues,
    const std::vector<PetscReal>& sourceValues,
    std::vector<PetscReal>& tauValues);

// Writes a dense ASCII view of the generic operator after PETSc assembly.
//
// This is intended for debugging small systems and MMS diagnostics.
PetscErrorCode writePetscDenseMatrixView(const std::filesystem::path& filePath,
                                         const DiscreteOperator& op);

} // namespace bgc
