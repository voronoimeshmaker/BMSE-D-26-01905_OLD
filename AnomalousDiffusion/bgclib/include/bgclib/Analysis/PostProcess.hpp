#pragma once

// -----------------------------------------------------------------------------
// PostProcess.hpp
//
// Small post-processing utilities shared by programs and runtime drivers.
//
// The analysis layer should hold operations that inspect results but do not
// assemble, solve, or mutate model coefficients. Examples include sampling
// schedules, convergence summaries, norm calculations, and derived diagnostic
// quantities. Keeping these helpers here prevents MMS programs from growing
// private copies of common analysis logic.
// -----------------------------------------------------------------------------

#include <vector>

#include <petsc.h>

namespace bgc {

[[nodiscard]] std::vector<PetscInt> makeUniformSampleSteps(PetscInt finalStep,
                                                           PetscInt sampleCount);

} // namespace bgc
