#pragma once

// Thermodynamic consistency checks and energy-like diagnostics.

#include <span>

#include <petsc.h>

namespace bgc {

[[nodiscard]] PetscReal computeSquaredGradientSum(std::span<const PetscReal> values);

[[nodiscard]] PetscReal computeQuadraticFreeEnergy(std::span<const PetscReal> values,
                                                   PetscReal cellWidth,
                                                   PetscReal gradientCoefficient);

} // namespace bgc
