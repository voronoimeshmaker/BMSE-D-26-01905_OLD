#pragma once

// Finite-volume error norm utilities backed by PETSc VecNorm.
//
// computeVectorNorms receives pointwise values sampled at cell centers and
// converts PETSc's discrete vector norms into finite-volume norms:
//
//   L1   = h * ||v||_1
//   L2   = sqrt(h) * ||v||_2
//   Linf = ||v||_inf
//
// computeErrorNorms first forms numerical - exact and then applies the same
// scaling. Keeping this in bgclib prevents individual MMS programs from owning
// slightly different norm conventions.

#include <span>

#include <petsc.h>

namespace bgc {

struct ErrorNorms {
    // Finite-volume L1 norm.
    PetscReal l1 {0.0};

    // Finite-volume L2 norm.
    PetscReal l2 {0.0};

    // Maximum absolute pointwise error.
    PetscReal linf {0.0};
};

[[nodiscard]] ErrorNorms computeVectorNorms(std::span<const PetscReal> values,
                                            PetscReal cellWidth);

[[nodiscard]] ErrorNorms computeErrorNorms(std::span<const PetscReal> numerical,
                                           std::span<const PetscReal> exact,
                                           PetscReal cellWidth);

} // namespace bgc
