#pragma once

// One-dimensional uniform grid metadata.
//
// This is a small DOD value type. It does not allocate PETSc objects and does
// not own discretization stencils. Numerical kernels can pass it by value or
// const reference without dragging runtime state into model code.

#include <petsc.h>

namespace bgc {

struct Grid1D {
    // Number of nodal points in the one-dimensional domain.
    PetscInt  nx     {0};

    // Physical domain length. The right boundary is x0 + length.
    PetscReal length {1.0};

    // Left boundary coordinate.
    PetscReal x0     {0.0};

    [[nodiscard]] constexpr PetscReal dx() const noexcept {
        return nx > 1 ? length / static_cast<PetscReal>(nx - 1) : 0.0;
    }

    [[nodiscard]] constexpr PetscReal x(const PetscInt i) const noexcept {
        return x0 + static_cast<PetscReal>(i) * dx();
    }

    [[nodiscard]] constexpr bool isValid() const noexcept {
        return nx >= 2 && length > 0.0;
    }
};

} // namespace bgc
