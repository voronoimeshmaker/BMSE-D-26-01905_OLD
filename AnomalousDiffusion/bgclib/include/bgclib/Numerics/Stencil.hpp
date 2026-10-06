#pragma once

// Small fixed-size stencil row and related helpers.
//
// Fourth-order one-dimensional anomalous diffusion models use up to five
// coefficients per row. Boundary rows store absolute columns. Interior rows may
// store relative offsets so the same row can be reused for every interior
// control volume.

#include <array>

#include <petsc.h>

namespace bgc {

struct StencilRow {
    std::array<PetscReal, 5> coef {};
    std::array<PetscInt, 5>  col {};
    PetscInt                 ncols {0};

    [[nodiscard]] constexpr bool isValid() const noexcept {
        return ncols >= 0 && ncols <= static_cast<PetscInt>(coef.size());
    }
};

} // namespace bgc
