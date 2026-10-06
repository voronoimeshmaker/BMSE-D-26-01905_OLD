#pragma once

// Pointwise comparison between numerical fields and exact fields.

#include <span>
#include <vector>

#include <petsc.h>

namespace bgc {

struct FieldComparisonPoint {
    PetscInt index {0};
    PetscReal x {0.0};
    PetscReal numerical {0.0};
    PetscReal exact {0.0};
    PetscReal error {0.0};
};

[[nodiscard]] std::vector<FieldComparisonPoint> compareFields(
    std::span<const PetscReal> numerical,
    std::span<const PetscReal> exact,
    PetscReal cellWidth,
    PetscReal x0 = 0.0);

} // namespace bgc
