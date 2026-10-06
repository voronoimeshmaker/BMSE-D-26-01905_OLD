#pragma once

// Generic model-independent right-hand-side description.
//
// A model emits row/value contributions here. Generic PETSc assembly code owns
// Vec allocation, insertion mode, communication, and final assembly.

#include <vector>

#include <petsc.h>

namespace bgc {

struct RHSEntry {
    // Global vector row.
    PetscInt  row   {0};

    // Value added or inserted for this row, depending on assembly policy.
    PetscReal value {0.0};
};

struct DiscreteRHS {
    // Global vector size.
    PetscInt              size {0};

    // Sparse RHS contributions.
    std::vector<RHSEntry> entries;

    [[nodiscard]] bool isValid() const noexcept {
        return size > 0;
    }
};

} // namespace bgc
