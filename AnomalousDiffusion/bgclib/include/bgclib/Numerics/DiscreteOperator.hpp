#pragma once

// Generic model-independent sparse operator description.
//
// This is the operator representation shared by all models. Model-specific
// code, such as Models/BGC/Operator.cpp, builds this plain data. PETSc-specific
// insertion, preallocation, assembly, and ownership details are handled by
// generic assembly code.

#include <vector>

#include <petsc.h>

#include <bgclib/Core/MatrixLayout.hpp>

namespace bgc {

struct OperatorEntry {
    // Global matrix row.
    PetscInt  row   {0};

    // Global matrix column.
    PetscInt  col   {0};

    // Coefficient inserted at (row, col).
    PetscReal value {0.0};
};

struct DiscreteOperator {
    // Matrix dimensions and scalar/block structure.
    MatrixLayout               layout;

    // Sparse triplets emitted by a model-specific operator builder.
    std::vector<OperatorEntry> entries;

    [[nodiscard]] bool isValid() const noexcept {
        return layout.isValid();
    }
};

} // namespace bgc
