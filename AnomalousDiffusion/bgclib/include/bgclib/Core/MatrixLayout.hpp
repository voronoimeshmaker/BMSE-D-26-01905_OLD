#pragma once

// Matrix layout descriptors used before PETSc Mat allocation.
//
// Models describe the mathematical layout here; generic PETSc assembly code
// decides how to allocate and fill Mat objects from this data.

#include <petsc.h>

namespace bgc {

enum class MatrixLayoutKind {
    // One scalar unknown per grid row.
    Scalar,

    // Multiple coupled unknowns per grid row.
    Block,
};

struct MatrixLayout {
    // Scalar or block matrix family.
    MatrixLayoutKind kind      {MatrixLayoutKind::Scalar};

    // Global matrix row count.
    PetscInt         rows      {0};

    // Global matrix column count.
    PetscInt         cols      {0};

    // Number of coupled unknowns per logical grid point for block layouts.
    PetscInt         blockSize {1};

    [[nodiscard]] constexpr bool isSquare() const noexcept {
        return rows == cols;
    }

    [[nodiscard]] constexpr bool isValid() const noexcept {
        return rows > 0 && cols > 0 && blockSize > 0;
    }
};

} // namespace bgc
