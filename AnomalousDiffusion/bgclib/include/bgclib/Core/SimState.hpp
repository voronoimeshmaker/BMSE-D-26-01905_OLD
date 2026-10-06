#pragma once

// DOD runtime state containing PETSc handles.
//
// SimState owns no policy and exposes no virtual interface. Creation,
// configuration, solve, and destruction stay in free functions so model code can
// remain statically dispatched while PETSc remains the runtime backend.

#include <petsc.h>

namespace bgc {

struct SimState {
    // Linear operator assembled for the current model/time step.
    Mat A {nullptr};

    // Solution vector.
    Vec x {nullptr};

    // Right-hand-side vector.
    Vec b {nullptr};

    // PETSc linear solver configured for A x = b.
    KSP ksp {nullptr};

    [[nodiscard]] bool hasLinearSystem() const noexcept {
        return A != nullptr && x != nullptr && b != nullptr;
    }
};

} // namespace bgc
