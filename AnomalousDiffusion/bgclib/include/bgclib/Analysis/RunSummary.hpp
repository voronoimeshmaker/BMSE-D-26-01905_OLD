#pragma once

// Structured summary returned by standard runs and reusable in driver programs.

#include <string>

#include <petsc.h>

namespace bgc {

struct RunSummary {
    PetscInt steps {0};
    PetscReal finalTime {0.0};
    KSPConvergedReason reason {KSP_CONVERGED_ITERATING};
    PetscInt iterations {0};
    std::string status;
};

[[nodiscard]] inline const char* convergenceStatus(const KSPConvergedReason reason) noexcept {
    return reason > 0 ? "solved" : "not_converged";
}

} // namespace bgc
