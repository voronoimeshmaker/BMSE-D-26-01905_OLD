#pragma once

// Time-domain configuration for fixed-step transient runs.
//
// This struct intentionally stores only scalar metadata. PETSc TS/KSP objects
// belong to runtime state, while model-specific time discretization choices
// belong to NumericsConfig or model traits.

#include <petsc.h>

namespace bgc {

struct TimeConfig {
    // Fixed time step.
    PetscReal dt          {0.0};

    // Inclusive final time requested by the driver.
    PetscReal finalTime   {0.0};

    // Initial simulation time.
    PetscReal initialTime {0.0};

    [[nodiscard]] constexpr bool isValid() const noexcept {
        return dt > 0.0 && finalTime >= initialTime;
    }

    [[nodiscard]] constexpr PetscInt steps() const noexcept {
        if (!isValid()) {
            return 0;
        }

        const auto interval = (finalTime - initialTime) / dt;
        return static_cast<PetscInt>(interval + 0.5);
    }
};

} // namespace bgc
