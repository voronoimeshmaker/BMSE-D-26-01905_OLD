#pragma once

// Cell-centered field statistics and moment diagnostics.
//
// These routines operate on plain arrays sampled at finite-volume cell centers.
// The caller supplies the control-volume width, so the same code can be used by
// MMS drivers, physical campaigns, and future model-independent diagnostics.
// No PETSc Vec ownership is involved here; PETSc scalar types are used only to
// keep numerical precision consistent with the rest of bgclib.

#include <span>

#include <petsc.h>

namespace bgc {

struct FieldStatistics {
    // Integral approximated by h * sum(values).
    PetscReal integral {0.0};

    // Arithmetic cell average, sum(values) / values.size().
    PetscReal average {0.0};

    // Minimum sampled cell value.
    PetscReal minimum {0.0};

    // Maximum sampled cell value.
    PetscReal maximum {0.0};
};

struct CenteredMomentDiagnostics {
    // Zeroth moment, usually interpreted as mass.
    PetscReal mass {0.0};

    // Raw second centered moment, integral((x-center)^2 * value dx).
    PetscReal moment2Raw {0.0};

    // Normalized second moment, moment2Raw / mass when mass is nonzero.
    PetscReal moment2 {0.0};

    PetscReal minimum {0.0};
    PetscReal maximum {0.0};

    // Near-boundary diagnostic samples. For cell-centered grids these use the
    // first interior cells adjacent to the boundary layer used by BGC programs.
    PetscReal boundaryWest {0.0};
    PetscReal boundaryEast {0.0};
};

[[nodiscard]] FieldStatistics computeFieldStatistics(std::span<const PetscReal> values,
                                                     PetscReal cellWidth);

[[nodiscard]] PetscReal computeCenteredRawMoment2(std::span<const PetscReal> values,
                                                  PetscReal cellWidth,
                                                  PetscReal x0,
                                                  PetscReal center);

[[nodiscard]] CenteredMomentDiagnostics computeCenteredMomentDiagnostics(
    std::span<const PetscReal> values,
    PetscReal cellWidth,
    PetscReal x0,
    PetscReal center);

} // namespace bgc
