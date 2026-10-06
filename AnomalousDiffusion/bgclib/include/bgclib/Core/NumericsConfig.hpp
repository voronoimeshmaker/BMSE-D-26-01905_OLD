#pragma once

// Numerical-method configuration.
//
// NumericsConfig stores method choices that are not part of the physical model:
// time discretization, whether the matrix can be assembled once or must be
// rebuilt, and small tolerances used by generic numerical code. Model constants
// such as Bv, lambdaC, lambdaR, or TSOM boundary modes do not belong here.

#include <petsc.h>

namespace bgc {

enum class TimeScheme {
    // Fully implicit first-order method.
    BackwardEuler,

    // Reserved for second-order theta=1/2 schemes.
    CrankNicolson,

    // Reserved for general theta-method workflows.
    Theta,
};

enum class MatrixAssemblyPolicy {
    // Matrix entries are independent of time and can be reused.
    AssembleOnce,

    // Matrix entries depend on time/state and must be rebuilt.
    AssembleEveryStep,
};

struct NumericsConfig {
    // Time-discretization family selected by a driver/runtime path.
    TimeScheme           timeScheme     {TimeScheme::BackwardEuler};

    // Controls whether a transient runner reuses or rebuilds the operator.
    MatrixAssemblyPolicy matrixAssembly {MatrixAssemblyPolicy::AssembleOnce};

    // Optional equation scaling for generic assembly/solve paths.
    PetscReal equationScale      {1.0};

    // Values below this magnitude may be treated as numerical zero.
    PetscReal zeroTolerance      {1.0e-14};

    // Boundary denominator/compatibility tolerance.
    PetscReal boundaryTolerance  {1.0e-30};
};

} // namespace bgc
