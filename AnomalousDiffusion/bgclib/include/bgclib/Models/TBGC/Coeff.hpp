#pragma once

// TBGC coefficient data and coefficient-building functions.
//
// The old-library reference is TBGCS. In this library the model name is TBGC.
// The block structure is part of the model and should not be collapsed away:
//
//     [ A11 A12 ] [ phi ] = [ b1 ]
//     [ A21 A22 ] [ mu  ]   [ b2 ]
//
// A22 is the identity and is emitted by the operator builder. Coefficient
// functions are pure data builders and do not allocate PETSc objects.

#include <petsc.h>

#include <bgclib/Core/BoundarySet.hpp>
#include <bgclib/Core/Grid1D.hpp>
#include <bgclib/Core/TimeConfig.hpp>
#include <bgclib/Models/TBGC/Model.hpp>
#include <bgclib/Numerics/DiscreteRHS.hpp>
#include <bgclib/Numerics/Stencil.hpp>

namespace bgc::models::tbgc {

struct StencilSet {
    // First control-volume row, absolute columns.
    StencilRow vol0;

    // Reusable interior row, relative column offsets.
    StencilRow interior;

    // Last control-volume row, absolute columns.
    StencilRow volNm1;
};

struct Coefficients {
    // Transient and fourth-order contribution in the phi equation.
    StencilSet A11;

    // Coupling from mu into the phi equation.
    StencilSet A12;

    // Coupling from phi into the constitutive equation.
    StencilSet A21;
};

struct BoundaryRHSCoefficients {
    // First control-volume boundary contribution.
    PetscReal vol0 {0.0};

    // Last control-volume boundary contribution.
    PetscReal volNm1 {0.0};
};

struct RHSCoefficients {
    // RHS contributions for the first block equation.
    BoundaryRHSCoefficients b1;

    // RHS contributions for the second block equation.
    BoundaryRHSCoefficients b2;
};

// Finite-volume spacing h = length / nx for cell-centered TBGC formulas.
[[nodiscard]] PetscReal controlVolumeSpacing(const Grid1D& grid) noexcept;

// Builds TBGC block coefficients. A22 is identity and is emitted by Operator.
[[nodiscard]] Coefficients computeCoefficients(const Grid1D& grid,
                                               const TimeConfig& time,
                                               const Constants& constants);

// Builds boundary-induced RHS terms for both block equations.
[[nodiscard]] RHSCoefficients computeBoundaryRHS(const Grid1D& grid,
                                                 const Constants& constants,
                                                 const BoundarySet& boundaries,
                                                 PetscReal t);

// Converts boundary RHS coefficients into generic flattened row/value entries.
[[nodiscard]] DiscreteRHS buildBoundaryRHS(const Grid1D& grid,
                                           const RHSCoefficients& rhs);

} // namespace bgc::models::tbgc
