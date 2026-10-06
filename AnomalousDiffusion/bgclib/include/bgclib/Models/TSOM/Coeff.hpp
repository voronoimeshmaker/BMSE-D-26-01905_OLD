#pragma once

// TSOM coefficient data and coefficient-building functions.
//
// This is the new-library DOD translation of the old CoeffTSOM formulas. The
// model is a two-field system for U and V:
//
//   [ A11 A12 ] [ U ] = [ b1 ]
//   [ A21 A22 ] [ V ]   [ b2 ]
//
// The V boundary-condition mode is part of Constants, not an MMS-specific
// choice. Each mode may alter both the boundary rows of A and the boundary
// source terms. Interior coefficients are shared across the modes.

#include <petsc.h>

#include <bgclib/Core/BoundarySet.hpp>
#include <bgclib/Core/Grid1D.hpp>
#include <bgclib/Core/TimeConfig.hpp>
#include <bgclib/Models/TSOM/Model.hpp>
#include <bgclib/Numerics/DiscreteRHS.hpp>
#include <bgclib/Numerics/Stencil.hpp>

namespace bgc::models::tsom {

struct DiffusionCoefficients {
    // Diffusion matrix multiplying [U_xx, V_xx]^T in the two TSOM equations.
    PetscReal d11 {0.0};
    PetscReal d12 {0.0};
    PetscReal d21 {0.0};
    PetscReal d22 {0.0};
};

struct StencilSet {
    // First control-volume row.
    StencilRow first;

    // Reusable interior row with columns stored as relative offsets.
    StencilRow interior;

    // Last control-volume row.
    StencilRow last;
};

struct Coefficients {
    // Equation for U, coefficients multiplying U.
    StencilSet A11;

    // Equation for U, coefficients multiplying V.
    StencilSet A12;

    // Equation for V, coefficients multiplying U.
    StencilSet A21;

    // Equation for V, coefficients multiplying V.
    StencilSet A22;
};

struct BoundaryRHS {
    // Source contribution added to the first control-volume equation.
    PetscReal first {0.0};

    // Source contribution added to the last control-volume equation.
    PetscReal last {0.0};
};

struct RHSCoefficients {
    // Boundary source contributions for the U equation.
    BoundaryRHS b1;

    // Boundary source contributions for the V equation.
    BoundaryRHS b2;
};

// Finite-volume spacing h = length / nx for cell-centered TSOM formulas.
[[nodiscard]] PetscReal controlVolumeSpacing(const Grid1D& grid) noexcept;

// Computes the TSOM diffusion matrix from alpha, rho, and theta.
[[nodiscard]] DiffusionCoefficients computeDiffusionCoefficients(
    const Constants& constants) noexcept;

// Builds all TSOM coefficient blocks for the configured V boundary mode.
[[nodiscard]] Coefficients computeCoefficients(const Grid1D& grid,
                                               const TimeConfig& time,
                                               const Constants& constants,
                                               const BoundarySet& boundaries);

// Builds boundary source terms for the configured V boundary mode.
[[nodiscard]] RHSCoefficients computeBoundaryRHS(const Grid1D& grid,
                                                 const Constants& constants,
                                                 const BoundarySet& boundaries,
                                                 PetscReal t);

// Converts boundary RHS coefficients into flattened row/value entries.
[[nodiscard]] DiscreteRHS buildBoundaryRHS(const Grid1D& grid,
                                           const RHSCoefficients& rhs);

} // namespace bgc::models::tsom
