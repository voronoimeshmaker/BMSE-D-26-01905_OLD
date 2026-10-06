#pragma once

// BGC coefficient data and coefficient-building functions.
//
// This is the model-specific mathematical coefficient layer for the scalar BGC
// finite-volume operator. Coefficients remain plain data and do not allocate
// PETSc Mat/Vec objects. Generic assembly code owns PETSc insertion and
// communication.
//
// The fourth-order BGC stencil has special rows at the first two and last two
// control volumes, plus one reusable interior row with relative column offsets.

#include <petsc.h>

#include <bgclib/Core/BoundarySet.hpp>
#include <bgclib/Core/Grid1D.hpp>
#include <bgclib/Models/BGC/Model.hpp>
#include <bgclib/Numerics/DiscreteRHS.hpp>
#include <bgclib/Numerics/Stencil.hpp>

namespace bgc::models::bgc {

struct Coefficients {
    // First control volume, absolute columns.
    StencilRow vol0;

    // Second control volume, absolute columns.
    StencilRow vol1;

    // Interior control volumes, relative offsets around the current row.
    StencilRow interior;

    // Second-to-last control volume, absolute columns.
    StencilRow volNm2;

    // Last control volume, absolute columns.
    StencilRow volNm1;
};

struct RHSCoefficients {
    // Boundary-induced RHS contribution for the first control volume.
    PetscReal vol0 {0.0};

    // Boundary-induced RHS contribution for the second control volume.
    PetscReal vol1 {0.0};

    // Boundary-induced RHS contribution for the second-to-last control volume.
    PetscReal volNm2 {0.0};

    // Boundary-induced RHS contribution for the last control volume.
    PetscReal volNm1 {0.0};
};

// Finite-volume spacing h = length / nx for cell-centered BGC formulas.
[[nodiscard]] PetscReal controlVolumeSpacing(const Grid1D& grid) noexcept;

// Builds the BGC matrix stencil coefficients for the supplied Bv.
[[nodiscard]] Coefficients computeCoefficients(const Grid1D& grid,
                                               const Constants& constants);

// Builds RHS terms introduced by nonzero boundary data.
[[nodiscard]] RHSCoefficients computeBoundaryRHS(const Grid1D& grid,
                                                 const Constants& constants,
                                                 const BoundarySet& boundaries,
                                                 PetscReal t);

// Converts boundary RHS coefficients into generic row/value entries.
[[nodiscard]] DiscreteRHS buildBoundaryRHS(const Grid1D& grid,
                                           const RHSCoefficients& rhs);

} // namespace bgc::models::bgc
