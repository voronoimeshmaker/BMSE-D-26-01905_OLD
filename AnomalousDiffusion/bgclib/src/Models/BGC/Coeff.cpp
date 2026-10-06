#include <bgclib/Models/BGC/Coeff.hpp>

#include <cmath>
#include <limits>

// BGC coefficient implementation.
//
// The scalar BGC operator is assembled as five groups of rows:
//
//   vol0, vol1, interior, volNm2, volNm1
//
// Boundary-adjacent rows use absolute global columns because their closures are
// different from the interior stencil. Interior rows use relative offsets and
// are expanded by the operator builder. Boundary RHS terms are computed
// separately so non-homogeneous boundary data does not get hidden inside matrix
// coefficients.

namespace bgc::models::bgc {

PetscReal controlVolumeSpacing(const Grid1D& grid) noexcept {
    return grid.nx > 0 ? grid.length / static_cast<PetscReal>(grid.nx) : 0.0;
}

Coefficients computeCoefficients(const Grid1D& grid, const Constants& constants) {
    const PetscInt n = grid.nx;
    const PetscReal h = controlVolumeSpacing(grid);
    const PetscReal h2 = h * h;
    const PetscReal h3 = h2 * h;
    const PetscReal bv = constants.bv;

    Coefficients coeffs;

    {
        const PetscReal deno = 1.0 / (225.0 * h3);
        const PetscReal ap = 225.0 * h2 + 10800.0 * bv;
        const PetscReal ae = -(250.0 * h2 + 2400.0 * bv);
        const PetscReal aee = 9.0 * h2 + 432.0 * bv;

        coeffs.vol0.ncols = 3;
        coeffs.vol0.col[0] = 0;
        coeffs.vol0.coef[0] = ap * deno;
        coeffs.vol0.col[1] = 1;
        coeffs.vol0.coef[1] = ae * deno;
        coeffs.vol0.col[2] = 2;
        coeffs.vol0.coef[2] = aee * deno;
    }

    {
        const PetscReal deno = 1.0 / (3600.0 * h3);
        const PetscReal aw = -(3750.0 * h2 + 3600.0 * bv);
        const PetscReal ap = 8050.0 * h2 + 20400.0 * bv;
        const PetscReal ae = -(4194.0 * h2 + 14256.0 * bv);
        const PetscReal aee = 150.0 * h2 + 3600.0 * bv;

        coeffs.vol1.ncols = 4;
        coeffs.vol1.col[0] = 0;
        coeffs.vol1.coef[0] = aw * deno;
        coeffs.vol1.col[1] = 1;
        coeffs.vol1.coef[1] = ap * deno;
        coeffs.vol1.col[2] = 2;
        coeffs.vol1.coef[2] = ae * deno;
        coeffs.vol1.col[3] = 3;
        coeffs.vol1.coef[3] = aee * deno;
    }

    {
        const PetscReal aww = 1.0 / (24.0 * h) + bv / h3;
        const PetscReal aw = -7.0 / (6.0 * h) - 4.0 * bv / h3;
        const PetscReal ap = 9.0 / (4.0 * h) + 6.0 * bv / h3;

        coeffs.interior.ncols = 5;
        coeffs.interior.col[0] = -2;
        coeffs.interior.coef[0] = aww;
        coeffs.interior.col[1] = -1;
        coeffs.interior.coef[1] = aw;
        coeffs.interior.col[2] = 0;
        coeffs.interior.coef[2] = ap;
        coeffs.interior.col[3] = 1;
        coeffs.interior.coef[3] = aw;
        coeffs.interior.col[4] = 2;
        coeffs.interior.coef[4] = aww;
    }

    {
        const PetscReal deno = 1.0 / (3600.0 * h3);
        const PetscReal ae = -(3750.0 * h2 + 3600.0 * bv);
        const PetscReal ap = 8050.0 * h2 + 20400.0 * bv;
        const PetscReal aw = -(4194.0 * h2 + 14256.0 * bv);
        const PetscReal aww = 150.0 * h2 + 3600.0 * bv;

        coeffs.volNm2.ncols = 4;
        coeffs.volNm2.col[0] = n - 4;
        coeffs.volNm2.coef[0] = aww * deno;
        coeffs.volNm2.col[1] = n - 3;
        coeffs.volNm2.coef[1] = aw * deno;
        coeffs.volNm2.col[2] = n - 2;
        coeffs.volNm2.coef[2] = ap * deno;
        coeffs.volNm2.col[3] = n - 1;
        coeffs.volNm2.coef[3] = ae * deno;
    }

    {
        const PetscReal deno = 1.0 / (225.0 * h3);
        const PetscReal ap = 225.0 * h2 + 10800.0 * bv;
        const PetscReal aw = -(250.0 * h2 + 2400.0 * bv);
        const PetscReal aww = 9.0 * h2 + 432.0 * bv;

        coeffs.volNm1.ncols = 3;
        coeffs.volNm1.col[0] = n - 3;
        coeffs.volNm1.coef[0] = aww * deno;
        coeffs.volNm1.col[1] = n - 2;
        coeffs.volNm1.coef[1] = aw * deno;
        coeffs.volNm1.col[2] = n - 1;
        coeffs.volNm1.coef[2] = ap * deno;
    }

    return coeffs;
}

RHSCoefficients computeBoundaryRHS(const Grid1D& grid,
                                   const Constants& constants,
                                   const BoundarySet& boundaries,
                                   const PetscReal t) {
    const PetscReal h = controlVolumeSpacing(grid);
    const PetscReal h2 = h * h;
    const PetscReal h3 = h2 * h;
    const PetscReal bv = constants.bv;
    const PetscReal eps = 1.0e-30;
    const PetscReal nan = std::numeric_limits<PetscReal>::quiet_NaN();

    const BoundaryCondition& w1 = boundaries.west[0];
    const BoundaryCondition& w2 = boundaries.west[1];
    const BoundaryCondition& e1 = boundaries.east[0];
    const BoundaryCondition& e2 = boundaries.east[1];

    const PetscReal a1w = w1.alpha();
    const PetscReal b1w = w1.beta();
    const PetscReal a2w = w2.alpha();
    const PetscReal b2w = w2.beta();
    const PetscReal g1w = w1.gamma(t);
    const PetscReal g2w = w2.gamma(t);

    const PetscReal a1e = e1.alpha();
    const PetscReal b1e = e1.beta();
    const PetscReal a2e = e2.alpha();
    const PetscReal b2e = e2.beta();
    const PetscReal g1e = e1.gamma(t);
    const PetscReal g2e = e2.gamma(t);

    const PetscReal detW = b2w * a1w - a2w * b1w;
    const PetscReal detE = b2e * a1e - a2e * b1e;

    RHSCoefficients rhs;

    if (std::abs(detW) <= eps) {
        rhs.vol0 = nan;
        rhs.vol1 = nan;
    } else {
        {
            const PetscReal deno = 225.0 * detW * h3;
            const PetscReal sp1 = -(2880.0 * h * bv + 240.0 * h3) * a2w +
                                  (8832.0 * bv - 16.0 * h2) * b2w;
            const PetscReal sp2 = (2880.0 * h * bv - 240.0 * h3) * a1w -
                                  (8832.0 * bv - 16.0 * h2) * b1w;
            rhs.vol0 = (sp1 * g1w + sp2 * g2w) / deno;
        }

        {
            const PetscReal deno = 225.0 * detW * h3 / (h2 + 24.0 * bv);
            const PetscReal sp1 = 16.0 * b2w - 15.0 * h * a2w;
            const PetscReal sp2 = -(16.0 * b1w - 15.0 * h * a1w);
            rhs.vol1 = (sp1 * g1w + sp2 * g2w) / deno;
        }
    }

    if (std::abs(detE) <= eps) {
        rhs.volNm2 = nan;
        rhs.volNm1 = nan;
    } else {
        {
            const PetscReal deno = 225.0 * detE * h3 / (h2 + 24.0 * bv);
            const PetscReal sp1 = 16.0 * b2e + 15.0 * h * a2e;
            const PetscReal sp2 = -(16.0 * b1e + 15.0 * h * a1e);
            rhs.volNm2 = (sp1 * g1e + sp2 * g2e) / deno;
        }

        {
            const PetscReal deno = 225.0 * detE * h3;
            const PetscReal sp1 = (2880.0 * h * bv - 240.0 * h3) * a2e +
                                  (8832.0 * bv - 16.0 * h2) * b2e;
            const PetscReal sp2 = -(2880.0 * h * bv - 240.0 * h3) * a1e -
                                  (8832.0 * bv - 16.0 * h2) * b1e;
            rhs.volNm1 = (sp1 * g1e + sp2 * g2e) / deno;
        }
    }

    return rhs;
}

DiscreteRHS buildBoundaryRHS(const Grid1D& grid, const RHSCoefficients& rhs) {
    return {
        .size = grid.nx,
        .entries = {
            {.row = 0, .value = rhs.vol0},
            {.row = 1, .value = rhs.vol1},
            {.row = grid.nx - 2, .value = rhs.volNm2},
            {.row = grid.nx - 1, .value = rhs.volNm1},
        },
    };
}

} // namespace bgc::models::bgc
