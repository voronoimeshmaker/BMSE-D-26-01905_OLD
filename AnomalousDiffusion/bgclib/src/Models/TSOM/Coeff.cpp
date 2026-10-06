#include <bgclib/Models/TSOM/Coeff.hpp>

#include <stdexcept>

// TSOM coefficient assembly is split by responsibility:
//
//   computeInteriorCoefficients  -> rows unaffected by V boundary mode
//   compute*BoundaryCoefficients -> first/last rows for each V boundary mode
//   compute*BoundaryRHS          -> source terms induced by each V boundary mode
//
// This separation mirrors the Maple derivations and makes it explicit that
// changing the V boundary condition can change both A and the RHS.

namespace bgc::models::tsom {

namespace {

struct CoefficientWorkData {
    PetscInt n {0};
    PetscReal h {0.0};
    PetscReal hinv {0.0};
    PetscReal dhdt {0.0};
    PetscReal lambdaC {0.0};
    PetscReal lambdaR {0.0};
    DiffusionCoefficients d {};
    PetscReal alphaw {0.0};
    PetscReal betaw {0.0};
    PetscReal alphae {0.0};
    PetscReal betae {0.0};
};

CoefficientWorkData makeWorkData(const Grid1D& grid,
                                 const TimeConfig& time,
                                 const Constants& constants,
                                 const BoundarySet& boundaries) {
    const BoundaryCondition& west = boundaries.west[0];
    const BoundaryCondition& east = boundaries.east[0];

    CoefficientWorkData w {
        .n = grid.nx,
        .h = controlVolumeSpacing(grid),
        .lambdaC = constants.lambdaC,
        .lambdaR = constants.lambdaR,
        .d = computeDiffusionCoefficients(constants),
        .alphaw = west.alpha(),
        .betaw = west.beta(),
        .alphae = east.alpha(),
        .betae = east.beta(),
    };
    w.hinv = 1.0 / w.h;
    w.dhdt = w.h / time.dt;

    if (w.alphaw == 0.0 && w.betaw == 0.0) {
        throw std::runtime_error("computeCoefficients(TSOM): invalid west Psi boundary.");
    }
    if (w.alphae == 0.0 && w.betae == 0.0) {
        throw std::runtime_error("computeCoefficients(TSOM): invalid east Psi boundary.");
    }
    return w;
}

void computeInteriorCoefficients(Coefficients& tc, const CoefficientWorkData& w) {
    {
        StencilSet& A = tc.A11;
        A.interior.ncols = 3;
        A.interior.col[0] = -1;
        A.interior.coef[0] = -w.d.d11 * w.hinv;
        A.interior.col[1] = 0;
        A.interior.coef[1] = w.dhdt + w.lambdaC * w.h + 2.0 * w.d.d11 * w.hinv;
        A.interior.col[2] = 1;
        A.interior.coef[2] = -w.d.d11 * w.hinv;
    }

    {
        StencilSet& A = tc.A12;
        A.interior.ncols = 3;
        A.interior.col[0] = -1;
        A.interior.coef[0] = -w.d.d12 * w.hinv;
        A.interior.col[1] = 0;
        A.interior.coef[1] = -w.lambdaR * w.h + 2.0 * w.d.d12 * w.hinv;
        A.interior.col[2] = 1;
        A.interior.coef[2] = -w.d.d12 * w.hinv;
    }

    {
        StencilSet& A = tc.A21;
        A.interior.ncols = 3;
        A.interior.col[0] = -1;
        A.interior.coef[0] = -w.d.d21 * w.hinv;
        A.interior.col[1] = 0;
        A.interior.coef[1] = -w.lambdaC * w.h + 2.0 * w.d.d21 * w.hinv;
        A.interior.col[2] = 1;
        A.interior.coef[2] = -w.d.d21 * w.hinv;
    }

    {
        StencilSet& A = tc.A22;
        A.interior.ncols = 3;
        A.interior.col[0] = -1;
        A.interior.coef[0] = -w.d.d22 * w.hinv;
        A.interior.col[1] = 0;
        A.interior.coef[1] = w.dhdt + w.lambdaR * w.h + 2.0 * w.d.d22 * w.hinv;
        A.interior.col[2] = 1;
        A.interior.coef[2] = -w.d.d22 * w.hinv;
    }
}

void applyZeroValueBoundaryCoefficients(Coefficients& tc, const CoefficientWorkData& w) {

    const PetscReal denow = 1.0 / (3.0 * w.alphaw * w.h - 8.0 * w.betaw);
    const PetscReal denoe = 1.0 / (3.0 * w.alphae * w.h + 8.0 * w.betae);

    {
        StencilSet& A = tc.A11;
        A.first.ncols = 2;
        A.first.col[0] = 0;
        A.first.coef[0] =
            w.dhdt + w.lambdaC * w.h +
            4.0 * w.d.d11 * (3.0 * w.alphaw * w.h - 2.0 * w.betaw) * denow * w.hinv;
        A.first.col[1] = 1;
        A.first.coef[1] = -w.d.d11 * (w.hinv + w.alphaw * denow);

        A.last.ncols = 2;
        A.last.col[0] = w.n - 2;
        A.last.coef[0] = -w.d.d11 * (w.hinv + w.alphae * denoe);
        A.last.col[1] = w.n - 1;
        A.last.coef[1] =
            w.dhdt + w.lambdaC * w.h +
            4.0 * w.d.d11 * (3.0 * w.alphae * w.h + 2.0 * w.betae) * denoe * w.hinv;
    }

    {
        StencilSet& A = tc.A12;
        A.first.ncols = 2;
        A.first.col[0] = 0;
        A.first.coef[0] = -w.lambdaR * w.h - w.d.d11 * (3 * w.hinv - 9.0 * w.alphaw * denow) + 4.0 * w.d.d12 * w.hinv;
        A.first.col[1] = 1;
        A.first.coef[1] = -(4.0 / 3.0) * w.d.d12 * w.hinv +
            (w.hinv / 3.0 - w.alphaw * denow) * w.d.d11;

        A.last.ncols = 2;
        A.last.col[0] = w.n - 2;
        A.last.coef[0] =
            -(4.0 / 3.0) * w.d.d12 * w.hinv +
            (w.hinv / 3.0 - w.alphae * denoe) * w.d.d11;
        A.last.col[1] = w.n - 1;
        A.last.coef[1] =
            -w.lambdaR * w.h + 4.0 * w.d.d12 * w.hinv -
            w.d.d11 * (3.0 * w.hinv - 9.0 * w.alphae * denoe);
    }

    {
        StencilSet& A = tc.A21;
        A.first.ncols = 2;
        A.first.col[0] = 0;
        A.first.coef[0] =
            -w.lambdaC * w.h +
            4.0 * w.d.d21 * w.hinv * (3.0 * w.alphaw * w.h - 2.0 * w.betaw) * denow;
        A.first.col[1] = 1;
        A.first.coef[1] =
            -4.0 * w.d.d21 * w.hinv * (w.alphaw * w.h - 2.0 * w.betaw) * denow;

        A.last.ncols = 2;
        A.last.col[0] = w.n - 2;
        A.last.coef[0] =
            -4.0 * w.d.d21 * w.hinv * (w.alphae * w.h + 2.0 * w.betae) * denoe;
        A.last.col[1] = w.n - 1;
        A.last.coef[1] =
            -w.lambdaC * w.h +
            4.0 * w.d.d21 * w.hinv * (3.0 * w.alphae * w.h + 2.0 * w.betae) * denoe;
    }

    {
        StencilSet& A = tc.A22;
        A.first.ncols = 2;
        A.first.col[0] = 0;
        A.first.coef[0] =
            w.dhdt + w.lambdaR * w.h + 4.0 * w.d.d22 * w.hinv +
            24.0 * w.betaw * w.d.d21 * denow * w.hinv;
        A.first.col[1] = 1;
        A.first.coef[1] =
            -(4.0 / 3.0) * w.d.d22 * w.hinv -
            (8.0 / 3.0) * w.betaw * w.d.d21 * denow * w.hinv;

        A.last.ncols = 2;
        A.last.col[0] = w.n - 2;
        A.last.coef[0] =
            -(4.0 / 3.0) * w.d.d22 * w.hinv +
            (8.0 / 3.0) * w.betae * w.d.d21 * denoe * w.hinv;
        A.last.col[1] = w.n - 1;
        A.last.coef[1] =
            w.dhdt + w.lambdaR * w.h + 4.0 * w.d.d22 * w.hinv -
            24.0 * w.betae * w.d.d21 * denoe * w.hinv;
    }
}

void applyConstantGradientBoundaryCoefficients(Coefficients& tc, const CoefficientWorkData& w) {
    const PetscReal denow = 1.0 / (3.0 * w.alphaw * w.h - 8.0 * w.betaw);
    const PetscReal denoe = 1.0 / (3.0 * w.alphae * w.h + 8.0 * w.betae);
    const PetscReal westDiffusionFactor = w.hinv + 9.0 * w.alphaw * denow;
    const PetscReal westNeighborFactor = w.hinv + w.alphaw * denow;
    const PetscReal eastDiffusionFactor = w.hinv + 9.0 * w.alphae * denoe;
    const PetscReal eastNeighborFactor = w.hinv + w.alphae * denoe;

    {
        StencilSet& A = tc.A11;
        A.first.ncols = 2;
        A.first.col[0] = 0;
        A.first.coef[0] =
            w.dhdt + w.lambdaC * w.h + w.d.d11 * westDiffusionFactor;
        A.first.col[1] = 1;
        A.first.coef[1] = -w.d.d11 * westNeighborFactor;

        A.last.ncols = 2;
        A.last.col[0] = w.n - 2;
        A.last.coef[0] = -w.d.d11 * eastNeighborFactor;
        A.last.col[1] = w.n - 1;
        A.last.coef[1] =
            w.dhdt + w.lambdaC * w.h + w.d.d11 * eastDiffusionFactor;
    }

    {
        StencilSet& A = tc.A12;
        A.first.ncols = 2;
        A.first.col[0] = 0;
        A.first.coef[0] = -w.lambdaR * w.h + w.d.d11 * westDiffusionFactor;
        A.first.col[1] = 1;
        A.first.coef[1] = -w.d.d11 * westNeighborFactor;

        A.last.ncols = 2;
        A.last.col[0] = w.n - 2;
        A.last.coef[0] = -w.d.d11 * eastNeighborFactor;
        A.last.col[1] = w.n - 1;
        A.last.coef[1] = -w.lambdaR * w.h + w.d.d11 * eastDiffusionFactor;
    }

    {
        StencilSet& A = tc.A21;
        A.first.ncols = 2;
        A.first.col[0] = 0;
        A.first.coef[0] = -w.lambdaC * w.h + w.d.d21 * westDiffusionFactor;
        A.first.col[1] = 1;
        A.first.coef[1] = -w.d.d21 * westNeighborFactor;

        A.last.ncols = 2;
        A.last.col[0] = w.n - 2;
        A.last.coef[0] = -w.d.d21 * eastNeighborFactor;
        A.last.col[1] = w.n - 1;
        A.last.coef[1] = -w.lambdaC * w.h + w.d.d21 * eastDiffusionFactor;
    }

    {
        StencilSet& A = tc.A22;
        A.first.ncols = 2;
        A.first.col[0] = 0;
        A.first.coef[0] =
            w.dhdt + w.lambdaR * w.h + w.d.d21 * westDiffusionFactor;
        A.first.col[1] = 1;
        A.first.coef[1] = -w.d.d21 * westNeighborFactor;

        A.last.ncols = 2;
        A.last.col[0] = w.n - 2;
        A.last.coef[0] = -w.d.d21 * eastNeighborFactor;
        A.last.col[1] = w.n - 1;
        A.last.coef[1] =
            w.dhdt + w.lambdaR * w.h + w.d.d21 * eastDiffusionFactor;
    }
}

void applyScaledPsiBoundaryCoefficients(Coefficients& tc, const CoefficientWorkData& w,
                                        const PetscReal sc) {
    const PetscReal k = 1.0 - sc;
    const PetscReal thirdHinv = w.hinv / 3.0;
    const PetscReal denow = 1.0 / (3.0 * w.alphaw * w.h - 8.0 * w.betaw);
    const PetscReal denoe = 1.0 / (3.0 * w.alphae * w.h + 8.0 * w.betae);

    const PetscReal bwP = -9.0 * w.betaw * denow;
    const PetscReal bwE = w.betaw * denow;
    const PetscReal dUwUP = (9.0 - 8.0 * sc * bwP) * thirdHinv;
    const PetscReal dUwUE = (-1.0 - 8.0 * sc * bwE) * thirdHinv;
    const PetscReal dUwVP = (-8.0 * sc * bwP) * thirdHinv;
    const PetscReal dUwVE = (-8.0 * sc * bwE) * thirdHinv;
    const PetscReal dVwUP = (-8.0 * k * bwP) * thirdHinv;
    const PetscReal dVwUE = (-8.0 * k * bwE) * thirdHinv;
    const PetscReal dVwVP = (9.0 - 8.0 * k * bwP) * thirdHinv;
    const PetscReal dVwVE = (-1.0 - 8.0 * k * bwE) * thirdHinv;

    const PetscReal beP = 9.0 * w.betae * denoe;
    const PetscReal beW = -w.betae * denoe;
    const PetscReal dUeUW = (1.0 + 8.0 * sc * beW) * thirdHinv;
    const PetscReal dUeUP = (-9.0 + 8.0 * sc * beP) * thirdHinv;
    const PetscReal dUeVW = (8.0 * sc * beW) * thirdHinv;
    const PetscReal dUeVP = (8.0 * sc * beP) * thirdHinv;
    const PetscReal dVeUW = (8.0 * k * beW) * thirdHinv;
    const PetscReal dVeUP = (8.0 * k * beP) * thirdHinv;
    const PetscReal dVeVW = (1.0 + 8.0 * k * beW) * thirdHinv;
    const PetscReal dVeVP = (-9.0 + 8.0 * k * beP) * thirdHinv;

    {
        StencilSet& A = tc.A11;
        A.first.ncols = 2;
        A.first.col[0] = 0;
        A.first.coef[0] =
            w.dhdt + w.lambdaC * w.h + w.d.d11 * (w.hinv + dUwUP) +
            w.d.d12 * dVwUP;
        A.first.col[1] = 1;
        A.first.coef[1] =
            w.d.d11 * (-w.hinv + dUwUE) + w.d.d12 * dVwUE;

        A.last.ncols = 2;
        A.last.col[0] = w.n - 2;
        A.last.coef[0] =
            w.d.d11 * (-w.hinv - dUeUW) - w.d.d12 * dVeUW;
        A.last.col[1] = w.n - 1;
        A.last.coef[1] =
            w.dhdt + w.lambdaC * w.h + w.d.d11 * (w.hinv - dUeUP) -
            w.d.d12 * dVeUP;
    }

    {
        StencilSet& A = tc.A12;
        A.first.ncols = 2;
        A.first.col[0] = 0;
        A.first.coef[0] =
            -w.lambdaR * w.h + w.d.d11 * dUwVP +
            w.d.d12 * (w.hinv + dVwVP);
        A.first.col[1] = 1;
        A.first.coef[1] =
            w.d.d11 * dUwVE + w.d.d12 * (-w.hinv + dVwVE);

        A.last.ncols = 2;
        A.last.col[0] = w.n - 2;
        A.last.coef[0] =
            -w.d.d11 * dUeVW + w.d.d12 * (-w.hinv - dVeVW);
        A.last.col[1] = w.n - 1;
        A.last.coef[1] =
            -w.lambdaR * w.h - w.d.d11 * dUeVP +
            w.d.d12 * (w.hinv - dVeVP);
    }

    {
        StencilSet& A = tc.A21;
        A.first.ncols = 2;
        A.first.col[0] = 0;
        A.first.coef[0] =
            -w.lambdaC * w.h + w.d.d21 * (w.hinv + dUwUP) +
            w.d.d22 * dVwUP;
        A.first.col[1] = 1;
        A.first.coef[1] =
            w.d.d21 * (-w.hinv + dUwUE) + w.d.d22 * dVwUE;

        A.last.ncols = 2;
        A.last.col[0] = w.n - 2;
        A.last.coef[0] =
            w.d.d21 * (-w.hinv - dUeUW) - w.d.d22 * dVeUW;
        A.last.col[1] = w.n - 1;
        A.last.coef[1] =
            -w.lambdaC * w.h + w.d.d21 * (w.hinv - dUeUP) -
            w.d.d22 * dVeUP;
    }

    {
        StencilSet& A = tc.A22;
        A.first.ncols = 2;
        A.first.col[0] = 0;
        A.first.coef[0] =
            w.dhdt + w.lambdaR * w.h + w.d.d21 * dUwVP +
            w.d.d22 * (w.hinv + dVwVP);
        A.first.col[1] = 1;
        A.first.coef[1] =
            w.d.d21 * dUwVE + w.d.d22 * (-w.hinv + dVwVE);

        A.last.ncols = 2;
        A.last.col[0] = w.n - 2;
        A.last.coef[0] =
            -w.d.d21 * dUeVW + w.d.d22 * (-w.hinv - dVeVW);
        A.last.col[1] = w.n - 1;
        A.last.coef[1] =
            w.dhdt + w.lambdaR * w.h - w.d.d21 * dUeVP +
            w.d.d22 * (w.hinv - dVeVP);
    }
}

RHSCoefficients computeZeroValueBoundaryRHS(const Grid1D& grid,
                                            const Constants& constants,
                                            const BoundarySet& boundaries,
                                            const PetscReal t) {
    const PetscReal h = controlVolumeSpacing(grid);
    const DiffusionCoefficients d = computeDiffusionCoefficients(constants);
    const BoundaryCondition& west = boundaries.west[0];
    const BoundaryCondition& east = boundaries.east[0];

    const PetscReal alphaw = west.alpha();
    const PetscReal betaw = west.beta();
    const PetscReal gammaw = west.gamma(t);
    const PetscReal alphae = east.alpha();
    const PetscReal betae = east.beta();
    const PetscReal gammae = east.gamma(t);

    const PetscReal denow = 8.0 * gammaw / (3.0 * alphaw * h - 8.0 * betaw);
    const PetscReal denoe = 8.0 * gammae / (3.0 * alphae * h + 8.0 * betae);

    return {
        .b1 = {
            .first = d.d11 * denow,
            .last = d.d11 * denoe,
        },
        .b2 = {
            .first = d.d21 * denow,
            .last = d.d21 * denoe,
        },
    };
}

RHSCoefficients computeConstantGradientBoundaryRHS(const Grid1D& grid,
                                                   const Constants& constants,
                                                   const BoundarySet& boundaries,
                                                   const PetscReal t) {
    const PetscReal h = controlVolumeSpacing(grid);
    const DiffusionCoefficients d = computeDiffusionCoefficients(constants);
    const BoundaryCondition& west = boundaries.west[0];
    const BoundaryCondition& east = boundaries.east[0];

    const PetscReal denow =
        8.0 * west.gamma(t) / (3.0 * west.alpha() * h - 8.0 * west.beta());
    const PetscReal denoe =
        8.0 * east.gamma(t) / (3.0 * east.alpha() * h + 8.0 * east.beta());

    return {
        .b1 = {
            .first = d.d11 * denow,
            .last = d.d11 * denoe,
        },
        .b2 = {
            .first = d.d21 * denow,
            .last = d.d21 * denoe,
        },
    };
}

RHSCoefficients computeScaledPsiBoundaryRHS(const Grid1D& grid,
                                            const Constants& constants,
                                            const BoundarySet& boundaries,
                                            const PetscReal t) {
    const PetscReal h = controlVolumeSpacing(grid);
    const PetscReal sc = constants.vBoundary.sc;
    const PetscReal k = 1.0 - sc;
    const DiffusionCoefficients d = computeDiffusionCoefficients(constants);
    const BoundaryCondition& west = boundaries.west[0];
    const BoundaryCondition& east = boundaries.east[0];

    const PetscReal denow =
        8.0 * west.gamma(t) / (3.0 * west.alpha() * h - 8.0 * west.beta());
    const PetscReal denoe =
        8.0 * east.gamma(t) / (3.0 * east.alpha() * h + 8.0 * east.beta());

    return {
        .b1 = {
            .first = (d.d11 * sc + d.d12 * k) * denow,
            .last = (d.d11 * sc + d.d12 * k) * denoe,
        },
        .b2 = {
            .first = (d.d21 * sc + d.d22 * k) * denow,
            .last = (d.d21 * sc + d.d22 * k) * denoe,
        },
    };
}

} // namespace

PetscReal controlVolumeSpacing(const Grid1D& grid) noexcept {
    return grid.nx > 0 ? grid.length / static_cast<PetscReal>(grid.nx) : 0.0;
}

DiffusionCoefficients computeDiffusionCoefficients(const Constants& constants) noexcept {
    const PetscReal alpha = constants.alpha;
    const PetscReal rho = constants.rho;
    const PetscReal theta = constants.theta;

    return {
        .d11 = (1.0 - alpha * theta) * (1.0 - alpha * (1.0 - theta)),
        .d12 = rho * theta * (1.0 - alpha * (1.0 - theta)),
        .d21 = alpha * (1.0 - theta) * (1.0 - alpha * theta),
        .d22 = alpha * rho * theta * (1.0 - theta),
    };
}

Coefficients computeCoefficients(const Grid1D& grid,
                                 const TimeConfig& time,
                                 const Constants& constants,
                                 const BoundarySet& boundaries) {
    Coefficients tc {};
    const CoefficientWorkData w = makeWorkData(grid, time, constants, boundaries);

    computeInteriorCoefficients(tc, w);

    switch (constants.vBoundary.type) {
    case VBoundaryCondition::ZeroValue:
        applyZeroValueBoundaryCoefficients(tc, w);
        break;
    case VBoundaryCondition::ConstantGradient:
        applyConstantGradientBoundaryCoefficients(tc, w);
        break;
    case VBoundaryCondition::ScaledPsi:
        applyScaledPsiBoundaryCoefficients(tc, w, constants.vBoundary.sc);
        break;
    }

    return tc;
}

RHSCoefficients computeBoundaryRHS(const Grid1D& grid,
                                   const Constants& constants,
                                   const BoundarySet& boundaries,
                                   const PetscReal t) {
    switch (constants.vBoundary.type) {
    case VBoundaryCondition::ZeroValue:
        return computeZeroValueBoundaryRHS(grid, constants, boundaries, t);
    case VBoundaryCondition::ConstantGradient:
        return computeConstantGradientBoundaryRHS(grid, constants, boundaries, t);
    case VBoundaryCondition::ScaledPsi:
        return computeScaledPsiBoundaryRHS(grid, constants, boundaries, t);
    }

    throw std::runtime_error("Unknown TSOM V boundary condition.");
}

DiscreteRHS buildBoundaryRHS(const Grid1D& grid, const RHSCoefficients& rhs) {
    const PetscInt n = grid.nx;
    return {
        .size = 2 * n,
        .entries = {
            {.row = 0, .value = rhs.b1.first},
            {.row = n - 1, .value = rhs.b1.last},
            {.row = n, .value = rhs.b2.first},
            {.row = 2 * n - 1, .value = rhs.b2.last},
        },
    };
}

} // namespace bgc::models::tsom
