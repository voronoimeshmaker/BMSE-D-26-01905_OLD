#include <bgclib/Models/TSOMPsiV/Coeff.hpp>

#include <stdexcept>

// TSOMPsiV coefficient assembly is split by responsibility:
//
//   computeInteriorCoefficients  -> rows unaffected by V boundary mode
//   compute*BoundaryCoefficients -> first/last rows for each V boundary mode
//   compute*BoundaryRHS          -> source terms induced by each V boundary mode
//
// This separation mirrors the Maple derivations and makes it explicit that
// changing the V boundary condition can change both A and the RHS.

namespace bgc::models::tsompsiv {

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

struct WestDerivative {
    PetscReal psiP {0.0};
    PetscReal psiE {0.0};
    PetscReal vP {0.0};
    PetscReal vE {0.0};
};

struct EastDerivative {
    PetscReal psiW {0.0};
    PetscReal psiP {0.0};
    PetscReal vW {0.0};
    PetscReal vP {0.0};
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
        throw std::runtime_error("computeCoefficients(TSOMPsiV): invalid west Psi boundary.");
    }
    if (w.alphae == 0.0 && w.betae == 0.0) {
        throw std::runtime_error("computeCoefficients(TSOMPsiV): invalid east Psi boundary.");
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
        A.interior.coef[1] = w.dhdt + 2.0 * w.d.d11 * w.hinv;
        A.interior.col[2] = 1;
        A.interior.coef[2] = -w.d.d11 * w.hinv;
    }

    {
        StencilSet& A = tc.A12;
        A.interior.ncols = 3;
        A.interior.col[0] = -1;
        A.interior.coef[0] = -w.d.d12 * w.hinv;
        A.interior.col[1] = 0;
        A.interior.coef[1] = 2.0 * w.d.d12 * w.hinv;
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
        A.interior.coef[1] =
            w.dhdt + (w.lambdaC + w.lambdaR) * w.h + 2.0 * w.d.d22 * w.hinv;
        A.interior.col[2] = 1;
        A.interior.coef[2] = -w.d.d22 * w.hinv;
    }
}

WestDerivative makeWestDerivative(const CoefficientWorkData& w,
                                  const VBoundaryCondition mode,
                                  const PetscReal sc) {
    const PetscReal denow = 1.0 / (3.0 * w.alphaw * w.h - 8.0 * w.betaw);
    const PetscReal k = 1.0 - sc;

    WestDerivative d {
        .psiP = 9.0 * w.alphaw * denow,
        .psiE = -w.alphaw * denow,
    };

    switch (mode) {
    case VBoundaryCondition::ZeroValue:
        d.vP = 3.0 * w.hinv;
        d.vE = -(1.0 / 3.0) * w.hinv;
        break;
    case VBoundaryCondition::ConstantGradient:
        d.vP = 0.0;
        d.vE = 0.0;
        break;
    case VBoundaryCondition::ScaledPsi:
        d.vP = 3.0 * w.hinv;
        d.vE = -(1.0 / 3.0) * w.hinv;
        d.psiP = 24.0 * k * w.betaw * denow * w.hinv;
        d.psiE = -(8.0 / 3.0) * k * w.betaw * denow * w.hinv;
        break;
    }

    return d;
}

EastDerivative makeEastDerivative(const CoefficientWorkData& w,
                                  const VBoundaryCondition mode,
                                  const PetscReal sc) {
    const PetscReal denoe = 1.0 / (3.0 * w.alphae * w.h + 8.0 * w.betae);
    const PetscReal k = 1.0 - sc;

    EastDerivative d {
        .psiW = w.alphae * denoe,
        .psiP = -9.0 * w.alphae * denoe,
    };

    switch (mode) {
    case VBoundaryCondition::ZeroValue:
        d.vW = (1.0 / 3.0) * w.hinv;
        d.vP = -3.0 * w.hinv;
        break;
    case VBoundaryCondition::ConstantGradient:
        d.vW = 0.0;
        d.vP = 0.0;
        break;
    case VBoundaryCondition::ScaledPsi:
        d.vW = (1.0 / 3.0) * w.hinv;
        d.vP = -3.0 * w.hinv;
        d.psiW = -(8.0 / 3.0) * k * w.betae * denoe * w.hinv;
        d.psiP = 24.0 * k * w.betae * denoe * w.hinv;
        break;
    }

    return d;
}

void applyBoundaryCoefficients(Coefficients& tc,
                               const CoefficientWorkData& w,
                               const VBoundaryCondition mode,
                               const PetscReal sc) {
    const WestDerivative west = makeWestDerivative(w, mode, sc);
    const EastDerivative east = makeEastDerivative(w, mode, sc);

    {
        StencilSet& A = tc.A11;
        A.first.ncols = 2;
        A.first.col[0] = 0;
        A.first.coef[0] = w.dhdt + w.d.d11 * (w.hinv + west.psiP);
        A.first.col[1] = 1;
        A.first.coef[1] = w.d.d11 * (-w.hinv + west.psiE);

        A.last.ncols = 2;
        A.last.col[0] = w.n - 2;
        A.last.coef[0] = -w.d.d11 * (w.hinv + east.psiW);
        A.last.col[1] = w.n - 1;
        A.last.coef[1] = w.dhdt + w.d.d11 * (w.hinv - east.psiP);
    }

    {
        StencilSet& A = tc.A12;
        A.first.ncols = 2;
        A.first.col[0] = 0;
        A.first.coef[0] = w.d.d12 * (w.hinv + west.vP);
        A.first.col[1] = 1;
        A.first.coef[1] = w.d.d12 * (-w.hinv + west.vE);

        A.last.ncols = 2;
        A.last.col[0] = w.n - 2;
        A.last.coef[0] = -w.d.d12 * (w.hinv + east.vW);
        A.last.col[1] = w.n - 1;
        A.last.coef[1] = w.d.d12 * (w.hinv - east.vP);
    }

    {
        StencilSet& A = tc.A21;
        A.first.ncols = 2;
        A.first.col[0] = 0;
        A.first.coef[0] = -w.lambdaC * w.h + w.d.d21 * (w.hinv + west.psiP);
        A.first.col[1] = 1;
        A.first.coef[1] = w.d.d21 * (-w.hinv + west.psiE);

        A.last.ncols = 2;
        A.last.col[0] = w.n - 2;
        A.last.coef[0] = -w.d.d21 * (w.hinv + east.psiW);
        A.last.col[1] = w.n - 1;
        A.last.coef[1] = -w.lambdaC * w.h + w.d.d21 * (w.hinv - east.psiP);
    }

    {
        StencilSet& A = tc.A22;
        A.first.ncols = 2;
        A.first.col[0] = 0;
        A.first.coef[0] =
            w.dhdt + (w.lambdaC + w.lambdaR) * w.h + w.d.d22 * (w.hinv + west.vP);
        A.first.col[1] = 1;
        A.first.coef[1] = w.d.d22 * (-w.hinv + west.vE);

        A.last.ncols = 2;
        A.last.col[0] = w.n - 2;
        A.last.coef[0] = -w.d.d22 * (w.hinv + east.vW);
        A.last.col[1] = w.n - 1;
        A.last.coef[1] =
            w.dhdt + (w.lambdaC + w.lambdaR) * w.h + w.d.d22 * (w.hinv - east.vP);
    }
}

RHSCoefficients computeBoundaryRHSForMode(const Grid1D& grid,
                                          const Constants& constants,
                                          const BoundarySet& boundaries,
                                          const PetscReal t,
                                          const VBoundaryCondition mode) {
    const PetscReal h = controlVolumeSpacing(grid);
    const DiffusionCoefficients d = computeDiffusionCoefficients(constants);
    const BoundaryCondition& west = boundaries.west[0];
    const BoundaryCondition& east = boundaries.east[0];

    const PetscReal westPsiSource =
        -8.0 * west.gamma(t) / (3.0 * west.alpha() * h - 8.0 * west.beta());
    const PetscReal eastPsiSource =
        8.0 * east.gamma(t) / (3.0 * east.alpha() * h + 8.0 * east.beta());

    PetscReal westVSource = 0.0;
    PetscReal eastVSource = 0.0;
    if (mode == VBoundaryCondition::ScaledPsi) {
        const PetscReal k = 1.0 - constants.vBoundary.sc;
        westVSource = k * westPsiSource;
        eastVSource = k * eastPsiSource;
    }

    // Constant terms are moved from the matrix equation to the RHS.
    const PetscReal westPsiRHS = -(d.d11 * westPsiSource + d.d12 * westVSource);
    const PetscReal eastPsiRHS = d.d11 * eastPsiSource + d.d12 * eastVSource;
    const PetscReal westVRHS = -(d.d21 * westPsiSource + d.d22 * westVSource);
    const PetscReal eastVRHS = d.d21 * eastPsiSource + d.d22 * eastVSource;

    return {
        .b1 = {
            .first = westPsiRHS,
            .last = eastPsiRHS,
        },
        .b2 = {
            .first = westVRHS,
            .last = eastVRHS,
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
    const PetscReal wpsi2 = 1.0 - (alpha + rho) * theta;

    return {
        .d11 = 1.0 - alpha * theta,
        .d12 = wpsi2,
        .d21 = alpha * (1.0 - theta) * (1.0 - alpha * theta),
        .d22 = -alpha * (1.0 - theta) * wpsi2,
    };
}

Coefficients computeCoefficients(const Grid1D& grid,
                                 const TimeConfig& time,
                                 const Constants& constants,
                                 const BoundarySet& boundaries) {
    Coefficients tc {};
    const CoefficientWorkData w = makeWorkData(grid, time, constants, boundaries);

    computeInteriorCoefficients(tc, w);

    applyBoundaryCoefficients(tc, w, constants.vBoundary.type, constants.vBoundary.sc);

    return tc;
}

RHSCoefficients computeBoundaryRHS(const Grid1D& grid,
                                   const Constants& constants,
                                   const BoundarySet& boundaries,
                                   const PetscReal t) {
    return computeBoundaryRHSForMode(grid, constants, boundaries, t, constants.vBoundary.type);
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

} // namespace bgc::models::tsompsiv
