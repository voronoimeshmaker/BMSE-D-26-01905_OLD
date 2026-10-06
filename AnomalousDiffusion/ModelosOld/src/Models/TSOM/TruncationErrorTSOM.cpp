#include <bgclib/Models/TSOM/TruncationErrorTSOM.hpp>
#include <bgclib/Models/TSOM/CoeffTSOM.hpp>
#include <bgclib/SimConfig.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace bgc {
namespace {

static PetscReal applyStencilRow(const StencilRow&              sr,
                                 const std::vector<PetscReal>& field,
                                 PetscInt                      i,
                                 bool                          relative) {
    PetscReal result = 0.0;

    const auto ncols = static_cast<std::size_t>(sr.ncols);
    for (std::size_t k = 0; k < ncols; ++k) {
        const PetscInt col = sr.col[k];
        const PetscInt j   = relative ? i + col : col;
        result += sr.coef[k] * field[static_cast<std::size_t>(j)];
    }
    return result;
}

} // namespace

PetscErrorCode computeTruncationErrorTSOM(const SimConfig&            cfg,
                                          const FieldFn&              exactUFn,
                                          const FieldFn&              exactVFn,
                                          const FieldFn&              sourceUFn,
                                          const FieldFn&              sourceVFn,
                                          PetscReal                   tPrev,
                                          PetscReal                   tNext,
                                          TSOMTruncationError&        te,
                                          std::vector<PetscReal>*     tauU,
                                          std::vector<PetscReal>*     tauV) {
    PetscFunctionBeginUser;

    const TSOMCoefficients    tc  = computeCoefficientsTSOM(cfg);
    const TSOMRHSCoefficients rhs = computeRHSTSOM(cfg, tNext);

    const PetscInt  n    = cfg.nx;
    const PetscReal h    = cfg.h;
    const PetscReal hinv = 1.0 / h;
    const PetscReal dhdt = h / cfg.dt;

    PetscInt xstart = 0;
    PetscInt xend   = n;

#ifdef BGC_HAS_DM
    {
        PetscInt xm = 0;
        PetscCall(DMDAGetCorners(cfg.dm, &xstart, nullptr, nullptr,
                                 &xm,    nullptr, nullptr));
        xend = xstart + xm;
    }
#endif

    const auto N = static_cast<std::size_t>(n);
    std::vector<PetscReal> uNext(N), vNext(N), uPrev(N), vPrev(N);

    for (PetscInt i = 0; i < n; ++i) {
        const PetscReal x   = cfg.xCenter(i);
        const auto      idx = static_cast<std::size_t>(i);
        uNext[idx] = exactUFn(cfg, x, tNext);
        vNext[idx] = exactVFn(cfg, x, tNext);
        uPrev[idx] = exactUFn(cfg, x, tPrev);
        vPrev[idx] = exactVFn(cfg, x, tPrev);
    }

    if (tauU != nullptr) { tauU->assign(N, 0.0); }
    if (tauV != nullptr) { tauV->assign(N, 0.0); }

    PetscReal localSumSqU = 0.0;
    PetscReal localSumSqV = 0.0;
    PetscReal localMaxU   = 0.0;
    PetscReal localMaxV   = 0.0;

    for (PetscInt i = xstart; i < xend; ++i) {
        const auto      idx = static_cast<std::size_t>(i);
        const PetscReal x   = cfg.xCenter(i);

        const bool isFirst = (i == 0);
        const bool isLast  = (i == n - 1);

        PetscReal lhsU = 0.0;
        PetscReal lhsV = 0.0;

        if (isFirst) {
            lhsU = applyStencilRow(tc.A11.first, uNext, i, false)
                 + applyStencilRow(tc.A12.first, vNext, i, false);
            lhsV = applyStencilRow(tc.A21.first, uNext, i, false)
                 + applyStencilRow(tc.A22.first, vNext, i, false);
        } else if (isLast) {
            lhsU = applyStencilRow(tc.A11.last, uNext, i, false)
                 + applyStencilRow(tc.A12.last, vNext, i, false);
            lhsV = applyStencilRow(tc.A21.last, uNext, i, false)
                 + applyStencilRow(tc.A22.last, vNext, i, false);
        } else {
            lhsU = applyStencilRow(tc.A11.interior, uNext, i, true)
                 + applyStencilRow(tc.A12.interior, vNext, i, true);
            lhsV = applyStencilRow(tc.A21.interior, uNext, i, true)
                 + applyStencilRow(tc.A22.interior, vNext, i, true);
        }

        PetscReal rhsU = dhdt * uPrev[idx] + h * sourceUFn(cfg, x, tNext);
        PetscReal rhsV = dhdt * vPrev[idx] + h * sourceVFn(cfg, x, tNext);

        if (isFirst) {
            rhsU += rhs.b1.first;
            rhsV += rhs.b2.first;
        } else if (isLast) {
            rhsU += rhs.b1.last;
            rhsV += rhs.b2.last;
        }

        const PetscReal tauUVal = (lhsU - rhsU) * hinv;
        const PetscReal tauVVal = (lhsV - rhsV) * hinv;

        if (tauU != nullptr) { (*tauU)[idx] = tauUVal; }
        if (tauV != nullptr) { (*tauV)[idx] = tauVVal; }

        localSumSqU += tauUVal * tauUVal * h;
        localSumSqV += tauVVal * tauVVal * h;
        localMaxU    = std::max(localMaxU, std::abs(tauUVal));
        localMaxV    = std::max(localMaxV, std::abs(tauVVal));
    }

    PetscReal globalSumSqU = 0.0;
    PetscReal globalSumSqV = 0.0;
    PetscReal globalMaxU   = 0.0;
    PetscReal globalMaxV   = 0.0;

    PetscCallMPI(MPIU_Allreduce(&localSumSqU, &globalSumSqU,
                                1, MPIU_REAL, MPIU_SUM, PETSC_COMM_WORLD));
    PetscCallMPI(MPIU_Allreduce(&localSumSqV, &globalSumSqV,
                                1, MPIU_REAL, MPIU_SUM, PETSC_COMM_WORLD));
    PetscCallMPI(MPIU_Allreduce(&localMaxU, &globalMaxU,
                                1, MPIU_REAL, MPIU_MAX, PETSC_COMM_WORLD));
    PetscCallMPI(MPIU_Allreduce(&localMaxV, &globalMaxV,
                                1, MPIU_REAL, MPIU_MAX, PETSC_COMM_WORLD));

    te.l2U   = std::sqrt(globalSumSqU);
    te.l2V   = std::sqrt(globalSumSqV);
    te.linfU = globalMaxU;
    te.linfV = globalMaxV;

    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace bgc
