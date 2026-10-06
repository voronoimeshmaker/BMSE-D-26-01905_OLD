#include <bgclib/Models/TSOM/AssemblyTSOM.hpp>
#include <bgclib/SimConfig.hpp>
#include <bgclib/SimState.hpp>

#include <algorithm>
#include <array>

namespace bgc {
namespace {

static PetscErrorCode insertRow(Mat               M,
                                PetscInt          row,
                                const StencilRow& sr,
                                bool              relativeCols = false) {
    PetscFunctionBeginUser;
    std::array<PetscInt, 5> cols {};
    for (size_t k = 0; k < static_cast<size_t>(sr.ncols); ++k) {
        cols[k] = relativeCols ? row + sr.col[k] : sr.col[k];
    }
    PetscCall(MatSetValues(M,
                           1,
                           &row,
                           sr.ncols,
                           cols.data(),
                           sr.coef.data(),
                           INSERT_VALUES));
    PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode assembleBlock(Mat                          M,
                                    const TSOMBlockCoefficients& sc,
                                    PetscInt                     xstart,
                                    PetscInt                     xend,
                                    PetscInt                     n) {
    PetscFunctionBeginUser;

    if (0 >= xstart && 0 < xend) {
        PetscCall(insertRow(M, 0, sc.first));
    }
    if (n - 1 >= xstart && n - 1 < xend) {
        PetscCall(insertRow(M, n - 1, sc.last));
    }

    const PetscInt iMin = std::max(xstart, PetscInt{1});
    const PetscInt iMax = std::min(xend, n - 1);
    for (PetscInt i = iMin; i < iMax; ++i) {
        PetscCall(insertRow(M, i, sc.interior, true));
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

PetscErrorCode assembleMatrixTSOM(const SimConfig&        cfg,
                                  SimState&               st,
                                  const TSOMCoefficients& tc) {
    PetscFunctionBeginUser;

    PetscInt xstart {0};
    PetscInt xm {0};
    PetscCall(DMDAGetCorners(st.dm, &xstart, nullptr, nullptr, &xm, nullptr, nullptr));
    const PetscInt xend = xstart + xm;
    const PetscInt n    = cfg.nx;

    PetscCall(MatZeroEntries(st.A11));
    PetscCall(assembleBlock(st.A11, tc.A11, xstart, xend, n));
    PetscCall(MatAssemblyBegin(st.A11, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(st.A11, MAT_FINAL_ASSEMBLY));

    PetscCall(MatZeroEntries(st.A12));
    PetscCall(assembleBlock(st.A12, tc.A12, xstart, xend, n));
    PetscCall(MatAssemblyBegin(st.A12, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(st.A12, MAT_FINAL_ASSEMBLY));

    PetscCall(MatZeroEntries(st.A21));
    PetscCall(assembleBlock(st.A21, tc.A21, xstart, xend, n));
    PetscCall(MatAssemblyBegin(st.A21, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(st.A21, MAT_FINAL_ASSEMBLY));

    PetscCall(MatZeroEntries(st.A22));
    PetscCall(assembleBlock(st.A22, tc.A22, xstart, xend, n));
    PetscCall(MatAssemblyBegin(st.A22, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(st.A22, MAT_FINAL_ASSEMBLY));

    PetscCall(MatAssemblyBegin(st.A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(st.A, MAT_FINAL_ASSEMBLY));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode assembleRHSTSOM(const SimConfig&           cfg,
                               SimState&                  st,
                               const TSOMRHSCoefficients& rhs) {
    PetscFunctionBeginUser;

    PetscCall(VecZeroEntries(st.b1));
    PetscCall(VecZeroEntries(st.b2));

    PetscInt xstart {0};
    PetscInt xm {0};
    PetscCall(DMDAGetCorners(st.dm, &xstart, nullptr, nullptr, &xm, nullptr, nullptr));
    const PetscInt xend = xstart + xm;
    const PetscInt n    = cfg.nx;

    auto setVal = [&](Vec v, PetscInt i, PetscReal val) -> PetscErrorCode {
        if (i >= xstart && i < xend) {
            PetscCall(VecSetValue(v, i, val, INSERT_VALUES));
        }
        return PETSC_SUCCESS;
    };

    PetscCall(setVal(st.b1, 0,     rhs.b1.first));
    PetscCall(setVal(st.b1, n - 1, rhs.b1.last));

    PetscCall(setVal(st.b2, 0,     rhs.b2.first));
    PetscCall(setVal(st.b2, n - 1, rhs.b2.last));

    PetscCall(VecAssemblyBegin(st.b1));
    PetscCall(VecAssemblyEnd(st.b1));
    PetscCall(VecAssemblyBegin(st.b2));
    PetscCall(VecAssemblyEnd(st.b2));
    PetscCall(VecAssemblyBegin(st.b));
    PetscCall(VecAssemblyEnd(st.b));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode assembleFullRHSTSOM(const SimConfig&           cfg,
                                   SimState&                  st,
                                   const TSOMRHSCoefficients& rhs,
                                   PetscReal                  t,
                                   const FieldFn&             sourceUFn,
                                   const FieldFn&             sourceVFn) {
    PetscFunctionBeginUser;

    PetscCheck(st.phi_0 && st.mu_0,
               PETSC_COMM_WORLD,
               PETSC_ERR_ARG_NULL,
               "assembleFullRHSTSOM requires phi_0 and mu_0 to be allocated.");

    PetscCall(VecZeroEntries(st.b1));
    PetscCall(VecZeroEntries(st.b2));

    PetscInt xs {0};
    PetscInt xe {0};
    PetscCall(VecGetOwnershipRange(st.b1, &xs, &xe));
    const PetscInt localSize = xe - xs;

    const PetscScalar* u0 = nullptr;
    const PetscScalar* v0 = nullptr;
    PetscCall(VecGetArrayRead(st.phi_0, &u0));
    PetscCall(VecGetArrayRead(st.mu_0,  &v0));

    const PetscReal mass = cfg.h / cfg.dt;

    for (PetscInt il = 0; il < localSize; ++il) {
        const PetscInt  i    = xs + il;
        const PetscReal x    = cfg.xCenter(i);
        PetscReal rhsU = mass * static_cast<PetscReal>(u0[il])
                       + cfg.h * sourceUFn(cfg, x, t);
        PetscReal rhsV = mass * static_cast<PetscReal>(v0[il])
                       + cfg.h * sourceVFn(cfg, x, t);

        if (i == 0) {
            rhsU += rhs.b1.first;
            rhsV += rhs.b2.first;
        } else if (i == cfg.nx - 1) {
            rhsU += rhs.b1.last;
            rhsV += rhs.b2.last;
        }

        PetscCall(VecSetValue(st.b1, i, rhsU, INSERT_VALUES));
        PetscCall(VecSetValue(st.b2, i, rhsV, INSERT_VALUES));
    }

    PetscCall(VecRestoreArrayRead(st.phi_0, &u0));
    PetscCall(VecRestoreArrayRead(st.mu_0,  &v0));

    PetscCall(VecAssemblyBegin(st.b1));
    PetscCall(VecAssemblyEnd(st.b1));
    PetscCall(VecAssemblyBegin(st.b2));
    PetscCall(VecAssemblyEnd(st.b2));
    PetscCall(VecAssemblyBegin(st.b));
    PetscCall(VecAssemblyEnd(st.b));

    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace bgc
