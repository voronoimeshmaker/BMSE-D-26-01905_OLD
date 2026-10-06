#include <bgclib/Models/TBGC/AssemblyTBGC.hpp>
#include <bgclib/SimConfig.hpp>
#include <bgclib/SimState.hpp>
#include <bgclib/Core/Coefficients.hpp>

#include <algorithm>
#include <array>

namespace bgc {

// ---------------------------------------------------------------------------
//  Helper interno  —  insere uma StencilRow na matriz M
//
//  relativeCols = true  : col[k] é offset relativo; soma row antes de inserir
//  relativeCols = false : col[k] é índice global absoluto
// ---------------------------------------------------------------------------

static PetscErrorCode insertRow(    Mat                    M,
                                    PetscInt               row,
                                    const StencilRow&      sr,
                                    bool                   relativeCols = false) {
    PetscFunctionBeginUser;
    std::array<PetscInt, 5> cols;
    for (size_t k = 0; k < static_cast<size_t>(sr.ncols); ++k)
        cols[k] = relativeCols ? row + sr.col[k] : sr.col[k];
    PetscCall(MatSetValues(M, 1, &row, sr.ncols,
                           cols.data(), sr.coef.data(), INSERT_VALUES));
    PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------------------------------------------
//  Helper interno  —  monta um bloco inteiro a partir de StencilCoefficients
// ---------------------------------------------------------------------------

static PetscErrorCode assembleBlock(Mat                        M,
                                     const StencilCoefficients& sc,
                                     PetscInt                   xstart,
                                     PetscInt                   xend,
                                     PetscInt                   n) {
    PetscFunctionBeginUser;

    if (0   >= xstart && 0   < xend) PetscCall(insertRow(M, 0,   sc.vol0));
    if (1   >= xstart && 1   < xend) PetscCall(insertRow(M, 1,   sc.vol1));
    if (n-2 >= xstart && n-2 < xend) PetscCall(insertRow(M, n-2, sc.volNm2));
    if (n-1 >= xstart && n-1 < xend) PetscCall(insertRow(M, n-1, sc.volNm1));

    const PetscInt iMin = std::max(xstart, PetscInt{2});
    const PetscInt iMax = std::min(xend,   n - 2);
    for (PetscInt i = iMin; i < iMax; ++i)
        PetscCall(insertRow(M, i, sc.interior, /*relativeCols=*/true));

    PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------------------------------------------
//  assembleMatrixTBGC
// ---------------------------------------------------------------------------

PetscErrorCode assembleMatrixTBGC(const SimConfig&        cfg,
                                   SimState&               st,
                                   const TBGCCoefficients& tc) {
    PetscFunctionBeginUser;

    PetscInt xstart, xm;
    PetscCall(DMDAGetCorners(st.dm, &xstart, nullptr, nullptr,
                             &xm, nullptr, nullptr));
    const PetscInt xend = xstart + xm;
    const PetscInt n    = cfg.nx;

    // -----------------------------------------------------------------------
    //  Bloco A11
    // -----------------------------------------------------------------------
    PetscCall(MatZeroEntries(st.A11));
    PetscCall(assembleBlock(st.A11, tc.A11, xstart, xend, n));
    PetscCall(MatAssemblyBegin(st.A11, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd  (st.A11, MAT_FINAL_ASSEMBLY));

    // -----------------------------------------------------------------------
    //  Bloco A12
    // -----------------------------------------------------------------------
    PetscCall(MatZeroEntries(st.A12));
    PetscCall(assembleBlock(st.A12, tc.A12, xstart, xend, n));
    PetscCall(MatAssemblyBegin(st.A12, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd  (st.A12, MAT_FINAL_ASSEMBLY));

    // -----------------------------------------------------------------------
    //  Bloco A21
    // -----------------------------------------------------------------------
    PetscCall(MatZeroEntries(st.A21));
    PetscCall(assembleBlock(st.A21, tc.A21, xstart, xend, n));
    PetscCall(MatAssemblyBegin(st.A21, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd  (st.A21, MAT_FINAL_ASSEMBLY));

    // -----------------------------------------------------------------------
    //  Bloco A22  —  identidade pura
    //  Não passa por TBGCCoefficients — montada diretamente aqui.
    // -----------------------------------------------------------------------
    PetscCall(MatZeroEntries(st.A22));
    for (PetscInt i = xstart; i < xend; ++i) {
        const PetscReal one = 1.0;
        PetscCall(MatSetValues(st.A22, 1, &i, 1, &i, &one, INSERT_VALUES));
    }
    PetscCall(MatAssemblyBegin(st.A22, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd  (st.A22, MAT_FINAL_ASSEMBLY));

    PetscCall(MatAssemblyBegin(st.A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd  (st.A, MAT_FINAL_ASSEMBLY));

    PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------------------------------------------
//  assembleRHSTBGC
// ---------------------------------------------------------------------------

PetscErrorCode assembleRHSTBGC(const SimConfig&           cfg,
                                SimState&                  st,
                                const TBGCRHSCoefficients& rhs) {
    PetscFunctionBeginUser;

    PetscCall(VecZeroEntries(st.b1));
    PetscCall(VecZeroEntries(st.b2));

    PetscInt xstart, xm;
    PetscCall(DMDAGetCorners(st.dm, &xstart, nullptr, nullptr,
                             &xm, nullptr, nullptr));
    const PetscInt xend = xstart + xm;
    const PetscInt n    = cfg.nx;

    // Lambda para inserir um valor num vetor se o índice pertence ao processo
    auto setVal = [&](Vec v, PetscInt i, PetscReal val) -> PetscErrorCode {
        if (i >= xstart && i < xend)
            PetscCall(VecSetValue(v, i, val, INSERT_VALUES));
        return PETSC_SUCCESS;
    };

    // b1  —  equação de phi
    PetscCall(setVal(st.b1, 0,   rhs.b1.vol0));
    PetscCall(setVal(st.b1, 1,   rhs.b1.vol1));
    PetscCall(setVal(st.b1, n-2, rhs.b1.volNm2));
    PetscCall(setVal(st.b1, n-1, rhs.b1.volNm1));

    // b2  —  equação de mu
    PetscCall(setVal(st.b2, 0,   rhs.b2.vol0));
    PetscCall(setVal(st.b2, 1,   rhs.b2.vol1));
    PetscCall(setVal(st.b2, n-2, rhs.b2.volNm2));
    PetscCall(setVal(st.b2, n-1, rhs.b2.volNm1));

    PetscCall(VecAssemblyBegin(st.b1)); PetscCall(VecAssemblyEnd(st.b1));
    PetscCall(VecAssemblyBegin(st.b2)); PetscCall(VecAssemblyEnd(st.b2));

    // b é o vetor monolítico pai de b1 e b2 — sincronizar após os sub-vetores
    PetscCall(VecAssemblyBegin(st.b));  PetscCall(VecAssemblyEnd(st.b));

    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace bgc