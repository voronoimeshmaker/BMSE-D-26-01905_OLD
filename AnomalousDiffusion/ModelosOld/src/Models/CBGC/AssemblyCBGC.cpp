#include <bgclib/Models/BGC/AssemblyBGC.hpp>
#include <bgclib/SimConfig.hpp>
#include <bgclib/SimState.hpp>

#include <algorithm>
#include <array>

namespace bgc {

// ---------------------------------------------------------------------------
//  assembleMatrixBGC
//
//  Insere StencilCoefficients na matriz st.A e chama MatAssembly.
//  Os índices do estêncil interior são offsets relativos — soma i antes
//  de inserir.
//
//  O termo temporal h⁵/dt NÃO está incluído aqui — o chamador deve
//  adicionar via MatShift(st.A, h5/dt) após esta chamada.
// ---------------------------------------------------------------------------

PetscErrorCode assembleMatrixCBGC(const SimConfig&           cfg,
                                  SimState&                  st,
                                  const StencilCoefficients& sc) {
    PetscFunctionBeginUser;

    PetscCall(MatZeroEntries(st.A));

    PetscInt xstart, xm;
    PetscCall(DMDAGetCorners(st.dm, &xstart, nullptr, nullptr,
                             &xm, nullptr, nullptr));
    const PetscInt xend = xstart + xm;
    const PetscInt n    = cfg.nx;

    auto insertRow = [&](PetscInt row, const StencilRow& sr,
                         bool relativeCols = false) -> PetscErrorCode {
        std::array<PetscInt, 5> cols;
        for (int k = 0; k < sr.ncols; ++k)
            cols[static_cast<std::size_t>(k)] =
                relativeCols ? row + sr.col[static_cast<std::size_t>(k)]
                             :       sr.col[static_cast<std::size_t>(k)];
        PetscCall(MatSetValues(st.A, 1, &row, sr.ncols,
                               cols.data(), sr.coef.data(), INSERT_VALUES));
        return PETSC_SUCCESS;
    };

    if (0   >= xstart && 0   < xend) PetscCall(insertRow(0,   sc.vol0));
    if (1   >= xstart && 1   < xend) PetscCall(insertRow(1,   sc.vol1));
    if (n-2 >= xstart && n-2 < xend) PetscCall(insertRow(n-2, sc.volNm2));
    if (n-1 >= xstart && n-1 < xend) PetscCall(insertRow(n-1, sc.volNm1));

    const PetscInt iMin = std::max(xstart, PetscInt{2});
    const PetscInt iMax = std::min(xend,   n - 2);
    for (PetscInt i = iMin; i < iMax; ++i)
        PetscCall(insertRow(i, sc.interior, /*relativeCols=*/true));


   
    PetscCall(MatAssemblyBegin(st.A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd  (st.A, MAT_FINAL_ASSEMBLY));

    const PetscReal hdt = cfg.h / cfg.dt;
    PetscCall(MatShift(st.A, hdt));

    PetscFunctionReturn(PETSC_SUCCESS);
    
}

// ---------------------------------------------------------------------------
//  assembleRHSBGC
// ---------------------------------------------------------------------------

PetscErrorCode assembleRHSCBGC(const SimConfig&       cfg,
                               SimState&              st,
                               const RHSCoefficients& rhs) {
    PetscFunctionBeginUser;

    PetscCall(VecZeroEntries(st.b));

    PetscInt xstart, xm;
    PetscCall(DMDAGetCorners(st.dm, &xstart, nullptr, nullptr,
                             &xm, nullptr, nullptr));
    const PetscInt xend = xstart + xm;
    const PetscInt n    = cfg.nx;

    auto set = [&](PetscInt i, PetscReal v) -> PetscErrorCode {
        if (i >= xstart && i < xend)
            PetscCall(VecSetValue(st.b, i, v, INSERT_VALUES));
        return PETSC_SUCCESS;
    };

    PetscCall(set(0,   rhs.vol0));
    PetscCall(set(1,   rhs.vol1));
    PetscCall(set(n-2, rhs.volNm2));
    PetscCall(set(n-1, rhs.volNm1));

    PetscCall(VecAssemblyBegin(st.b));
    PetscCall(VecAssemblyEnd  (st.b));
    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace bgc