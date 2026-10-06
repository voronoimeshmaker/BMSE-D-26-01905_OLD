#include <iostream>
#include <format>
#include <cmath>

#include <petsc.h>

#include <bgclib/SimConfig.hpp>
#include <bgclib/SimState.hpp>
#include <bgclib/Core/Coefficients.hpp>
#include <bgclib/Models/BGC/CoeffBGC.hpp>
#include <bgclib/Models/BGC/AssemblyBGC.hpp>
#include <bgclib/Models/TBGC/CoeffTBGC.hpp>
#include <bgclib/Models/TBGC/AssemblyTBGC.hpp>

// ---------------------------------------------------------------------------
//  Parâmetros de referência — idênticos ao testCoefficients
// ---------------------------------------------------------------------------

static constexpr PetscInt  NX  = 8;
static constexpr PetscReal LX  = 1.0;
static constexpr PetscReal BV  = 0.01;
static constexpr PetscReal DT  = 1.0e-4;
static constexpr PetscReal H   = LX / NX;
static constexpr PetscReal TOL = 1.0e-10;

// ---------------------------------------------------------------------------
//  Contadores globais
// ---------------------------------------------------------------------------

static int total  = 0;
static int passed = 0;

// ---------------------------------------------------------------------------
//  Utilitários de diagnóstico — usam PetscPrintf para MPI safety
// ---------------------------------------------------------------------------

static void checkClose(PetscReal valor,
                       PetscReal esperado,
                       const std::string& descricao,
                       PetscReal tol = TOL) {
    const PetscReal diff = std::abs(valor - esperado);
    ++total;
    if (diff < tol) {
        ++passed;
        PetscPrintf(PETSC_COMM_WORLD, "  [OK]     %s  (%.6e)\n",
                    descricao.c_str(), (double)valor);
    } else {
        PetscPrintf(PETSC_COMM_WORLD,
                    "  [FALHOU] %s  obtido=%.6e  esperado=%.6e  diff=%.3e\n",
                    descricao.c_str(),
                    (double)valor, (double)esperado, (double)diff);
    }
}

static void secao(const std::string& nome) {
    PetscPrintf(PETSC_COMM_WORLD, "\n%s\n%s\n",
                nome.c_str(), std::string(nome.size(), '-').c_str());
}

// ---------------------------------------------------------------------------
//  Helpers para leitura de entradas de matriz e vetor
// ---------------------------------------------------------------------------

static PetscReal matGet(Mat A, PetscInt row, PetscInt col) {
    PetscScalar val = 0.0;
    MatGetValues(A, 1, &row, 1, &col, &val);
    return static_cast<PetscReal>(val);
}

static PetscReal vecGet(Vec v, PetscInt i) {
    PetscScalar val = 0.0;
    VecGetValues(v, 1, &i, &val);
    return static_cast<PetscReal>(val);
}

// ---------------------------------------------------------------------------
//  Helper — SimConfig mínimo com Neumann homogêneo
// ---------------------------------------------------------------------------

static bgc::SimConfig makeConfig() {
    bgc::SimConfig cfg;
    cfg.nx = NX;  cfg.lx = LX;  cfg.h = H;
    cfg.bv = BV;  cfg.dt = DT;
    for (int k = 0; k < 2; ++k) {
        cfg.bcWest[k] = bgc::BoundaryCondition::neumann(0.0);
        cfg.bcEast[k] = bgc::BoundaryCondition::neumann(0.0);
    }
    return cfg;
}

// ---------------------------------------------------------------------------
//  Testes — todas as funções retornam PetscErrorCode para poder usar PetscCall
// ---------------------------------------------------------------------------

static PetscErrorCode testarAssemblyMatrizBGC() {
    PetscFunctionBeginUser;

    secao("BGC — assembleMatrixBGC");

    const auto cfg = makeConfig();
    bgc::SimState st;
    PetscCall(bgc::createStateBGC(cfg, st));

    const auto sc = bgc::computeCoefficientsBGC(cfg);
    PetscCall(bgc::assembleMatrixBGC(cfg, st, sc));

    // Coeficientes do vol0
    checkClose(matGet(st.A, 0, 0), sc.vol0.coef[0], "A[0,0] == vol0.coef[0] (ap)");
    checkClose(matGet(st.A, 0, 1), sc.vol0.coef[1], "A[0,1] == vol0.coef[1] (ae)");
    checkClose(matGet(st.A, 0, 2), sc.vol0.coef[2], "A[0,2] == vol0.coef[2] (aee)");
    checkClose(matGet(st.A, 0, 3), sc.vol0.coef[3], "A[0,3] == vol0.coef[3] (aeee)");
    checkClose(matGet(st.A, 0, 4), 0.0,             "A[0,4] == 0.0 (fora do estêncil)");

    // Estêncil interior (i=4)
    const PetscInt i = 4;
    checkClose(matGet(st.A, i, i-2), sc.interior.coef[0], "A[4,2] == interior.coef[0] (aww)");
    checkClose(matGet(st.A, i, i-1), sc.interior.coef[1], "A[4,3] == interior.coef[1] (aw)");
    checkClose(matGet(st.A, i, i  ), sc.interior.coef[2], "A[4,4] == interior.coef[2] (ap)");
    checkClose(matGet(st.A, i, i+1), sc.interior.coef[3], "A[4,5] == interior.coef[3] (ae)");
    checkClose(matGet(st.A, i, i+2), sc.interior.coef[4], "A[4,6] == interior.coef[4] (aee)");

    // volNm1
    const PetscInt nm1 = NX - 1;
    checkClose(matGet(st.A, nm1, nm1  ), sc.volNm1.coef[3], "A[n-1,n-1] == volNm1.coef[3] (ap)");
    checkClose(matGet(st.A, nm1, nm1-1), sc.volNm1.coef[2], "A[n-1,n-2] == volNm1.coef[2] (aw)");

    // MatShift
    const PetscReal h5dt   = H*H*H*H*H / DT;
    const PetscReal ap_pre = matGet(st.A, i, i);
    PetscCall(MatShift(st.A, h5dt));
    checkClose(matGet(st.A, i, i), ap_pre + h5dt,
               "após MatShift: A[4,4] == ap + h5/dt");

    PetscCall(bgc::destroyState(st));
    PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode testarAssemblyRHSBGC() {
    PetscFunctionBeginUser;

    secao("BGC — assembleRHSBGC  (Neumann homogêneo — rhs == 0)");

    const auto cfg = makeConfig();
    bgc::SimState st;
    PetscCall(bgc::createStateBGC(cfg, st));

    const auto rhs = bgc::computeRHSBGC(cfg, 0.0);
    PetscCall(bgc::assembleRHSBGC(cfg, st, rhs));

    for (PetscInt j = 0; j < NX; ++j)
        checkClose(vecGet(st.b, j), 0.0,
                   std::format("b[{}] == 0.0", (int)j));

    PetscCall(bgc::destroyState(st));
    PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode testarAssemblyRHSBGCDirichlet() {
    PetscFunctionBeginUser;

    secao("BGC — assembleRHSBGC  (Dirichlet g1w=1.5)");

    bgc::SimConfig cfg = makeConfig();
    cfg.bcWest[0] = bgc::BoundaryCondition::dirichlet(1.5);
    cfg.bcEast[0] = bgc::BoundaryCondition::dirichlet(0.0);

    bgc::SimState st;
    PetscCall(bgc::createStateBGC(cfg, st));

    const auto rhs = bgc::computeRHSBGC(cfg, 0.0);
    PetscCall(bgc::assembleRHSBGC(cfg, st, rhs));

    // Volumes interiores devem ser zero
    for (PetscInt j = 2; j <= NX-3; ++j)
        checkClose(vecGet(st.b, j), 0.0,
                   std::format("b[{}] == 0.0 (interior)", (int)j));

    // Volumes de fronteira devem bater com computeRHSBGC
    checkClose(vecGet(st.b, 0),    rhs.vol0,   "b[0]   == rhs.vol0");
    checkClose(vecGet(st.b, 1),    rhs.vol1,   "b[1]   == rhs.vol1");
    checkClose(vecGet(st.b, NX-2), rhs.volNm2, "b[n-2] == rhs.volNm2");
    checkClose(vecGet(st.b, NX-1), rhs.volNm1, "b[n-1] == rhs.volNm1");

    PetscCall(bgc::destroyState(st));
    PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode testarAssemblyMatrizTBGC() {
    PetscFunctionBeginUser;

    secao("TBGC — assembleMatrixTBGC");

    const auto cfg = makeConfig();
    bgc::SimState st;
    PetscCall(bgc::createStateTBGC(cfg, st));

    const auto tc = bgc::computeCoefficientsTBGC(cfg);
    PetscCall(bgc::assembleMatrixTBGC(cfg, st, tc));

    // A11 — vol0 e interior
    checkClose(matGet(st.A11, 0, 0), tc.A11.vol0.coef[0],
               "A11[0,0] == vol0.coef[0] (ap)");
    const PetscInt i = 4;
    checkClose(matGet(st.A11, i, i), tc.A11.interior.coef[0],
               "A11[4,4] == h/dt (interior diagonal)");

    // A12 — estêncil interior
    checkClose(matGet(st.A12, i, i-1), tc.A12.interior.coef[0],
               "A12[4,3] == interior.coef[0] (aw)");
    checkClose(matGet(st.A12, i, i  ), tc.A12.interior.coef[1],
               "A12[4,4] == interior.coef[1] (ap)");
    checkClose(matGet(st.A12, i, i+1), tc.A12.interior.coef[2],
               "A12[4,5] == interior.coef[2] (ae)");

    // A21 — estêncil interior de cinco pontos
    checkClose(matGet(st.A21, i, i-2), tc.A21.interior.coef[0],
               "A21[4,2] == interior.coef[0] (aww)");
    checkClose(matGet(st.A21, i, i  ), tc.A21.interior.coef[2],
               "A21[4,4] == interior.coef[2] (ap)");
    checkClose(matGet(st.A21, i, i+2), tc.A21.interior.coef[4],
               "A21[4,6] == interior.coef[4] (aee)");

    // A22 — identidade
    for (PetscInt j = 0; j < NX; ++j)
        checkClose(matGet(st.A22, j, j), 1.0,
                   std::format("A22[{0},{0}] == 1.0 (identidade)", (int)j));

    PetscCall(bgc::destroyState(st));
    PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode testarAssemblyRHSTBGC() {
    PetscFunctionBeginUser;

    secao("TBGC — assembleRHSTBGC  (Neumann homogêneo — rhs == 0)");

    const auto cfg = makeConfig();
    bgc::SimState st;
    PetscCall(bgc::createStateTBGC(cfg, st));

    const auto rhs = bgc::computeRHSTBGC(cfg, 0.0);
    PetscCall(bgc::assembleRHSTBGC(cfg, st, rhs));

    for (PetscInt j = 0; j < NX; ++j) {
        checkClose(vecGet(st.b1, j), 0.0,
                   std::format("b1[{}] == 0.0", (int)j));
        checkClose(vecGet(st.b2, j), 0.0,
                   std::format("b2[{}] == 0.0", (int)j));
    }

    PetscCall(bgc::destroyState(st));
    PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------------------------------------------
//  main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {

    PetscInitialize(&argc, &argv, nullptr,
                    "testAssembly — verificação da montagem PETSc");

    PetscPrintf(PETSC_COMM_WORLD, "\n%s\n",
                std::string(60, '=').c_str());
    PetscPrintf(PETSC_COMM_WORLD,
                "  testAssembly  —  verificação da montagem PETSc\n");
    PetscPrintf(PETSC_COMM_WORLD,
                "  nx=%d  h=%.4f  bv=%.4f  dt=%.2e\n",
                (int)NX, (double)H, (double)BV, (double)DT);
    PetscPrintf(PETSC_COMM_WORLD, "%s\n",
                std::string(60, '=').c_str());

    PetscCall(testarAssemblyMatrizBGC());
    PetscCall(testarAssemblyRHSBGC());
    PetscCall(testarAssemblyRHSBGCDirichlet());
    PetscCall(testarAssemblyMatrizTBGC());
    PetscCall(testarAssemblyRHSTBGC());

    PetscPrintf(PETSC_COMM_WORLD, "\n%s\n",
                std::string(60, '-').c_str());
    PetscPrintf(PETSC_COMM_WORLD, "  Resultado: %d/%d testes passaram\n",
                passed, total);
    PetscPrintf(PETSC_COMM_WORLD, "%s\n\n",
                std::string(60, '-').c_str());

    PetscFinalize();
    return (passed == total) ? 0 : 1;
}