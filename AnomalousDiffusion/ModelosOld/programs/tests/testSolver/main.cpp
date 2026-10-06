#include <cmath>
#include <format>

#include <petsc.h>

#include <bgclib/SimConfig.hpp>
#include <bgclib/SimState.hpp>
#include <bgclib/Core/Coefficients.hpp>
#include <bgclib/Models/BGC/CoeffBGC.hpp>
#include <bgclib/Models/BGC/AssemblyBGC.hpp>
#include <bgclib/Models/TBGC/CoeffTBGC.hpp>
#include <bgclib/Models/TBGC/AssemblyTBGC.hpp>
#include <bgclib/Solver.hpp>

// ---------------------------------------------------------------------------
//  testSolver — verificação do solver linear
//
//  Estratégia: usa a identidade como sistema linear para isolar o solver.
//
//  Teste 1 — BGC  A=I  b=sin(πx):
//      MatZeroEntries + MatShift(1.0) → A = I
//      Solução esperada: phi = b = sin(πx)
//
//  Teste 2 — TBGC  A=I₂ₓ₂  b1=sin(πx)  b2=cos(πx):
//      Monta identidade em cada bloco.
//      Solução esperada: phi = b1 = sin(πx),  mu = b2 = cos(πx)
// ---------------------------------------------------------------------------

static constexpr PetscInt  NX  = 8;
static constexpr PetscReal LX  = 1.0;
static constexpr PetscReal H   = LX / NX;
static constexpr PetscReal TOL = 1.0e-10;
static constexpr PetscReal PI  = 3.14159265358979323846;

static int total  = 0;
static int passed = 0;

// ---------------------------------------------------------------------------
//  Utilitários
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

static PetscReal vecGet(Vec v, PetscInt i) {
    PetscScalar val = 0.0;
    VecGetValues(v, 1, &i, &val);
    return static_cast<PetscReal>(val);
}

static bgc::SimConfig makeConfig() {
    bgc::SimConfig cfg;
    cfg.nx = NX;  cfg.lx = LX;  cfg.h = H;
    cfg.bv = 0.01;  cfg.dt = 1.0e-4;
    cfg.verbose = false;
    for (int k = 0; k < 2; ++k) {
        cfg.bcWest[k] = bgc::BoundaryCondition::neumann(0.0);
        cfg.bcEast[k] = bgc::BoundaryCondition::neumann(0.0);
    }
    return cfg;
}

// ---------------------------------------------------------------------------
//  Teste 1 — BGC  A=I  b=sin(πx)
//
//  Usa MatZeroEntries + MatShift para montar a identidade sobre a matriz
//  já alocada pelo createStateBGC, sem conflitar com o padrão de esparsidade.
// ---------------------------------------------------------------------------

static PetscErrorCode testarSolverBGCIdentidade() {
    PetscFunctionBeginUser;

    secao("BGC — solver  A=I  b=sin(πx)  (solução exacta: phi=sin(πx))");

    auto cfg = makeConfig();
    cfg.model = bgc::Model::BGC;   // obrigatório: default é TBGC
    bgc::SimState st;
    PetscPrintf(PETSC_COMM_WORLD, "[DEBUG] Iniciando createStateBGC...\n");
    PetscCall(bgc::createStateBGC(cfg, st));
    PetscPrintf(PETSC_COMM_WORLD, "[DEBUG] createStateBGC OK\n");

    // Para BGC, st.x deve apontar para st.phi
    st.x = st.phi;
    st.is_phi = nullptr;
    st.is_mu  = nullptr;

    PetscPrintf(PETSC_COMM_WORLD, "[DEBUG] Zerando matriz A...\n");
    PetscCall(MatZeroEntries(st.A));
    PetscPrintf(PETSC_COMM_WORLD, "[DEBUG] MatZeroEntries OK\n");

    PetscPrintf(PETSC_COMM_WORLD, "[DEBUG] Aplicando MatShift(1.0)...\n");
    PetscCall(MatShift(st.A, 1.0));
    PetscPrintf(PETSC_COMM_WORLD, "[DEBUG] MatShift OK\n");

    PetscPrintf(PETSC_COMM_WORLD, "[DEBUG] Iniciando assembly de A...\n");
    PetscCall(MatAssemblyBegin(st.A, MAT_FINAL_ASSEMBLY));
    PetscPrintf(PETSC_COMM_WORLD, "[DEBUG] MatAssemblyBegin OK\n");
    PetscCall(MatAssemblyEnd(st.A, MAT_FINAL_ASSEMBLY));
    PetscPrintf(PETSC_COMM_WORLD, "[DEBUG] MatAssemblyEnd OK\n");

    // configureSolver deve ser chamado APÓS MatAssemblyEnd
    PetscPrintf(PETSC_COMM_WORLD, "[DEBUG] Configurando solver BGC...\n");
    PetscCall(bgc::configureSolver(cfg, st));
    PetscPrintf(PETSC_COMM_WORLD, "[DEBUG] configureSolver OK\n");

    // b = sin(πx)
    PetscInt xstart, xm;
    PetscPrintf(PETSC_COMM_WORLD, "[DEBUG] Obtendo corners do DMDA...\n");
    PetscCall(DMDAGetCorners(st.dm, &xstart, nullptr, nullptr, &xm, nullptr, nullptr));
    PetscPrintf(PETSC_COMM_WORLD, "[DEBUG] DMDAGetCorners: xstart=%d, xm=%d\n", (int)xstart, (int)xm);

    PetscPrintf(PETSC_COMM_WORLD, "[DEBUG] Preenchendo vetor b...\n");
    for (PetscInt i = xstart; i < xstart + xm; ++i) {
        const PetscReal xi = (i + 0.5) * H;
        PetscCall(VecSetValue(st.b, i, std::sin(PI * xi), INSERT_VALUES));
    }
    PetscPrintf(PETSC_COMM_WORLD, "[DEBUG] SetValues de b concluído\n");

    PetscPrintf(PETSC_COMM_WORLD, "[DEBUG] Iniciando assembly de b...\n");
    PetscCall(VecAssemblyBegin(st.b));
    PetscPrintf(PETSC_COMM_WORLD, "[DEBUG] VecAssemblyBegin OK\n");
    PetscCall(VecAssemblyEnd(st.b));
    PetscPrintf(PETSC_COMM_WORLD, "[DEBUG] VecAssemblyEnd OK\n");

    PetscPrintf(PETSC_COMM_WORLD, "[DEBUG] Chamando solveLinearSystem...\n");
    PetscCall(bgc::solveLinearSystem(cfg, st));
    PetscPrintf(PETSC_COMM_WORLD, "[DEBUG] solveLinearSystem retornou\n");

    // Verifica phi = b = sin(πx)
    for (PetscInt i = 0; i < NX; ++i) {
        const PetscReal xi      = (i + 0.5) * H;
        const PetscReal esperado = std::sin(PI * xi);
        checkClose(vecGet(st.phi, i), esperado,
                   std::format("phi[{}]  x={:.4f}  == sin(πx)={:.6f}",
                               (int)i, (double)xi, (double)esperado));
    }

    // Prevenir dupla liberação: st.x aponta para st.phi
    st.x = nullptr;

    PetscPrintf(PETSC_COMM_WORLD, "[DEBUG] Chamando destroyState...\n");
    PetscCall(bgc::destroyState(st));
    PetscPrintf(PETSC_COMM_WORLD, "[DEBUG] destroyState retornou\n");

    PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------------------------------------------
//  Teste 2 — TBGC  A=I₂ₓ₂  b1=sin(πx)  b2=cos(πx)
// ---------------------------------------------------------------------------

static PetscErrorCode testarSolverTBGCIdentidade() {
    PetscFunctionBeginUser;

    secao("TBGC — solver  A=I₂ₓ₂  (phi=sin(πx),  mu=cos(πx))");

    const auto cfg = makeConfig();
    bgc::SimState st;
    PetscPrintf(PETSC_COMM_WORLD, "[DEBUG] Iniciando createStateTBGC...\n");
    PetscCall(bgc::createStateTBGC(cfg, st));
    PetscPrintf(PETSC_COMM_WORLD, "[DEBUG] createStateTBGC OK\n");

    // Primeiro, montamos os sub‑blocos com zeros
    PetscPrintf(PETSC_COMM_WORLD, "[DEBUG] Montando A11, A12, A21, A22 (zero)...\n");
    PetscCall(MatAssemblyBegin(st.A11, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd  (st.A11, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyBegin(st.A12, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd  (st.A12, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyBegin(st.A21, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd  (st.A21, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyBegin(st.A22, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd  (st.A22, MAT_FINAL_ASSEMBLY));
    PetscPrintf(PETSC_COMM_WORLD, "[DEBUG] Matrizes sub‑blocos montadas (zeros)\n");

    // Agora aplicamos MatShift para tornar A11 e A22 = I
    PetscPrintf(PETSC_COMM_WORLD, "[DEBUG] Aplicando MatShift(1.0) em A11...\n");
    PetscCall(MatShift(st.A11, 1.0));
    PetscPrintf(PETSC_COMM_WORLD, "[DEBUG] Aplicando MatShift(1.0) em A22...\n");
    PetscCall(MatShift(st.A22, 1.0));

    // Montar a matriz aninhada st.A
    PetscPrintf(PETSC_COMM_WORLD, "[DEBUG] Montando matriz aninhada st.A...\n");
    PetscCall(MatAssemblyBegin(st.A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(st.A, MAT_FINAL_ASSEMBLY));
    PetscPrintf(PETSC_COMM_WORLD, "[DEBUG] Matriz aninhada montada\n");

    // configureSolver deve ser chamado APÓS MatAssemblyEnd
    PetscPrintf(PETSC_COMM_WORLD, "[DEBUG] Configurando solver TBGC...\n");
    PetscCall(bgc::configureSolver(cfg, st));
    PetscPrintf(PETSC_COMM_WORLD, "[DEBUG] configureSolver OK\n");

    // b1 = sin(πx),  b2 = cos(πx)
    PetscInt xstart, xm;
    PetscPrintf(PETSC_COMM_WORLD, "[DEBUG] Obtendo corners do DMDA...\n");
    PetscCall(DMDAGetCorners(st.dm, &xstart, nullptr, nullptr,
                             &xm, nullptr, nullptr));
    PetscPrintf(PETSC_COMM_WORLD, "[DEBUG] DMDAGetCorners: xstart=%d, xm=%d\n", (int)xstart, (int)xm);

    PetscPrintf(PETSC_COMM_WORLD, "[DEBUG] Preenchendo vetores b1 e b2...\n");
    for (PetscInt i = xstart; i < xstart + xm; ++i) {
        const PetscReal xi = (i + 0.5) * H;
        PetscCall(VecSetValue(st.b1, i, std::sin(PI * xi), INSERT_VALUES));
        PetscCall(VecSetValue(st.b2, i, std::cos(PI * xi), INSERT_VALUES));
    }
    PetscPrintf(PETSC_COMM_WORLD, "[DEBUG] SetValues de b1 e b2 concluído\n");

    PetscPrintf(PETSC_COMM_WORLD, "[DEBUG] Iniciando assembly de b1, b2 e b...\n");
    PetscCall(VecAssemblyBegin(st.b1)); PetscCall(VecAssemblyEnd(st.b1));
    PetscCall(VecAssemblyBegin(st.b2)); PetscCall(VecAssemblyEnd(st.b2));
    PetscCall(VecAssemblyBegin(st.b));  PetscCall(VecAssemblyEnd(st.b));
    PetscPrintf(PETSC_COMM_WORLD, "[DEBUG] Assembly dos vetores concluído\n");

    // Resolve
    PetscPrintf(PETSC_COMM_WORLD, "[DEBUG] Chamando solveLinearSystem...\n");
    PetscCall(bgc::solveLinearSystem(cfg, st));
    PetscPrintf(PETSC_COMM_WORLD, "[DEBUG] solveLinearSystem retornou\n");

    // Verifica phi = sin(πx),  mu = cos(πx)
    for (PetscInt i = 0; i < NX; ++i) {
        const PetscReal xi = (i + 0.5) * H;
        checkClose(vecGet(st.phi, i), std::sin(PI * xi),
                   std::format("phi[{}]  == sin(πx={:.4f})", (int)i, (double)xi));
        checkClose(vecGet(st.mu,  i), std::cos(PI * xi),
                   std::format("mu[{}]   == cos(πx={:.4f})", (int)i, (double)xi));
    }

    PetscPrintf(PETSC_COMM_WORLD, "[DEBUG] Chamando destroyState...\n");
    PetscCall(bgc::destroyState(st));
    PetscPrintf(PETSC_COMM_WORLD, "[DEBUG] destroyState retornou\n");

    PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------------------------------------------
//  main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {

    PetscInitialize(&argc, &argv, nullptr,
                    "testSolver — verificação do solver linear");

    PetscPrintf(PETSC_COMM_WORLD, "\n%s\n",
                std::string(60, '=').c_str());
    PetscPrintf(PETSC_COMM_WORLD,
                "  testSolver  —  verificação do solver linear\n");
    PetscPrintf(PETSC_COMM_WORLD, "  nx=%d  h=%.4f\n",
                (int)NX, (double)H);
    PetscPrintf(PETSC_COMM_WORLD, "%s\n",
                std::string(60, '=').c_str());

    PetscCall(testarSolverBGCIdentidade());
    PetscCall(testarSolverTBGCIdentidade());

    PetscPrintf(PETSC_COMM_WORLD, "\n%s\n",
                std::string(60, '-').c_str());
    PetscPrintf(PETSC_COMM_WORLD, "  Resultado: %d/%d testes passaram\n",
                passed, total);
    PetscPrintf(PETSC_COMM_WORLD, "%s\n\n",
                std::string(60, '-').c_str());

    PetscFinalize();
    return (passed == total) ? 0 : 1;
}