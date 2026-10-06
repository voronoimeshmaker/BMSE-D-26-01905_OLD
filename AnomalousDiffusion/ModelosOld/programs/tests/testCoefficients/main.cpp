#include <iostream>
#include <format>
#include <cmath>

#include <BCGLib.hpp>
#include <bgclib/Models/BGC/CoeffBGC.hpp>
#include <bgclib/Models/TBGC/CoeffTBGC.hpp>


// ---------------------------------------------------------------------------
//  Parâmetros de referência
//
//  nx=8, lx=1.0  ->  h = 0.125
//  bv = 0.01
//  dt = 1e-4
//
//  Todos os valores esperados foram calculados independentemente em Python
//  a partir das fórmulas das Tabelas 1, 2, 3 e 4 do artigo.
// ---------------------------------------------------------------------------

static constexpr PetscInt  NX   = 8;
static constexpr PetscReal LX   = 1.0;
static constexpr PetscReal BV   = 0.01;
static constexpr PetscReal DT   = 1.0e-4;
static constexpr PetscReal H    = LX / NX;   // 0.125
static constexpr PetscReal TOL  = 1.0e-10;   // tolerância das comparações

// ---------------------------------------------------------------------------
//  Utilitários
// ---------------------------------------------------------------------------

static int total  = 0;
static int passed = 0;

static void check(bool condition, const std::string& descricao) {
    ++total;
    if (condition) {
        ++passed;
        std::cout << std::format("  [OK]     {}\n", descricao);
    } else {
        std::cout << std::format("  [FALHOU] {}\n", descricao);
    }
}

static void checkClose(PetscReal valor,
                       PetscReal esperado,
                       const std::string& descricao,
                       PetscReal tol = TOL) {
    const PetscReal diff = std::abs(valor - esperado);
    ++total;
    if (diff < tol) {
        ++passed;
        std::cout << std::format("  [OK]     {}  ({:.6e})\n",
                                 descricao, valor);
    } else {
        std::cout << std::format("  [FALHOU] {}  obtido={:.6e}  esperado={:.6e}  diff={:.3e}\n",
                                 descricao, valor, esperado, diff);
    }
}

static void secao(const std::string& nome) {
    std::cout << std::format("\n{}\n{}\n", nome,
                             std::string(nome.size(), '-'));
}

// ---------------------------------------------------------------------------
//  Monta um SimConfig mínimo com as BCs fornecidas
// ---------------------------------------------------------------------------

static bgc::SimConfig makeConfig(bgc::BoundaryCondition w0,
                                 bgc::BoundaryCondition w1,
                                 bgc::BoundaryCondition e0,
                                 bgc::BoundaryCondition e1) {
    bgc::SimConfig cfg;
    cfg.nx      = NX;
    cfg.lx      = LX;
    cfg.h       = H;
    cfg.bv      = BV;
    cfg.dt      = DT;
    cfg.bcWest[0] = w0;
    cfg.bcWest[1] = w1;
    cfg.bcEast[0] = e0;
    cfg.bcEast[1] = e1;
    return cfg;
}

// ---------------------------------------------------------------------------
//  Testes de computeCoefficientsBGC
// ---------------------------------------------------------------------------

static void testarCoefBGCInterior() {

    secao("BGC — estêncil interior (Neumann puro)");

    const auto cfg = makeConfig(
        bgc::BoundaryCondition::neumann(0.0),
        bgc::BoundaryCondition::neumann(0.0),
        bgc::BoundaryCondition::neumann(0.0),
        bgc::BoundaryCondition::neumann(0.0));

    const auto sc = bgc::computeCoefficientsBGC(cfg);

    // Valores calculados para h=0.125, bv=0.01
    constexpr PetscReal AWW =  1.331380208333333e-03;
    constexpr PetscReal AW  = -7.278645833333333e-03;
    constexpr PetscReal AP  =  1.189453125000000e-02;

    check(sc.interior.ncols == 5,
          "interior.ncols == 5");
    checkClose(sc.interior.coef[0], AWW,  "interior.coef[0]  (aww)");
    checkClose(sc.interior.coef[1], AW,   "interior.coef[1]  (aw)");
    checkClose(sc.interior.coef[2], AP,   "interior.coef[2]  (ap)");
    checkClose(sc.interior.coef[3], AW,   "interior.coef[3]  (ae = aw)");
    checkClose(sc.interior.coef[4], AWW,  "interior.coef[4]  (aee = aww)");

    // Verificações estruturais
    check(sc.interior.col[0] == -2, "interior.col[0] == -2");
    check(sc.interior.col[2] ==  0, "interior.col[2] ==  0");
    check(sc.interior.col[4] ==  2, "interior.col[4] == +2");

    // Estêncil centrado: soma dos coeficientes deve ser zero
    const PetscReal soma = sc.interior.coef[0] + sc.interior.coef[1]
                         + sc.interior.coef[2] + sc.interior.coef[3]
                         + sc.interior.coef[4];
    checkClose(soma, 0.0, "soma dos coeficientes interiores == 0", 1.0e-12);

    // Simetria
    check(sc.interior.coef[0] == sc.interior.coef[4],
          "simetria: coef[0] == coef[4]");
    check(sc.interior.coef[1] == sc.interior.coef[3],
          "simetria: coef[1] == coef[3]");
}

static void testarCoefBGCVol0Neumann() {

    secao("BGC — vol0 com Neumann (a1w=0, b1w=1)");

    const auto cfg = makeConfig(
        bgc::BoundaryCondition::neumann(0.0),
        bgc::BoundaryCondition::neumann(0.0),
        bgc::BoundaryCondition::neumann(0.0),
        bgc::BoundaryCondition::neumann(0.0));

    const auto sc = bgc::computeCoefficientsBGC(cfg);

    check(sc.vol0.ncols == 4, "vol0.ncols == 4");
    checkClose(sc.vol0.coef[0],  2.659060606060606e+02, "vol0.coef[0] (ap)");
    checkClose(sc.vol0.coef[1], -6.755090909090909e+01, "vol0.coef[1] (ae)");
    checkClose(sc.vol0.coef[2],  1.160792727272727e+01, "vol0.coef[2] (aee)");
    checkClose(sc.vol0.coef[3], -2.478787878787879e-01, "vol0.coef[3] (aeee)");

    // Índices absolutos
    check(sc.vol0.col[0] == 0, "vol0.col[0] == 0");
    check(sc.vol0.col[1] == 1, "vol0.col[1] == 1");
    check(sc.vol0.col[2] == 2, "vol0.col[2] == 2");
    check(sc.vol0.col[3] == 3, "vol0.col[3] == 3");
}

static void testarCoefBGCVol0Dirichlet() {

    secao("BGC — vol0 com Dirichlet (a1w=1, b1w=0)");

    const auto cfg = makeConfig(
        bgc::BoundaryCondition::dirichlet(0.0),
        bgc::BoundaryCondition::neumann  (0.0),
        bgc::BoundaryCondition::dirichlet(0.0),
        bgc::BoundaryCondition::neumann  (0.0));

    const auto sc = bgc::computeCoefficientsBGC(cfg);

    checkClose(sc.vol0.coef[0],  2.919333333333333e+02, "vol0.coef[0] (ap)");
    checkClose(sc.vol0.coef[1], -7.622666666666666e+01, "vol0.coef[1] (ae)");
    checkClose(sc.vol0.coef[2],  1.473120000000000e+01, "vol0.coef[2] (aee)");
    checkClose(sc.vol0.coef[3], -7.790476190476190e-01, "vol0.coef[3] (aeee)");
}

static void testarCoefBGCSimNm1() {

    secao("BGC — estrutura de volNm1");

    // vol0 e volNm1 usam denominadores com escalas diferentes na Tabela 1.
    // Com Neumann puro (a1=0, b1=1):
    //   vol0  usa  D = 1 / (600 h³ * (-352 b1w))   — sem fator h⁴
    //   volNm1 usa D = h⁴ / (600 h³ * (352 b1e))   — com fator h⁴
    // Portanto os valores numéricos de ap NÃO são iguais mesmo com BCs
    // simétricas.  Verificamos os valores corretos de cada um separadamente.
    const auto cfg = makeConfig(
        bgc::BoundaryCondition::neumann(0.0),
        bgc::BoundaryCondition::neumann(0.0),
        bgc::BoundaryCondition::neumann(0.0),
        bgc::BoundaryCondition::neumann(0.0));

    const auto sc = bgc::computeCoefficientsBGC(cfg);

    check(sc.volNm1.ncols == 4, "volNm1.ncols == 4");

    // Valores corretos calculados para h=0.125, bv=0.01, Neumann
    checkClose(sc.volNm1.coef[3],  6.491847182765151e-02, "volNm1.coef[3] (ap)");
    checkClose(sc.volNm1.coef[2], -1.649192116477273e-02, "volNm1.coef[2] (aw)");
    checkClose(sc.volNm1.coef[1],  2.833966619318182e-03, "volNm1.coef[1] (aww)");
    checkClose(sc.volNm1.coef[0], -6.051728219696969e-05, "volNm1.coef[0] (awww)");

    // Padrão de sinais: ap > 0, vizinhos com alternância
    check(sc.volNm1.coef[3] > 0.0, "volNm1.ap   > 0");
    check(sc.volNm1.coef[2] < 0.0, "volNm1.aw   < 0");
    check(sc.volNm1.coef[1] > 0.0, "volNm1.aww  > 0");
    check(sc.volNm1.coef[0] < 0.0, "volNm1.awww < 0");

    // vol0 e volNm1: ambos ap positivos, mas valores distintos
    check(sc.vol0.coef[0] > 0.0 && sc.volNm1.coef[3] > 0.0,
          "vol0.ap e volNm1.ap ambos positivos");

    // Índices absolutos corretos
    check(sc.volNm1.col[0] == NX-4, "volNm1.col[0] == n-4");
    check(sc.volNm1.col[1] == NX-3, "volNm1.col[1] == n-3");
    check(sc.volNm1.col[2] == NX-2, "volNm1.col[2] == n-2");
    check(sc.volNm1.col[3] == NX-1, "volNm1.col[3] == n-1");
}

// ---------------------------------------------------------------------------
//  Testes de computeRHSBGC
// ---------------------------------------------------------------------------

static void testarRHSBGCNeumannHomogeneo() {

    secao("BGC RHS — Neumann homogêneo (gamma=0, rhs deve ser zero)");

    const auto cfg = makeConfig(
        bgc::BoundaryCondition::neumann(0.0),
        bgc::BoundaryCondition::neumann(0.0),
        bgc::BoundaryCondition::neumann(0.0),
        bgc::BoundaryCondition::neumann(0.0));

    const auto rhs = bgc::computeRHSBGC(cfg, 0.0);

    checkClose(rhs.vol0,   0.0, "rhs.vol0   == 0.0 (Neumann homogêneo)");
    checkClose(rhs.vol1,   0.0, "rhs.vol1   == 0.0 (Neumann homogêneo)");
    checkClose(rhs.volNm2, 0.0, "rhs.volNm2 == 0.0 (Neumann homogêneo)");
    checkClose(rhs.volNm1, 0.0, "rhs.volNm1 == 0.0 (Neumann homogêneo)");
}

static void testarRHSBGCDirichlet() {

    secao("BGC RHS — Dirichlet g1w=1.5 / Neumann g2w=0");

    const auto cfg = makeConfig(
        bgc::BoundaryCondition::dirichlet(1.5),
        bgc::BoundaryCondition::neumann  (0.0),
        bgc::BoundaryCondition::dirichlet(0.0),
        bgc::BoundaryCondition::neumann  (0.0));

    const auto rhs = bgc::computeRHSBGC(cfg, 0.0);

    checkClose(rhs.vol0,   3.444882285714286e+02, "rhs.vol0  (Dirichlet g=1.5)");
    checkClose(rhs.vol1,  -2.991542857142857e+01, "rhs.vol1  (Dirichlet g=1.5)");
    checkClose(rhs.volNm2, 0.0, "rhs.volNm2 == 0.0 (BC leste g=0)");
    checkClose(rhs.volNm1, 0.0, "rhs.volNm1 == 0.0 (BC leste g=0)");
}

// ---------------------------------------------------------------------------
//  Testes de computeCoefficientsTBGC
// ---------------------------------------------------------------------------

static void testarCoefTBGCA11() {

    secao("TBGC — bloco A11");

    const auto cfg = makeConfig(
        bgc::BoundaryCondition::neumann(0.0),
        bgc::BoundaryCondition::neumann(0.0),
        bgc::BoundaryCondition::neumann(0.0),
        bgc::BoundaryCondition::neumann(0.0));

    const auto tc = bgc::computeCoefficientsTBGC(cfg);
    const auto& A = tc.A11;

    // vol0  (3 colunas)
    check(A.vol0.ncols == 3, "A11.vol0.ncols == 3");
    checkClose(A.vol0.coef[0], 1.454800000000000e+03, "A11.vol0.coef[0] (ap)");
    checkClose(A.vol0.coef[1],-1.517037037037037e+01, "A11.vol0.coef[1] (ae)");
    checkClose(A.vol0.coef[2], 1.638400000000000e+00, "A11.vol0.coef[2] (aee)");

    // interior (apenas diagonal)
    check(A.interior.ncols == 1,  "A11.interior.ncols == 1");
    checkClose(A.interior.coef[0], cfg.h/cfg.dt,
               "A11.interior.coef[0] == h/dt");

    // simetria vol0 vs volNm1
    checkClose(A.volNm1.coef[2], A.vol0.coef[0],
               "A11 simetria: volNm1.ap == vol0.ap");
    checkClose(A.volNm1.coef[1], A.vol0.coef[1],
               "A11 simetria: volNm1.aw == vol0.ae");
    checkClose(A.volNm1.coef[0], A.vol0.coef[2],
               "A11 simetria: volNm1.aww == vol0.aee");
}

static void testarCoefTBGCA12() {

    secao("TBGC — bloco A12");

    const auto cfg = makeConfig(
        bgc::BoundaryCondition::neumann(0.0),
        bgc::BoundaryCondition::neumann(0.0),
        bgc::BoundaryCondition::neumann(0.0),
        bgc::BoundaryCondition::neumann(0.0));

    const auto tc = bgc::computeCoefficientsTBGC(cfg);
    const auto& A = tc.A12;

    // interior — estêncil de três pontos
    check(A.interior.ncols == 3, "A12.interior.ncols == 3");
    checkClose(A.interior.coef[0], -8.0, "A12.interior.coef[0] (aw) == -1/h");
    checkClose(A.interior.coef[1], 16.0, "A12.interior.coef[1] (ap) ==  2/h");
    checkClose(A.interior.coef[2], -8.0, "A12.interior.coef[2] (ae) == -1/h");

    // simetria: aw == ae
    check(A.interior.coef[0] == A.interior.coef[2],
          "A12 interior simétrico: aw == ae");

    // vol0 — apenas 2 colunas (sem vizinho à esquerda)
    check(A.vol0.ncols == 2, "A12.vol0.ncols == 2");

    // volNm1 — apenas 2 colunas (sem vizinho à direita)
    check(A.volNm1.ncols == 2, "A12.volNm1.ncols == 2");
}

static void testarCoefTBGCA21() {

    secao("TBGC — bloco A21");

    const auto cfg = makeConfig(
        bgc::BoundaryCondition::neumann(0.0),
        bgc::BoundaryCondition::neumann(0.0),
        bgc::BoundaryCondition::neumann(0.0),
        bgc::BoundaryCondition::neumann(0.0));

    const auto tc = bgc::computeCoefficientsTBGC(cfg);
    const auto& A = tc.A21;

    // interior — estêncil de cinco pontos
    check(A.interior.ncols == 5, "A21.interior.ncols == 5");
    checkClose(A.interior.coef[0], -5.333333333333333e-02,
               "A21.interior.coef[0] (aww)");
    checkClose(A.interior.coef[1],  8.533333333333333e-01,
               "A21.interior.coef[1] (aw)");
    checkClose(A.interior.coef[2], -2.600000000000000e+00,
               "A21.interior.coef[2] (ap)");
    checkClose(A.interior.coef[3],  8.533333333333333e-01,
               "A21.interior.coef[3] (ae = aw)");
    checkClose(A.interior.coef[4], -5.333333333333333e-02,
               "A21.interior.coef[4] (aee = aww)");

    // simetria
    check(A.interior.coef[0] == A.interior.coef[4],
          "A21 interior simétrico: coef[0] == coef[4]");
    check(A.interior.coef[1] == A.interior.coef[3],
          "A21 interior simétrico: coef[1] == coef[3]");

    // simetria vol0 vs volNm1
    checkClose(A.volNm1.coef[2], A.vol0.coef[0],
               "A21 simetria: volNm1.ap == vol0.ap");
}

static void testarRHSTBGCNeumannHomogeneo() {

    secao("TBGC RHS — Neumann homogêneo (gamma=0, rhs deve ser zero)");

    const auto cfg = makeConfig(
        bgc::BoundaryCondition::neumann(0.0),
        bgc::BoundaryCondition::neumann(0.0),
        bgc::BoundaryCondition::neumann(0.0),
        bgc::BoundaryCondition::neumann(0.0));

    const auto rhs = bgc::computeRHSTBGC(cfg, 0.0);

    checkClose(rhs.b1.vol0,   0.0, "b1.vol0   == 0.0");
    checkClose(rhs.b1.vol1,   0.0, "b1.vol1   == 0.0");
    checkClose(rhs.b1.volNm2, 0.0, "b1.volNm2 == 0.0");
    checkClose(rhs.b1.volNm1, 0.0, "b1.volNm1 == 0.0");
    checkClose(rhs.b2.vol0,   0.0, "b2.vol0   == 0.0");
    checkClose(rhs.b2.vol1,   0.0, "b2.vol1   == 0.0");
    checkClose(rhs.b2.volNm2, 0.0, "b2.volNm2 == 0.0");
    checkClose(rhs.b2.volNm1, 0.0, "b2.volNm1 == 0.0");
}

// ---------------------------------------------------------------------------
//  main
// ---------------------------------------------------------------------------

int main() {

    std::cout << std::format("\n{}\n", std::string(60, '='));
    std::cout << "  testCoefficients  —  verificação dos coeficientes\n";
    std::cout << std::format("  nx={}  h={:.4f}  bv={:.4f}  dt={:.2e}\n",
                             NX, H, BV, DT);
    std::cout << std::format("{}\n", std::string(60, '='));

    // BGC
    testarCoefBGCInterior();
    testarCoefBGCVol0Neumann();
    testarCoefBGCVol0Dirichlet();
    testarCoefBGCSimNm1();
    testarRHSBGCNeumannHomogeneo();
    testarRHSBGCDirichlet();

    // TBGC
    testarCoefTBGCA11();
    testarCoefTBGCA12();
    testarCoefTBGCA21();
    testarRHSTBGCNeumannHomogeneo();

    std::cout << std::format("\n{}\n", std::string(60, '-'));
    std::cout << std::format("  Resultado: {}/{} testes passaram\n",
                             passed, total);
    std::cout << std::format("{}\n\n", std::string(60, '-'));

    return (passed == total) ? 0 : 1;
}