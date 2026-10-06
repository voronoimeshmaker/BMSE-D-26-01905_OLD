#include <iostream>
#include <format>
#include <cmath>

#include <BCGLib.hpp>

// ---------------------------------------------------------------------------
//  Utilitários de diagnóstico
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

static void secao(const std::string& nome) {
    std::cout << std::format("\n{}\n{}\n", nome,
                             std::string(nome.size(), '-'));
}

// ---------------------------------------------------------------------------
//  Testes de BoundaryCondition
// ---------------------------------------------------------------------------

static void testarBoundaryCondition() {

    secao("BoundaryCondition — tipo");
    {
        auto bc = bgc::BoundaryCondition::dirichlet(1.0);
        check(bc.isDirichlet(),  "dirichlet() -> isDirichlet()");
        check(!bc.isNeumann(),   "dirichlet() -> !isNeumann()");
        check(!bc.isRobin(),     "dirichlet() -> !isRobin()");
        check(bc.type() == bgc::BoundaryCondition::Type::Dirichlet,
                                 "dirichlet() -> type() == Dirichlet");
    }
    {
        auto bc = bgc::BoundaryCondition::neumann(0.0);
        check(bc.isNeumann(),    "neumann() -> isNeumann()");
        check(!bc.isDirichlet(), "neumann() -> !isDirichlet()");
        check(!bc.isRobin(),     "neumann() -> !isRobin()");
    }
    {
        auto bc = bgc::BoundaryCondition::robin(2.0, 1.0, 0.5);
        check(bc.isRobin(),      "robin() -> isRobin()");
        check(!bc.isDirichlet(), "robin() -> !isDirichlet()");
        check(!bc.isNeumann(),   "robin() -> !isNeumann()");
    }

    secao("BoundaryCondition — alpha e beta");
    {
        auto bc = bgc::BoundaryCondition::dirichlet(3.0);
        check(bc.alpha() == 1.0, "dirichlet: alpha == 1.0");
        check(bc.beta()  == 0.0, "dirichlet: beta  == 0.0");
    }
    {
        auto bc = bgc::BoundaryCondition::neumann(0.0);
        check(bc.alpha() == 0.0, "neumann: alpha == 0.0");
        check(bc.beta()  == 1.0, "neumann: beta  == 1.0");
    }
    {
        auto bc = bgc::BoundaryCondition::robin(2.0, 3.0, 0.5);
        check(bc.alpha() == 2.0, "robin: alpha == 2.0");
        check(bc.beta()  == 3.0, "robin: beta  == 3.0");
    }

    secao("BoundaryCondition — gamma constante");
    {
        auto bc = bgc::BoundaryCondition::dirichlet(1.5);
        check(!bc.isTimeDependent(),    "gamma constante -> !isTimeDependent()");
        check(bc.gamma()    == 1.5,     "gamma() sem argumento retorna 1.5");
        check(bc.gamma(0.0) == 1.5,     "gamma(0.0) retorna 1.5");
        check(bc.gamma(9.9) == 1.5,     "gamma(9.9) retorna 1.5 (t ignorado)");
    }
    {
        auto bc = bgc::BoundaryCondition::neumann(0.0);
        check(bc.gamma() == 0.0, "neumann gamma constante == 0.0");
    }

    secao("BoundaryCondition — gamma função do tempo");
    {
        const PetscReal lambda = -1.0;
        auto bc = bgc::BoundaryCondition::dirichlet(
            [lambda](PetscReal t) { return std::exp(lambda * t); });

        check(bc.isTimeDependent(), "lambda -> isTimeDependent()");
        check(bc.isDirichlet(),     "lambda -> isDirichlet()");
        check(std::abs(bc.gamma(0.0) - std::exp(lambda*0.0)) < 1e-14,
                                    "gamma(0.0) == exp(0) == 1.0");
        check(std::abs(bc.gamma(1.0) - std::exp(lambda*1.0)) < 1e-14,
                                    "gamma(1.0) == exp(-1)");
        check(std::abs(bc.gamma(2.5) - std::exp(lambda*2.5)) < 1e-14,
                                    "gamma(2.5) == exp(-2.5)");
    }
    {
        auto fn = [](PetscReal t) -> PetscReal { return std::sin(t); };
        auto bc = bgc::BoundaryCondition::neumann(fn);
        check(bc.isTimeDependent(), "ponteiro de função -> isTimeDependent()");
        check(std::abs(bc.gamma(0.0) - std::sin(0.0)) < 1e-14,
                                    "gamma(0.0) == sin(0.0)");
        check(std::abs(bc.gamma(1.0) - std::sin(1.0)) < 1e-14,
                                    "gamma(1.0) == sin(1.0)");
    }

    secao("BoundaryCondition — fromYAML");
    {
        YAML::Node node;
        auto bc0 = bgc::BoundaryCondition::fromYAML(node, 0);
        auto bc1 = bgc::BoundaryCondition::fromYAML(node, 1);
        check(bc0.isDirichlet(),  "fromYAML k=0 padrão -> Dirichlet");
        check(bc0.alpha() == 1.0, "fromYAML k=0 padrão -> alpha == 1.0");
        check(bc0.beta()  == 0.0, "fromYAML k=0 padrão -> beta  == 0.0");
        check(bc0.gamma() == 0.0, "fromYAML k=0 padrão -> gamma == 0.0");
        check(bc1.isNeumann(),    "fromYAML k=1 padrão -> Neumann");
        check(bc1.alpha() == 0.0, "fromYAML k=1 padrão -> alpha == 0.0");
        check(bc1.beta()  == 1.0, "fromYAML k=1 padrão -> beta  == 1.0");
        check(bc1.gamma() == 0.0, "fromYAML k=1 padrão -> gamma == 0.0");
    }
    {
        YAML::Node node;
        node["alpha0"] = 2.0;
        node["beta0"]  = 3.0;
        node["gamma0"] = 0.5;
        auto bc = bgc::BoundaryCondition::fromYAML(node, 0);
        check(bc.isRobin(),       "fromYAML Robin explícito");
        check(bc.alpha() == 2.0,  "fromYAML alpha == 2.0");
        check(bc.beta()  == 3.0,  "fromYAML beta  == 3.0");
        check(bc.gamma() == 0.5,  "fromYAML gamma == 0.5");
    }
}

// ---------------------------------------------------------------------------
//  Testes de Coefficients
// ---------------------------------------------------------------------------

static void testarCoefficients() {

    secao("VolumeRegion — classifyVolume  (n = 8)");
    {
        const PetscInt n = 8;
        check(bgc::classifyVolume(0, n) == bgc::VolumeRegion::First,
              "i=0 -> First");
        check(bgc::classifyVolume(1, n) == bgc::VolumeRegion::Second,
              "i=1 -> Second");
        check(bgc::classifyVolume(2, n) == bgc::VolumeRegion::Interior,
              "i=2 -> Interior");
        check(bgc::classifyVolume(3, n) == bgc::VolumeRegion::Interior,
              "i=3 -> Interior");
        check(bgc::classifyVolume(4, n) == bgc::VolumeRegion::Interior,
              "i=4 -> Interior");
        check(bgc::classifyVolume(5, n) == bgc::VolumeRegion::Interior,
              "i=5 -> Interior");
        check(bgc::classifyVolume(6, n) == bgc::VolumeRegion::SecondToLast,
              "i=6 -> SecondToLast");
        check(bgc::classifyVolume(7, n) == bgc::VolumeRegion::Last,
              "i=7 -> Last");
    }

    secao("VolumeRegion — classifyVolume  (n = 4, malha mínima)");
    {
        // Com n=4 não há volumes interiores — os quatro especiais cobrem tudo.
        const PetscInt n = 4;
        check(bgc::classifyVolume(0, n) == bgc::VolumeRegion::First,
              "n=4: i=0 -> First");
        check(bgc::classifyVolume(1, n) == bgc::VolumeRegion::Second,
              "n=4: i=1 -> Second");
        check(bgc::classifyVolume(2, n) == bgc::VolumeRegion::SecondToLast,
              "n=4: i=2 -> SecondToLast");
        check(bgc::classifyVolume(3, n) == bgc::VolumeRegion::Last,
              "n=4: i=3 -> Last");
    }

    secao("StencilRow — estrutura e valores padrão");
    {
        bgc::StencilRow row;
        check(row.ncols    == 0,   "StencilRow padrão: ncols == 0");
        check(row.coef[0]  == 0.0, "StencilRow padrão: coef[0] == 0.0");
        check(row.col[0]   == 0,   "StencilRow padrão: col[0]  == 0");
    }
    {
        // Estêncil interior simétrico de cinco pontos
        bgc::StencilRow row;
        row.ncols    = 5;
        row.col[0]   = -2;  row.coef[0] =  1.0;
        row.col[1]   = -1;  row.coef[1] = -4.0;
        row.col[2]   =  0;  row.coef[2] =  6.0;
        row.col[3]   =  1;  row.coef[3] = -4.0;
        row.col[4]   =  2;  row.coef[4] =  1.0;

        check(row.ncols   == 5,   "StencilRow: ncols == 5");
        check(row.coef[2] == 6.0, "StencilRow: coef central == 6.0");
        check(row.coef[0] == row.coef[4],
                                  "StencilRow simétrico: coef[0] == coef[4]");
        check(row.coef[1] == row.coef[3],
                                  "StencilRow simétrico: coef[1] == coef[3]");
        check(row.col[0]  == -row.col[4],
                                  "StencilRow: offsets opostos col[0] == -col[4]");
        check(row.col[1]  == -row.col[3],
                                  "StencilRow: offsets opostos col[1] == -col[3]");
    }

    secao("StencilCoefficients — inicialização padrão");
    {
        bgc::StencilCoefficients sc;
        check(sc.vol0.ncols     == 0, "vol0.ncols     == 0 (padrão)");
        check(sc.vol1.ncols     == 0, "vol1.ncols     == 0 (padrão)");
        check(sc.interior.ncols == 0, "interior.ncols == 0 (padrão)");
        check(sc.volNm2.ncols   == 0, "volNm2.ncols   == 0 (padrão)");
        check(sc.volNm1.ncols   == 0, "volNm1.ncols   == 0 (padrão)");
    }

    secao("RHSCoefficients — inicialização e atribuição");
    {
        bgc::RHSCoefficients rhs;
        check(rhs.vol0   == 0.0, "vol0   == 0.0 (padrão)");
        check(rhs.vol1   == 0.0, "vol1   == 0.0 (padrão)");
        check(rhs.volNm2 == 0.0, "volNm2 == 0.0 (padrão)");
        check(rhs.volNm1 == 0.0, "volNm1 == 0.0 (padrão)");
    }
    {
        bgc::RHSCoefficients rhs;
        rhs.vol0   =  1.5;
        rhs.volNm1 = -1.5;
        check(rhs.vol0   ==  1.5,          "vol0   ==  1.5 (atribuído)");
        check(rhs.volNm1 == -1.5,          "volNm1 == -1.5 (atribuído)");
        check(rhs.vol0   == -rhs.volNm1,   "simetria: vol0 == -volNm1");
        check(rhs.vol1   ==  0.0,          "vol1 não alterado == 0.0");
        check(rhs.volNm2 ==  0.0,          "volNm2 não alterado == 0.0");
    }
}

// ---------------------------------------------------------------------------
//  main
// ---------------------------------------------------------------------------

int main() {

    std::cout << std::format("\n{}\n", std::string(60, '='));
    std::cout << "  testLib  —  verificação da bgclib\n";
    std::cout << std::format("{}\n", std::string(60, '='));

    testarBoundaryCondition();
    testarCoefficients();

    std::cout << std::format("\n{}\n", std::string(60, '-'));
    std::cout << std::format("  Resultado: {}/{} testes passaram\n",
                             passed, total);
    std::cout << std::format("{}\n\n", std::string(60, '-'));

    return (passed == total) ? 0 : 1;
}