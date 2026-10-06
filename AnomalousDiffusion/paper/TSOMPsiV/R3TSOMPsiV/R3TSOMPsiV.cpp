#include "PaperTSOMPsiV.hpp"

using namespace paper_tsompsiv;

namespace {
void buildSources(const Mesh& mesh,
                  const bgc::models::tsompsiv::Constants& c,
                  std::vector<PetscReal>& sPsi,
                  std::vector<PetscReal>& sV) {
    const auto d = bgc::models::tsompsiv::computeDiffusionCoefficients(c);
    sPsi.assign(static_cast<std::size_t>(mesh.n), 0.0);
    sV.assign(static_cast<std::size_t>(mesh.n), 0.0);
    for (PetscInt i = 0; i < mesh.n; ++i) {
        const PetscReal x = xCell(mesh, i);
        const PetscReal g = polyProfile(x);
        const PetscReal gpp = polySecond(x);
        const PetscReal psi = 2.0 * g;
        const PetscReal v = g;
        const PetscReal psipp = 2.0 * gpp;
        const PetscReal vpp = gpp;
        sPsi[static_cast<std::size_t>(i)] = -psi - d.d11 * psipp - d.d12 * vpp;
        sV[static_cast<std::size_t>(i)] =
            -v - c.lambdaC * psi + (c.lambdaC + c.lambdaR) * v - d.d21 * psipp - d.d22 * vpp;
    }
}
}

int main(int argc, char** argv) {
    PetscCallAbort(PETSC_COMM_WORLD, PetscInitialize(&argc, &argv, nullptr, nullptr));
    const auto out = outputDir(__FILE__);
    const auto bc = noFluxBoundaries();
    std::ofstream summary = bgc::openOutputFile(out / "summary.csv");
    summary << "param_set,theta,nx,l2_psi,l2_v\n";
    for (const auto& p : {std::pair{"P2", std::pair{0.75, 0.25}},
                          std::pair{"P1", std::pair{0.5, 0.5}}}) {
        TsomRun theta0N256;
        TsomRun theta1N256;
        for (const PetscReal theta : {0.0, 1.0}) {
            for (const PetscInt n : {32, 64, 128, 256}) {
                const Mesh mesh = makeMesh(n, 1.0);
                const auto g0 = initialField(mesh, polyProfile);
                const auto psi0 = constantTimes(g0, 2.0);
                const auto v0 = g0;
                const PetscReal lambdaR = p.first == std::string_view("P1") ? 1.0 : 1.0 / 3.0;
                const auto c = tsomConstants(p.second.first, p.second.second, 1.0, lambdaR, theta);
                std::vector<PetscReal> sPsi, sV;
                buildSources(mesh, c, sPsi, sV);
                const TsomRun run = runTsom(mesh, c, bc, psi0, v0, {}, false, &sPsi, &sV, 1.0);
                const auto exactG = initialField(mesh, polyProfile);
                const auto exactPsi = constantTimes(exactG, 2.0 * PetscExpReal(-1.0));
                const auto exactV = constantTimes(exactG, PetscExpReal(-1.0));
                summary << p.first << ',' << theta << ',' << n << ','
                        << l2Difference(mesh, run.psi, exactPsi) << ','
                        << l2Difference(mesh, run.v, exactV) << '\n';
                if (n == 256 && theta == 0.0) theta0N256 = run;
                if (n == 256 && theta == 1.0) theta1N256 = run;
            }
        }
        writeProfile(out / (std::string(p.first) + "_profiles_N256.csv"), theta0N256.mesh, {
            {"v_theta0", theta0N256.v},
            {"v_theta1", theta1N256.v},
            {"abs_v_diff", [&] { auto d = theta0N256.v; for (std::size_t i=0;i<d.size();++i) d[i]=PetscAbsReal(theta0N256.v[i]-theta1N256.v[i]); return d; }()},
            {"psi_theta0", theta0N256.psi},
            {"psi_theta1", theta1N256.psi},
        });
    }
    PetscCallAbort(PETSC_COMM_WORLD, PetscFinalize());
    return 0;
}
