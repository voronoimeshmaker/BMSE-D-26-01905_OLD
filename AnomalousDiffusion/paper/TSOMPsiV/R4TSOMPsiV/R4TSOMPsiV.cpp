#include "PaperTSOMPsiV.hpp"

using namespace paper_tsompsiv;

int main(int argc, char** argv) {
    PetscCallAbort(PETSC_COMM_WORLD, PetscInitialize(&argc, &argv, nullptr, nullptr));
    const auto out = outputDir(__FILE__);
    const Mesh mesh = makeMesh(256, 0.5);
    const auto psi0 = initialField(mesh, sin2Profile);
    const auto v0 = constantTimes(psi0, 0.25);
    const auto bc = noFluxBoundaries();
    const ScalarRun fick = runFickian(mesh, 0.25, false, psi0, {0.0, mesh.tf});

    std::ofstream summary = bgc::openOutputFile(out / "fickian_limit.csv");
    summary << "case,lambda_c,lambda_r,l2_psi_minus_fick,center_ratio_final\n";
    std::vector<std::pair<std::string, std::vector<PetscReal>>> fields {{"fickian", fick.phi}};
    for (const auto& [name, lc] : {std::pair{"R4a", 0.1}, std::pair{"R4b", 1.0},
                                   std::pair{"R4c", 10.0}, std::pair{"R4d", 100.0}}) {
        const auto c = tsomConstants(0.75, 0.25, lc, lc / 3.0, 1.0);
        const TsomRun run = runTsom(mesh, c, bc, psi0, v0, {0.0, mesh.tf});
        summary << name << ',' << lc << ',' << lc / 3.0 << ','
                << l2Difference(mesh, run.psi, fick.phi) << ','
                << run.samples.back().centerRatio << '\n';
        fields.push_back({std::string(name) + "_psi", run.psi});
    }
    writeProfile(out / "profiles_final.csv", mesh, fields);
    PetscCallAbort(PETSC_COMM_WORLD, PetscFinalize());
    return 0;
}
