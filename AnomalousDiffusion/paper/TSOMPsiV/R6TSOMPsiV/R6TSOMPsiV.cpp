#include "PaperTSOMPsiV.hpp"

using namespace paper_tsompsiv;

int main(int argc, char** argv) {
    PetscCallAbort(PETSC_COMM_WORLD, PetscInitialize(&argc, &argv, nullptr, nullptr));
    const auto out = outputDir(__FILE__);
    const Mesh mesh = makeMesh(512, 5.0e-3);
    const auto psi0 = initialField(mesh, [](const PetscReal x) { return gaussianProfile(x); });
    const auto bc = dirichletNeumannBoundaries();
    std::vector<PetscReal> times;
    for (PetscInt i = 0; i <= 50; ++i) times.push_back(mesh.tf * static_cast<PetscReal>(i) / 50.0);
    const ScalarRun fick = runFickian(mesh, 0.25, true, psi0, times);

    std::ofstream sigma = bgc::openOutputFile(out / "variance_history.csv");
    std::ofstream summary = bgc::openOutputFile(out / "summary.csv");
    sigma << "case,time,variance\n";
    summary << "case,lambda_c,lambda_r,variance_final,variance_ratio_to_fick,min_psi,energy_ratio\n";
    for (const auto& s : fick.samples) sigma << "Fickian," << s.time << ',' << s.variance << '\n';
    summary << "Fickian,0,0," << fick.varianceFinal << ",1," << fick.minPhi << ",nan\n";

    for (const auto& [name, lc] : {std::pair{"R6_T1", 0.5}, std::pair{"R6_T2", 2.0},
                                   std::pair{"R6_T3", 10.0}}) {
        const auto c = tsomConstants(0.75, 0.25, lc, lc / 3.0, 1.0);
        const auto v0 = constantTimes(psi0, 0.25);
        const TsomRun run = runTsom(mesh, c, bc, psi0, v0, times);
        for (const auto& s : run.samples) sigma << name << ',' << s.time << ',' << s.variance << '\n';
        summary << name << ',' << lc << ',' << lc / 3.0 << ',' << variance(mesh, run.psi) << ','
                << variance(mesh, run.psi) / fick.varianceFinal << ',' << run.minPsi << ','
                << run.energyFinal / run.energy0 << '\n';
    }
    PetscCallAbort(PETSC_COMM_WORLD, PetscFinalize());
    return 0;
}
