#include "PaperTSOMPsiV.hpp"

using namespace paper_tsompsiv;

int main(int argc, char** argv) {
    PetscCallAbort(PETSC_COMM_WORLD, PetscInitialize(&argc, &argv, nullptr, nullptr));
    const auto out = outputDir(__FILE__);
    const Mesh mesh = makeMesh(256, 5.0);
    const auto psi0 = initialField(mesh, sin2Profile);
    const auto constants = tsomConstants(0.75, 0.25, 1.0, 1.0 / 3.0, 1.0);
    const auto bc = noFluxBoundaries();
    std::vector<PetscReal> times;
    for (PetscReal t = 0.0; t <= 5.0 + 1.0e-12; t += 0.01) times.push_back(t);
    const std::vector<PetscReal> profileTimes {0.0, 0.5, 1.0, 2.0, 5.0};

    std::ofstream ratio = bgc::openOutputFile(out / "center_ratio_and_exchange.csv");
    std::ofstream varianceFile = bgc::openOutputFile(out / "variance.csv");
    std::ofstream energy = bgc::openOutputFile(out / "energy.csv");
    constexpr PetscReal vEquilibriumFraction = 0.75;
    ratio << "case,time,v_over_psi_center,log_abs_ratio_minus_v_equilibrium\n";
    varianceFile << "case,time,variance\n";
    energy << "case,time,free_energy\n";
    for (const auto& [name, frac] : {std::pair{"R5a", 0.0}, std::pair{"R5b", 0.5},
                                     std::pair{"R5c", 1.0}}) {
        const TsomRun run = runTsom(mesh, constants, bc, psi0, constantTimes(psi0, frac), times);
        for (const auto& s : run.samples) {
            const PetscReal diff = PetscAbsReal(s.centerRatio - vEquilibriumFraction);
            ratio << name << ',' << s.time << ',' << s.centerRatio << ','
                  << (diff > 0.0 ? PetscLogReal(diff) : -PETSC_INFINITY) << '\n';
            varianceFile << name << ',' << s.time << ',' << s.variance << '\n';
            energy << name << ',' << s.time << ',' << s.energy << '\n';
        }
        if (name == "R5a") {
            const TsomRun profiles = runTsom(mesh, constants, bc, psi0, constantTimes(psi0, frac),
                                            profileTimes, true);
            for (const auto& s : profiles.samples) {
                auto ratioField = s.v;
                for (std::size_t i = 0; i < ratioField.size(); ++i) ratioField[i] /= s.psi[i];
                writeProfile(out / ("ratio_profile_R5a_t" + std::to_string(static_cast<int>(10*s.time)) + ".csv"),
                             mesh, {{"v_over_psi", ratioField}});
            }
        }
    }
    PetscCallAbort(PETSC_COMM_WORLD, PetscFinalize());
    return 0;
}
