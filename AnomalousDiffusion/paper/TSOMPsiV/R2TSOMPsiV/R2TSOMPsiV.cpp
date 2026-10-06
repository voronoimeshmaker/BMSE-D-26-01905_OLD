#include "PaperTSOMPsiV.hpp"

using namespace paper_tsompsiv;

int main(int argc, char** argv) {
    PetscCallAbort(PETSC_COMM_WORLD, PetscInitialize(&argc, &argv, nullptr, nullptr));
    const auto out = outputDir(__FILE__);
    const Mesh mesh = makeMesh(256, 5.0);
    const auto psi0 = initialField(mesh, sin2Profile);
    const auto constants = tsomConstants(0.75, 0.25, 1.0, 1.0 / 3.0, 1.0);
    const auto bc = noFluxBoundaries();
    const std::vector<PetscReal> times {0.0, 0.05, 0.1, 0.5, 1.0, 5.0};

    const std::vector<std::pair<std::string, PetscReal>> cases {
        {"R2a", 0.0}, {"R2b", 1.0}, {"R2c", 0.25},
    };
    std::ofstream ratio = bgc::openOutputFile(out / "center_ratio.csv");
    std::ofstream energy = bgc::openOutputFile(out / "energy.csv");
    std::ofstream massFile = bgc::openOutputFile(out / "mass.csv");
    ratio << "case,time,v_over_psi_center\n";
    energy << "case,time,free_energy_normalized\n";
    massFile << "case,time,mass\n";
    for (const auto& [name, frac] : cases) {
        const auto v0 = constantTimes(psi0, frac);
        const TsomRun run = runTsom(mesh, constants, bc, psi0, v0, times, name == "R2a");
        for (const auto& s : run.samples) {
            ratio << name << ',' << s.time << ',' << s.centerRatio << '\n';
            energy << name << ',' << s.time << ',' << s.energy / run.energy0 << '\n';
            massFile << name << ',' << s.time << ',' << s.mass << '\n';
        }
        if (name == "R2a") {
            for (const auto& s : run.samples) {
                auto u = s.psi;
                for (std::size_t i = 0; i < u.size(); ++i) u[i] -= s.v[i];
                writeProfile(out / ("profiles_R2a_t" + std::to_string(static_cast<int>(100*s.time)) + ".csv"),
                             mesh, {{"u", u}, {"v", s.v}, {"psi", s.psi}});
            }
        }
    }
    PetscCallAbort(PETSC_COMM_WORLD, PetscFinalize());
    return 0;
}
