#include "PaperTSOMPsiV.hpp"

using namespace paper_tsompsiv;

int main(int argc, char** argv) {
    PetscCallAbort(PETSC_COMM_WORLD, PetscInitialize(&argc, &argv, nullptr, nullptr));
    const auto out = outputDir(__FILE__);
    const Mesh mesh = makeMesh(512, 5.0e-3);
    const auto psi0 = initialField(mesh, [](const PetscReal x) { return gaussianProfile(x); });
    const PetscReal vEquilibriumFraction = 0.75;
    const auto bc = dirichletNeumannBoundaries();

    const TsomRun tsomOriginal = runTsom(
        mesh,
        tsomConstants(0.75, 0.25, 2.0, 2.0 / 3.0, 1.0),
        bc,
        psi0,
        constantTimes(psi0, vEquilibriumFraction),
        {0.0, mesh.tf},
        true);
    const TsomRun tsomV0 = runTsom(
        mesh,
        tsomConstants(0.75, 0.25, 2.0, 2.0 / 3.0, 1.0),
        bc,
        psi0,
        constantTimes(psi0, 0.0),
        {0.0, mesh.tf},
        true);
    const TsomRun tsomFast100 = runTsom(
        mesh,
        tsomConstants(0.75, 0.25, 100.0, 100.0 / 3.0, 1.0),
        bc,
        psi0,
        constantTimes(psi0, vEquilibriumFraction),
        {0.0, mesh.tf},
        true);
    const TsomRun tsomFast500 = runTsom(
        mesh,
        tsomConstants(0.75, 0.25, 500.0, 500.0 / 3.0, 1.0),
        bc,
        psi0,
        constantTimes(psi0, vEquilibriumFraction),
        {0.0, mesh.tf},
        true);
    const ScalarRun bgc5e4 = runBGC(mesh, 5.0e-4, bc, psi0, {0.0, mesh.tf});
    const ScalarRun bgc1e3 = runBGC(mesh, 1.0e-3, bc, psi0, {0.0, mesh.tf});
    const ScalarRun fick = runFickian(mesh, 0.25, true, psi0, {0.0, mesh.tf});

    writeProfile(out / "profiles_final.csv", mesh, {
        {"tsom_original_psi", tsomOriginal.psi},
        {"tsom_v0_psi", tsomV0.psi},
        {"tsom_fast100_psi", tsomFast100.psi},
        {"tsom_fast500_psi", tsomFast500.psi},
        {"bgc_bv_5e-4", bgc5e4.phi},
        {"bgc_bv_1e-3", bgc1e3.phi},
        {"fickian_gamma_0p25", fick.phi},
    });
    auto writeTsomUv = [&](const std::string& name, const TsomRun& run) {
        writeProfile(out / (name + "_uv_final.csv"), mesh, {
            {"psi", run.psi},
            {"u", [&] { auto u = run.psi; for (std::size_t i=0;i<u.size();++i) u[i]-=run.v[i]; return u; }()},
            {"v", run.v},
        });
    };
    writeTsomUv("tsom_original", tsomOriginal);
    writeTsomUv("tsom_v0", tsomV0);
    writeTsomUv("tsom_fast100", tsomFast100);
    writeTsomUv("tsom_fast500", tsomFast500);

    {
        std::ofstream file = bgc::openOutputFile(out / "minima.csv");
        auto minU = [](const TsomRun& run) {
            PetscReal value = PETSC_MAX_REAL;
            for (std::size_t i = 0; i < run.psi.size(); ++i) {
                value = PetscMin(value, run.psi[i] - run.v[i]);
            }
            return value;
        };
        auto minV = [](const TsomRun& run) {
            return *std::min_element(run.v.begin(), run.v.end());
        };
        auto maxPsi = [](const std::vector<PetscReal>& values) {
            return *std::max_element(values.begin(), values.end());
        };
        const PetscReal fickPeak = maxPsi(fick.phi);
        auto writeTsomRow = [&](const std::string& name, const TsomRun& run) {
            file << name << ',' << maxPsi(run.psi) << ',' << run.minPsi << ','
                 << minU(run) << ',' << maxPsi(run.v) << ','
                 << run.mass0 << ',' << run.massFinal << ',' << variance(mesh, run.psi) << '\n';
        };
        file << "case,max_psi,min_psi,min_u,min_v,max_v,mass0,mass_final,variance,l2_vs_fick,peak_rel_diff_vs_fick\n";
        auto writeTsomRowWithFick = [&](const std::string& name, const TsomRun& run) {
            file << name << ',' << maxPsi(run.psi) << ',' << run.minPsi << ','
                 << minU(run) << ',' << minV(run) << ',' << maxPsi(run.v) << ','
                 << run.mass0 << ',' << run.massFinal << ',' << variance(mesh, run.psi) << ','
                 << l2Difference(mesh, run.psi, fick.phi) << ','
                 << (maxPsi(run.psi) - fickPeak) / fickPeak << '\n';
        };
        writeTsomRowWithFick("TSOM_original_equilibrium_Lc2", tsomOriginal);
        writeTsomRowWithFick("TSOM_V0_Lc2", tsomV0);
        writeTsomRowWithFick("TSOM_equilibrium_Lc100", tsomFast100);
        writeTsomRowWithFick("TSOM_equilibrium_Lc500", tsomFast500);
        auto writeScalarRow = [&](const std::string& name, const ScalarRun& run) {
            file << name << ',' << maxPsi(run.phi) << ',' << run.minPhi
                 << ",nan,nan,nan," << run.mass0 << ',' << run.massFinal << ','
                 << variance(mesh, run.phi) << ','
                 << l2Difference(mesh, run.phi, fick.phi) << ','
                 << (maxPsi(run.phi) - fickPeak) / fickPeak << '\n';
        };
        writeScalarRow("BGC_5e-4", bgc5e4);
        writeScalarRow("BGC_1e-3", bgc1e3);
        writeScalarRow("Fickian", fick);
    }
    {
        std::ofstream file = bgc::openOutputFile(out / "mass_history.csv");
        file << "case,time,mass_normalized\n";
        for (const auto& s : tsomOriginal.samples) file << "TSOM_original," << s.time << ',' << s.mass / tsomOriginal.mass0 << '\n';
        for (const auto& s : tsomV0.samples) file << "TSOM_V0," << s.time << ',' << s.mass / tsomV0.mass0 << '\n';
        for (const auto& s : tsomFast100.samples) file << "TSOM_Lc100," << s.time << ',' << s.mass / tsomFast100.mass0 << '\n';
        for (const auto& s : tsomFast500.samples) file << "TSOM_Lc500," << s.time << ',' << s.mass / tsomFast500.mass0 << '\n';
        for (const auto& s : bgc5e4.samples) file << "BGC_5e-4," << s.time << ',' << s.mass / bgc5e4.mass0 << '\n';
        for (const auto& s : bgc1e3.samples) file << "BGC_1e-3," << s.time << ',' << s.mass / bgc1e3.mass0 << '\n';
        for (const auto& s : fick.samples) file << "Fickian," << s.time << ',' << s.mass / fick.mass0 << '\n';
    }
    PetscCallAbort(PETSC_COMM_WORLD, PetscFinalize());
    return 0;
}
