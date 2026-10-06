#include "PaperTSOMPsiV.hpp"

#include <limits>

using namespace paper_tsompsiv;

namespace {

struct Scenario {
    std::string name;
    PetscReal alpha {0.75};
    PetscReal rho {0.25};
    PetscReal lambdaC {1.0};
};

PetscReal firstTimeAtFraction(const std::vector<Sample>& samples,
                              const PetscReal equilibrium,
                              const PetscReal fraction) {
    const PetscReal threshold = fraction * equilibrium;
    for (const Sample& sample : samples) {
        if (sample.centerRatio >= threshold) {
            return sample.time;
        }
    }
    return std::numeric_limits<PetscReal>::quiet_NaN();
}

PetscReal minU(const TsomRun& run) {
    PetscReal value = PETSC_MAX_REAL;
    for (std::size_t i = 0; i < run.psi.size(); ++i) {
        value = PetscMin(value, run.psi[i] - run.v[i]);
    }
    return value;
}

void runScenario(const Scenario& scenario,
                 const Mesh& mesh,
                 const std::vector<PetscReal>& psi0,
                 const std::vector<PetscReal>& sampleTimes,
                 std::ofstream& ratio,
                 std::ofstream& summary) {
    const PetscReal lambdaR = scenario.lambdaC * scenario.rho / scenario.alpha;
    const PetscReal vEquilibrium = scenario.alpha / (scenario.alpha + scenario.rho);
    const auto constants =
        tsomConstants(scenario.alpha, scenario.rho, scenario.lambdaC, lambdaR, 1.0);
    const TsomRun run =
        runTsom(mesh, constants, noFluxBoundaries(), psi0, constantTimes(psi0, 0.0), sampleTimes);

    for (const Sample& sample : run.samples) {
        ratio << scenario.name << ',' << scenario.alpha << ',' << scenario.rho << ','
              << scenario.lambdaC << ',' << lambdaR << ',' << vEquilibrium << ','
              << sample.time << ',' << sample.centerRatio << ','
              << sample.energy / run.energy0 << ',' << sample.mass << '\n';
    }

    summary << scenario.name << ',' << scenario.alpha << ',' << scenario.rho << ','
            << scenario.lambdaC << ',' << lambdaR << ',' << lambdaR + scenario.lambdaC << ','
            << vEquilibrium << ',' << run.samples.back().centerRatio << ','
            << firstTimeAtFraction(run.samples, vEquilibrium, 0.95) << ','
            << run.minPsi << ',' << minU(run) << ','
            << run.energyFinal / run.energy0 << ',' << run.massFinal << '\n';
}

} // namespace

int main(int argc, char** argv) {
    PetscCallAbort(PETSC_COMM_WORLD, PetscInitialize(&argc, &argv, nullptr, nullptr));

    const auto out = outputDir(__FILE__);
    const Mesh mesh = makeMesh(256, 5.0);
    const auto psi0 = initialField(mesh, sin2Profile);
    std::vector<PetscReal> sampleTimes;
    for (PetscReal t = 0.0; t <= 5.0 + 1.0e-12; t += 0.01) {
        sampleTimes.push_back(t);
    }

    std::ofstream ratio = bgc::openOutputFile(out / "partition_sensitivity.csv");
    std::ofstream summary = bgc::openOutputFile(out / "summary.csv");
    ratio << "scenario,alpha,rho,lambda_c,lambda_r,v_equilibrium,time,"
             "v_over_psi_center,free_energy_normalized,mass\n";
    summary << "scenario,alpha,rho,lambda_c,lambda_r,lambda_exchange,"
               "v_equilibrium,final_center_ratio,t95,min_psi,min_u,"
               "energy_ratio_final,mass_final\n";

    for (const Scenario& scenario : {
             Scenario {"alpha090_rho010_Lc1", 0.90, 0.10, 1.0},
             Scenario {"alpha075_rho025_Lc1", 0.75, 0.25, 1.0},
             Scenario {"alpha050_rho050_Lc1", 0.50, 0.50, 1.0},
             Scenario {"alpha025_rho075_Lc1", 0.25, 0.75, 1.0},
         }) {
        runScenario(scenario, mesh, psi0, sampleTimes, ratio, summary);
    }

    for (const Scenario& scenario : {
             Scenario {"alpha075_rho025_Lc0p5", 0.75, 0.25, 0.5},
             Scenario {"alpha075_rho025_Lc1", 0.75, 0.25, 1.0},
             Scenario {"alpha075_rho025_Lc2", 0.75, 0.25, 2.0},
             Scenario {"alpha075_rho025_Lc10", 0.75, 0.25, 10.0},
         }) {
        runScenario(scenario, mesh, psi0, sampleTimes, ratio, summary);
    }

    PetscCallAbort(PETSC_COMM_WORLD, PetscFinalize());
    return 0;
}
