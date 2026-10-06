#include <petsc.h>

#include <bgclib/BGCLib.hpp>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numbers>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct MMS4Input {
    PetscReal tf {5.0};
    PetscReal dt {1.0e-4};
    PetscInt nx {512};
    std::vector<PetscReal> thetaList {0.0, 0.5, 1.0};
    std::filesystem::path outputDir {"Saida"};
    bool verbose {true};
    bool debug {false};
};

struct EnergyRecord {
    PetscReal theta {0.0};
    PetscReal initialEnergy {0.0};
    PetscReal finalEnergy {0.0};
    PetscReal minDelta {0.0};
    PetscReal maxDelta {0.0};
    PetscInt positiveDeltaCount {0};
    PetscReal initialMass {0.0};
    PetscReal finalMass {0.0};
    KSPConvergedReason reason {KSP_CONVERGED_ITERATING};
    PetscInt iterations {0};
};

std::filesystem::path caseDirectory() {
    return bgc::caseDirectoryFromSource(__FILE__);
}

MMS4Input readInput(const std::filesystem::path& dataDir) {
    MMS4Input input;
    const std::unordered_map<std::string, std::string> data =
        bgc::readKeyValueFile(dataDir / "simulation.dat");
    if (data.contains("tf")) {
        input.tf = bgc::parseReal(data.at("tf"));
    }
    if (data.contains("dt")) {
        input.dt = bgc::parseReal(data.at("dt"));
    }
    if (data.contains("nx")) {
        input.nx = bgc::parseInt(data.at("nx"));
    }
    if (data.contains("theta_list")) {
        input.thetaList = bgc::parseRealList(data.at("theta_list"));
    }
    if (data.contains("output_dir")) {
        input.outputDir = bgc::trim(data.at("output_dir"));
    }
    if (data.contains("verbose")) {
        input.verbose = bgc::parseBool(data.at("verbose"));
    }
    if (data.contains("debug")) {
        input.debug = bgc::parseBool(data.at("debug"));
    }
    return input;
}

bgc::models::tsompsiv::Constants defaultConstants() {
    bgc::models::tsompsiv::Constants constants;
    constants.alpha = 0.75;
    constants.rho = 0.25;
    constants.theta = 0.5;
    constants.lambdaC = 1.0;
    constants.lambdaR = 1.0 / 3.0;
    constants.vBoundary = {
        .type = bgc::models::tsompsiv::VBoundaryCondition::ScaledPsi,
        .sc = 0.25,
    };
    return constants;
}

bgc::models::tsompsiv::Constants readTSOMPsiVConstants(const std::filesystem::path& dataDir) {
    bgc::models::tsompsiv::Constants constants = defaultConstants();
    const std::unordered_map<std::string, std::string> data =
        bgc::readKeyValueFile(dataDir / "simulation.dat");

    if (data.contains("alpha")) {
        constants.alpha = bgc::parseReal(data.at("alpha"));
    }
    if (data.contains("rho")) {
        constants.rho = bgc::parseReal(data.at("rho"));
    }
    if (data.contains("lambda_c")) {
        constants.lambdaC = bgc::parseReal(data.at("lambda_c"));
    }
    if (data.contains("lambda_r")) {
        constants.lambdaR = bgc::parseReal(data.at("lambda_r"));
    }
    if (data.contains("v_boundary")) {
        constants.vBoundary.type =
            bgc::models::tsompsiv::parseVBoundaryCondition(bgc::trim(data.at("v_boundary")));
    }
    if (data.contains("sc")) {
        constants.vBoundary.sc = bgc::parseReal(data.at("sc"));
    } else {
        constants.vBoundary.sc = constants.rho / (constants.alpha + constants.rho);
    }
    return constants;
}

bgc::BoundarySet makeNoFluxBoundaries() {
    return {
        .west = {
            .conditions = {
                bgc::BoundaryCondition::neumann(0.0),
                bgc::BoundaryCondition::neumann(0.0),
            },
        },
        .east = {
            .conditions = {
                bgc::BoundaryCondition::neumann(0.0),
                bgc::BoundaryCondition::neumann(0.0),
            },
        },
    };
}

bgc::Grid1D makeGrid(const PetscInt nx) {
    return {
        .nx = nx,
        .length = 1.0,
        .x0 = 0.0,
    };
}

bgc::TimeConfig makeTime(const MMS4Input& input) {
    return {
        .dt = input.dt,
        .finalTime = input.tf,
        .initialTime = 0.0,
    };
}

std::vector<PetscReal> buildInitialState(const MMS4Input& input) {
    const PetscReal h = 1.0 / static_cast<PetscReal>(input.nx);
    std::vector<PetscReal> state(static_cast<std::size_t>(2 * input.nx), 0.0);
    for (PetscInt i = 0; i < input.nx; ++i) {
        const PetscReal x = (static_cast<PetscReal>(i) + 0.5) * h;
        const PetscReal psi = 1.0 + PetscSinReal(std::numbers::pi_v<PetscReal> * x);
        const PetscReal s = PetscSinReal(0.5 * std::numbers::pi_v<PetscReal> * x);
        const PetscReal v = s * s;
        state[static_cast<std::size_t>(i)] = psi;
        state[static_cast<std::size_t>(input.nx + i)] = v;
    }
    return state;
}

PetscReal computeMass(const MMS4Input& input, const std::vector<PetscReal>& state) {
    const PetscReal h = 1.0 / static_cast<PetscReal>(input.nx);
    PetscReal sum = 0.0;
    for (PetscInt i = 0; i < input.nx; ++i) {
        const auto psiIndex = static_cast<std::size_t>(i);
        sum += state[psiIndex];
    }
    return h * sum;
}

PetscReal computeFreeEnergy(const MMS4Input& input,
                            const bgc::models::tsompsiv::Constants& constants,
                            const std::vector<PetscReal>& state) {
    const PetscReal h = 1.0 / static_cast<PetscReal>(input.nx);
    const PetscReal eta = constants.rho / constants.alpha;
    PetscReal sum = 0.0;
    for (PetscInt i = 0; i < input.nx; ++i) {
        const auto psiIndex = static_cast<std::size_t>(i);
        const auto vIndex = static_cast<std::size_t>(input.nx + i);
        const PetscReal u = state[psiIndex] - state[vIndex];
        sum += u * u + eta * state[vIndex] * state[vIndex];
    }
    return 0.5 * h * sum;
}

bgc::DiscreteOperator buildTSOMPsiVOperator(const MMS4Input& input,
                                        const bgc::models::tsompsiv::Constants& constants) {
    const bgc::Grid1D grid = makeGrid(input.nx);
    const auto coefficients = bgc::models::tsompsiv::computeCoefficients(
        grid,
        makeTime(input),
        constants,
        makeNoFluxBoundaries());
    return bgc::models::tsompsiv::buildOperator(grid, coefficients);
}

std::vector<PetscReal> buildRHS(const MMS4Input& input,
                                const std::vector<PetscReal>& previousState) {
    const PetscReal h = 1.0 / static_cast<PetscReal>(input.nx);
    const PetscReal hdt = h / input.dt;
    std::vector<PetscReal> rhs(static_cast<std::size_t>(2 * input.nx), 0.0);
    for (PetscInt i = 0; i < 2 * input.nx; ++i) {
        rhs[static_cast<std::size_t>(i)] = hdt * previousState[static_cast<std::size_t>(i)];
    }
    return rhs;
}

EnergyRecord runTheta(const MMS4Input& input,
                      bgc::models::tsompsiv::Constants constants,
                      const PetscReal theta,
                      std::ofstream& history) {
    constants.theta = theta;
    if (!bgc::ModelTraits<bgc::models::tsompsiv::Tag>::constantsAreValid(constants)) {
        throw std::runtime_error("Invalid TSOMPsiV constants in MMS4TSOMPsiV.");
    }

    const auto op = buildTSOMPsiVOperator(input, constants);
    bgc::models::bgc::MMSLinearSolver solver(op, 0.0);
    std::vector<PetscReal> state = buildInitialState(input);
    const PetscInt nTimes = static_cast<PetscInt>(input.tf / input.dt + 0.5);

    KSPConvergedReason reason = KSP_CONVERGED_ITERATING;
    PetscInt iterations = 0;
    PetscReal previousEnergy = computeFreeEnergy(input, constants, state);
    EnergyRecord record {
        .theta = theta,
        .initialEnergy = previousEnergy,
        .finalEnergy = previousEnergy,
        .minDelta = 0.0,
        .maxDelta = 0.0,
        .positiveDeltaCount = 0,
        .initialMass = computeMass(input, state),
        .finalMass = computeMass(input, state),
        .reason = reason,
        .iterations = iterations,
    };

    history << theta << ",0,0.0000000000000000e+00,"
            << previousEnergy << ",0.0000000000000000e+00,"
            << record.initialMass << '\n';

    for (PetscInt step = 1; step <= nTimes; ++step) {
        const std::vector<PetscReal> rhs = buildRHS(input, state);
        PetscCallAbort(PETSC_COMM_WORLD, solver.solve(rhs, state, reason, iterations));

        const PetscReal time = static_cast<PetscReal>(step) * input.dt;
        const PetscReal energy = computeFreeEnergy(input, constants, state);
        const PetscReal delta = energy - previousEnergy;
        const PetscReal mass = computeMass(input, state);

        record.minDelta = PetscMin(record.minDelta, delta);
        record.maxDelta = PetscMax(record.maxDelta, delta);
        if (delta > 1.0e-12) {
            ++record.positiveDeltaCount;
        }

        history << theta << ',' << step << ',' << time << ',' << energy << ','
                << delta << ',' << mass << '\n';
        previousEnergy = energy;
        record.finalEnergy = energy;
        record.finalMass = mass;
        record.reason = reason;
        record.iterations = iterations;
    }

    return record;
}

void writeSummary(const std::filesystem::path& path,
                  const std::vector<EnergyRecord>& records,
                  const MMS4Input& input,
                  const bgc::models::tsompsiv::Constants& constants) {
    std::ofstream file = bgc::openOutputFile(path);
    file << std::scientific << std::setprecision(16);
    file << "theta,nx,dt,tf,alpha,rho,lambdaC,lambdaR,vBoundary,sc,"
            "initialEnergy,finalEnergy,totalDelta,minDelta,maxDelta,"
            "positiveDeltaCount,initialMass,finalMass,massDrift,"
            "ksp_reason,ksp_iterations,status\n";
    for (const EnergyRecord& r : records) {
        file << r.theta << ',' << input.nx << ',' << input.dt << ',' << input.tf << ','
             << constants.alpha << ',' << constants.rho << ',' << constants.lambdaC << ','
             << constants.lambdaR << ','
             << bgc::models::tsompsiv::toString(constants.vBoundary.type) << ','
             << constants.vBoundary.sc << ','
             << r.initialEnergy << ',' << r.finalEnergy << ','
             << r.finalEnergy - r.initialEnergy << ','
             << r.minDelta << ',' << r.maxDelta << ','
             << r.positiveDeltaCount << ','
             << r.initialMass << ',' << r.finalMass << ','
             << r.finalMass - r.initialMass << ','
             << static_cast<int>(r.reason) << ',' << r.iterations << ','
             << bgc::convergenceStatus(r.reason) << '\n';
    }
}

} // namespace

int main(int argc, char** argv) {
    PetscCallAbort(PETSC_COMM_WORLD,
                   PetscInitialize(&argc,
                                   &argv,
                                   nullptr,
                                   "MMS4TSOMPsiV -- TSOMPsiV free-energy dissipation experiment"));

    PetscInt exitCode = 0;
    try {
        const std::filesystem::path caseDir = caseDirectory();
        const std::filesystem::path dataDir = caseDir / "Dados";
        const MMS4Input input = readInput(dataDir);
        const bgc::models::tsompsiv::Constants constants = readTSOMPsiVConstants(dataDir);
        const std::filesystem::path outputDir =
            bgc::resolveOutputDirectory(caseDir, input.outputDir);
        bgc::ensureDirectory(outputDir);

        const auto d = bgc::models::tsompsiv::computeDiffusionCoefficients(constants);
        const std::string vBoundaryText {
            bgc::models::tsompsiv::toString(constants.vBoundary.type)
        };
        PetscCallAbort(PETSC_COMM_WORLD, bgc::models::bgc::printCpuUsageSummary());
        PetscPrintf(PETSC_COMM_WORLD,
                    "\nMMS4TSOMPsiV -- TSOMPsiV free-energy dissipation check\n"
                    "  Psi(x,0) = 1 + sin(pi*x)\n"
                    "  V(x,0)   = sin^2(pi*x/2)\n"
                    "  nx=%d dt=%.6e tf=%.6e\n"
                    "  alpha=%.6e rho=%.6e lambdaC=%.6e lambdaR=%.6e\n"
                    "  diffusion at default theta: d11=%.6e d12=%.6e d21=%.6e d22=%.6e\n"
                    "  V boundary: %s sc=%.6e\n"
                    "  output: %s\n",
                    static_cast<int>(input.nx),
                    static_cast<PetscReal>(input.dt),
                    static_cast<PetscReal>(input.tf),
                    static_cast<PetscReal>(constants.alpha),
                    static_cast<PetscReal>(constants.rho),
                    static_cast<PetscReal>(constants.lambdaC),
                    static_cast<PetscReal>(constants.lambdaR),
                    static_cast<PetscReal>(d.d11),
                    static_cast<PetscReal>(d.d12),
                    static_cast<PetscReal>(d.d21),
                    static_cast<PetscReal>(d.d22),
                    vBoundaryText.c_str(),
                    static_cast<PetscReal>(constants.vBoundary.sc),
                    outputDir.string().c_str());

        std::vector<EnergyRecord> records;
        if (bgc::models::bgc::isWorldRankZero()) {
            std::ofstream history = bgc::openOutputFile(outputDir / "mms4tsompsiv_energy_history.csv");
            history << std::scientific << std::setprecision(16);
            history << "theta,step,time,energy,deltaEnergy,mass\n";
            for (const PetscReal theta : input.thetaList) {
                records.push_back(runTheta(input, constants, theta, history));
            }
            writeSummary(outputDir / "mms4tsompsiv_energy_summary.csv", records, input, constants);
        } else {
            std::ofstream sink;
            for (const PetscReal theta : input.thetaList) {
                records.push_back(runTheta(input, constants, theta, sink));
            }
        }
    } catch (const std::exception& ex) {
        PetscPrintf(PETSC_COMM_WORLD, "\nERROR: %s\n", ex.what());
        exitCode = 1;
    }

    PetscCallAbort(PETSC_COMM_WORLD, PetscFinalize());
    return exitCode;
}
