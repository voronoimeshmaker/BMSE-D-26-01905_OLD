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

using InputData = bgc::models::bgc::MMSInputData;
using MeshTimeData = bgc::models::bgc::MMSMeshTimeData;

struct MassRecord {
    PetscInt nx {0};
    PetscReal h {0.0};
    PetscReal dt {0.0};
    PetscInt nTimes {0};
    PetscReal mass0 {0.0};
    PetscReal massFinal {0.0};
    PetscReal maxCenteredResidual {0.0};
    PetscReal maxStepDrift {0.0};
    PetscReal finalDrift {0.0};
    KSPConvergedReason reason {KSP_CONVERGED_ITERATING};
    PetscInt iterations {0};
};

std::filesystem::path caseDirectory() {
    return bgc::caseDirectoryFromSource(__FILE__);
}

InputData defaultInput() {
    return {
        .tf = 1.0,
        .cdt = 1.024e-1,
        .nxList = {8, 16, 32, 64, 128, 256, 512},
        .bvList = {1.0},
        .outputDir = "Saida",
        .verbose = true,
        .debug = false,
    };
}

bgc::models::tsompsiv::Constants defaultConstants() {
    bgc::models::tsompsiv::Constants constants;
    constants.alpha = 0.75;
    constants.rho = 0.25;
    constants.theta = 1.0;
    constants.lambdaC = 1.0;
    constants.lambdaR = 1.0 / 3.0;
    constants.vBoundary = {
        .type = bgc::models::tsompsiv::VBoundaryCondition::ConstantGradient,
        .sc = 0.0,
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
    if (data.contains("theta")) {
        constants.theta = bgc::parseReal(data.at("theta"));
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
    }

    if (!bgc::ModelTraits<bgc::models::tsompsiv::Tag>::constantsAreValid(constants)) {
        throw std::runtime_error("Invalid TSOMPsiV constants in MMS3TSOMPsiV simulation.dat.");
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

bgc::Grid1D makeGrid(const MeshTimeData& mesh) {
    return {
        .nx = mesh.nx,
        .length = 1.0,
        .x0 = 0.0,
    };
}

bgc::TimeConfig makeTime(const MeshTimeData& mesh, const InputData& input) {
    return {
        .dt = mesh.dt,
        .finalTime = input.tf,
        .initialTime = 0.0,
    };
}

PetscReal psiInitial(const PetscReal x) {
    const PetscReal s = PetscSinReal(std::numbers::pi_v<PetscReal> * x);
    return s * s;
}

std::vector<PetscReal> buildInitialState(const MeshTimeData& mesh,
                                         const bgc::models::tsompsiv::Constants& constants) {
    const PetscReal sc = constants.rho / (constants.alpha + constants.rho);
    const PetscReal vFraction = 1.0 - sc;
    std::vector<PetscReal> state(static_cast<std::size_t>(2 * mesh.nx), 0.0);
    for (PetscInt i = 0; i < mesh.nx; ++i) {
        const PetscReal x = (static_cast<PetscReal>(i) + 0.5) * mesh.h;
        const PetscReal psi = psiInitial(x);
        state[static_cast<std::size_t>(i)] = psi;
        state[static_cast<std::size_t>(mesh.nx + i)] = vFraction * psi;
    }
    return state;
}

PetscReal computeMass(const MeshTimeData& mesh, const std::vector<PetscReal>& state) {
    PetscReal sum = 0.0;
    for (PetscInt i = 0; i < mesh.nx; ++i) {
        const auto psiIndex = static_cast<std::size_t>(i);
        sum += state[psiIndex];
    }
    return mesh.h * sum;
}

bgc::DiscreteOperator buildTSOMPsiVOperator(const MeshTimeData& mesh,
                                        const InputData& input,
                                        const bgc::models::tsompsiv::Constants& constants) {
    const bgc::Grid1D grid = makeGrid(mesh);
    const auto coefficients = bgc::models::tsompsiv::computeCoefficients(
        grid,
        makeTime(mesh, input),
        constants,
        makeNoFluxBoundaries());
    return bgc::models::tsompsiv::buildOperator(grid, coefficients);
}

std::vector<PetscReal> buildRHS(const MeshTimeData& mesh,
                                const std::vector<PetscReal>& previousState) {
    const PetscReal hdt = mesh.h / mesh.dt;
    std::vector<PetscReal> rhs(static_cast<std::size_t>(2 * mesh.nx), 0.0);
    for (PetscInt i = 0; i < 2 * mesh.nx; ++i) {
        rhs[static_cast<std::size_t>(i)] = hdt * previousState[static_cast<std::size_t>(i)];
    }
    return rhs;
}

MassRecord runMesh(const MeshTimeData& mesh,
                   const InputData& input,
                   const bgc::models::tsompsiv::Constants& constants,
                   std::vector<PetscReal>& finalState) {
    const auto op = buildTSOMPsiVOperator(mesh, input, constants);
    bgc::models::bgc::MMSLinearSolver solver(op, 0.0);

    finalState = buildInitialState(mesh, constants);
    std::vector<PetscReal> masses;
    masses.reserve(static_cast<std::size_t>(mesh.nTimes + 1));
    masses.push_back(computeMass(mesh, finalState));

    KSPConvergedReason reason = KSP_CONVERGED_ITERATING;
    PetscInt iterations = 0;
    for (PetscInt step = 1; step <= mesh.nTimes; ++step) {
        const std::vector<PetscReal> rhs = buildRHS(mesh, finalState);
        PetscCallAbort(PETSC_COMM_WORLD, solver.solve(rhs, finalState, reason, iterations));
        masses.push_back(computeMass(mesh, finalState));
    }

    PetscReal maxCenteredResidual = 0.0;
    for (std::size_t i = 1; i + 1 < masses.size(); ++i) {
        const PetscReal residual = (masses[i + 1] - masses[i - 1]) / (2.0 * mesh.dt);
        maxCenteredResidual = PetscMax(maxCenteredResidual, PetscAbsReal(residual));
    }

    PetscReal maxStepDrift = 0.0;
    for (const PetscReal mass : masses) {
        maxStepDrift = PetscMax(maxStepDrift, PetscAbsReal(mass - masses.front()));
    }

    return {
        .nx = mesh.nx,
        .h = mesh.h,
        .dt = mesh.dt,
        .nTimes = mesh.nTimes,
        .mass0 = masses.front(),
        .massFinal = masses.back(),
        .maxCenteredResidual = maxCenteredResidual,
        .maxStepDrift = maxStepDrift,
        .finalDrift = masses.back() - masses.front(),
        .reason = reason,
        .iterations = iterations,
    };
}

void writeFields(const std::filesystem::path& path,
                 const MeshTimeData& mesh,
                 const std::vector<PetscReal>& state) {
    std::ofstream file = bgc::openOutputFile(path);
    file << std::scientific << std::setprecision(16);
    file << "# P x psi v\n";
    for (PetscInt i = 0; i < mesh.nx; ++i) {
        const auto psiIndex = static_cast<std::size_t>(i);
        const auto vIndex = static_cast<std::size_t>(mesh.nx + i);
        const PetscReal x = (static_cast<PetscReal>(i) + 0.5) * mesh.h;
        file << i + 1 << ' ' << x << ' ' << state[psiIndex] << ' ' << state[vIndex] << '\n';
    }
}

void writeMassSummary(const std::filesystem::path& path,
                      const std::vector<MassRecord>& records) {
    std::ofstream file = bgc::openOutputFile(path);
    file << std::scientific << std::setprecision(16);
    file << "nx,h,dt,nTimes,mass0,massFinal,finalDrift,maxStepDrift,"
            "maxCenteredResidual,ksp_reason,ksp_iterations,status\n";
    for (const MassRecord& r : records) {
        file << r.nx << ',' << r.h << ',' << r.dt << ',' << r.nTimes << ','
             << r.mass0 << ',' << r.massFinal << ',' << r.finalDrift << ','
             << r.maxStepDrift << ',' << r.maxCenteredResidual << ','
             << static_cast<int>(r.reason) << ',' << r.iterations << ','
             << bgc::convergenceStatus(r.reason) << '\n';
    }
}

void writeSetup(const std::filesystem::path& path,
                const std::vector<MassRecord>& records,
                const bgc::models::tsompsiv::Constants& constants) {
    std::ofstream file = bgc::openOutputFile(path);
    file << std::scientific << std::setprecision(16);
    file << "alpha,rho,theta,lambdaC,lambdaR,vBoundary,sc,fStar,nx,h,dt,nTimes\n";
    const PetscReal fStar = constants.rho / (constants.alpha + constants.rho);
    for (const MassRecord& r : records) {
        file << constants.alpha << ',' << constants.rho << ',' << constants.theta << ','
             << constants.lambdaC << ',' << constants.lambdaR << ','
             << bgc::models::tsompsiv::toString(constants.vBoundary.type) << ','
             << constants.vBoundary.sc << ',' << fStar << ','
             << r.nx << ',' << r.h << ',' << r.dt << ',' << r.nTimes << '\n';
    }
}

} // namespace

int main(int argc, char** argv) {
    PetscCallAbort(PETSC_COMM_WORLD,
                   PetscInitialize(&argc,
                                   &argv,
                                   nullptr,
                                   "MMS3TSOMPsiV -- TSOMPsiV mass-conservation verification"));

    PetscInt exitCode = 0;
    try {
        const std::filesystem::path caseDir = caseDirectory();
        const std::filesystem::path dataDir = caseDir / "Dados";
        const InputData input = bgc::models::bgc::readMMSInputData(dataDir, defaultInput());
        bgc::models::tsompsiv::Constants constants = readTSOMPsiVConstants(dataDir);
        constants.theta = 1.0;
        constants.vBoundary.type = bgc::models::tsompsiv::VBoundaryCondition::ConstantGradient;
        constants.vBoundary.sc = 0.0;
        const std::filesystem::path outputDir =
            bgc::models::bgc::resolveMMSOutputDirectory(caseDir, input.outputDir);
        bgc::ensureDirectory(outputDir);

        const auto d = bgc::models::tsompsiv::computeDiffusionCoefficients(constants);
        PetscCallAbort(PETSC_COMM_WORLD, bgc::models::bgc::printCpuUsageSummary());
        PetscPrintf(PETSC_COMM_WORLD,
                    "\nMMS3TSOMPsiV -- TSOMPsiV discrete mass-conservation check\n"
                    "  Psi(x,0) = sin^2(pi*x), no source, no-flux Psi boundary\n"
                    "  alpha=%.6e rho=%.6e theta=%.6e lambdaC=%.6e lambdaR=%.6e\n"
                    "  diffusion: d11=%.6e d12=%.6e d21=%.6e d22=%.6e\n"
                    "  output: %s\n",
                    static_cast<PetscReal>(constants.alpha),
                    static_cast<PetscReal>(constants.rho),
                    static_cast<PetscReal>(constants.theta),
                    static_cast<PetscReal>(constants.lambdaC),
                    static_cast<PetscReal>(constants.lambdaR),
                    static_cast<PetscReal>(d.d11),
                    static_cast<PetscReal>(d.d12),
                    static_cast<PetscReal>(d.d21),
                    static_cast<PetscReal>(d.d22),
                    outputDir.string().c_str());

        std::vector<MassRecord> records;
        for (const PetscInt nx : input.nxList) {
            const MeshTimeData mesh = bgc::models::bgc::makeMMSMeshTimeData(input, nx);
            std::vector<PetscReal> state;
            records.push_back(runMesh(mesh, input, constants, state));
            if (bgc::models::bgc::isWorldRankZero()) {
                writeFields(outputDir / ("mms3tsompsiv_fields_N" + std::to_string(nx) + ".dat"),
                            mesh,
                            state);
            }
        }

        if (bgc::models::bgc::isWorldRankZero()) {
            writeMassSummary(outputDir / "mms3tsompsiv_mass.csv", records);
            writeSetup(outputDir / "mms3tsompsiv_setup.csv", records, constants);
        }
    } catch (const std::exception& ex) {
        PetscPrintf(PETSC_COMM_WORLD, "\nERROR: %s\n", ex.what());
        exitCode = 1;
    }

    PetscCallAbort(PETSC_COMM_WORLD, PetscFinalize());
    return exitCode;
}
