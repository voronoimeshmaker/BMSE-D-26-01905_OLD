#include <petsc.h>

#include <bgclib/BGCLib.hpp>

// MMS2TSOM exercises the new bgclib TSOM model path with a manufactured
// two-field solution. The driver owns only the manufactured U/V fields, source
// terms, and output table layout. TSOM coefficients, V boundary-condition
// variants, PETSc operator assembly, truncation error, and finite-volume norms
// come from bgclib.
//
// Unknown ordering is flattened as:
//
//   [ U_0 ... U_{N-1}  V_0 ... V_{N-1} ]

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using InputData = bgc::models::bgc::MMSInputData;
using MeshTimeData = bgc::models::bgc::MMSMeshTimeData;

struct NormRecord {
    PetscInt nx {0};
    PetscReal h {0.0};
    PetscReal dt {0.0};
    PetscInt nTimes {0};
    PetscReal l1Psi {0.0};
    PetscReal l2Psi {0.0};
    PetscReal linfPsi {0.0};
    PetscReal l1U {0.0};
    PetscReal l2U {0.0};
    PetscReal linfU {0.0};
    PetscReal l1V {0.0};
    PetscReal l2V {0.0};
    PetscReal linfV {0.0};
    PetscReal lteL2Psi {0.0};
    PetscReal lteLinfPsi {0.0};
    PetscReal lteL2U {0.0};
    PetscReal lteLinfU {0.0};
    PetscReal lteL2V {0.0};
    PetscReal lteLinfV {0.0};
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
        .debug = true,
    };
}

bgc::models::tsom::Constants defaultConstants() {
    bgc::models::tsom::Constants constants;
    constants.alpha = 0.75;
    constants.rho = 0.25;
    constants.theta = 1.0;
    constants.lambdaC = 1.0;
    constants.lambdaR = 1.0 / 3.0;
    constants.vBoundary = {
        .type = bgc::models::tsom::VBoundaryCondition::ZeroValue,
        .sc = 0.0,
    };
    return constants;
}

bgc::models::tsom::Constants readTSOMConstants(const std::filesystem::path& dataDir) {
    bgc::models::tsom::Constants constants = defaultConstants();
    const std::unordered_map<std::string, std::string> data =
        ::bgc::readKeyValueFile(dataDir / "simulation.dat");

    if (data.contains("alpha")) {
        constants.alpha = ::bgc::parseReal(data.at("alpha"));
    }
    if (data.contains("alpha_tsom")) {
        constants.alpha = ::bgc::parseReal(data.at("alpha_tsom"));
    }
    if (data.contains("rho")) {
        constants.rho = ::bgc::parseReal(data.at("rho"));
    }
    if (data.contains("rho_tsom")) {
        constants.rho = ::bgc::parseReal(data.at("rho_tsom"));
    }
    if (data.contains("theta")) {
        constants.theta = ::bgc::parseReal(data.at("theta"));
    }
    if (data.contains("theta_tsom")) {
        constants.theta = ::bgc::parseReal(data.at("theta_tsom"));
    }
    if (data.contains("lambda_c")) {
        constants.lambdaC = ::bgc::parseReal(data.at("lambda_c"));
    }
    if (data.contains("lambdac")) {
        constants.lambdaC = ::bgc::parseReal(data.at("lambdac"));
    }
    if (data.contains("lambda_r")) {
        constants.lambdaR = ::bgc::parseReal(data.at("lambda_r"));
    }
    if (data.contains("lambdar")) {
        constants.lambdaR = ::bgc::parseReal(data.at("lambdar"));
    }
    if (data.contains("v_boundary")) {
        constants.vBoundary.type = bgc::models::tsom::parseVBoundaryCondition(
            ::bgc::trim(data.at("v_boundary")));
    }
    if (data.contains("v_boundary_condition")) {
        constants.vBoundary.type = bgc::models::tsom::parseVBoundaryCondition(
            ::bgc::trim(data.at("v_boundary_condition")));
    }
    if (data.contains("v_boundary_type")) {
        constants.vBoundary.type = bgc::models::tsom::parseVBoundaryCondition(
            ::bgc::trim(data.at("v_boundary_type")));
    }
    if (data.contains("sc")) {
        constants.vBoundary.sc = ::bgc::parseReal(data.at("sc"));
    }
    if (data.contains("v_boundary_sc")) {
        constants.vBoundary.sc =
            ::bgc::parseReal(data.at("v_boundary_sc"));
    }

    if (!bgc::ModelTraits<bgc::models::tsom::Tag>::constantsAreValid(constants)) {
        throw std::runtime_error("Invalid TSOM constants in simulation.dat.");
    }
    return constants;
}

bgc::BoundarySet makeMMS2Boundaries() {
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

PetscReal gMMS2(const PetscReal x) {
    const PetscReal y = 1.0 - x;
    return x * x * y * y;
}

PetscReal gppMMS2(const PetscReal x) {
    return 2.0 * (1.0 - 6.0 * x + 6.0 * x * x);
}

PetscReal uExact(const PetscReal x, const PetscReal t) {
    return PetscExpReal(-t) * gMMS2(x);
}

PetscReal vExact(const PetscReal x, const PetscReal t) {
    return PetscExpReal(-t) * gMMS2(x);
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

std::vector<PetscReal> buildExactState(const MeshTimeData& mesh,
                                       const PetscReal time) {
    std::vector<PetscReal> state(static_cast<std::size_t>(2 * mesh.nx), 0.0);
    for (PetscInt i = 0; i < mesh.nx; ++i) {
        const PetscReal x = (static_cast<PetscReal>(i) + 0.5) * mesh.h;
        state[static_cast<std::size_t>(i)] = uExact(x, time);
        state[static_cast<std::size_t>(mesh.nx + i)] = vExact(x, time);
    }
    return state;
}

struct MMS2RHSCache {
    std::vector<PetscReal> uProfile;
    std::vector<PetscReal> vProfile;
    std::vector<bgc::RHSEntry> boundaryEntries;
};

MMS2RHSCache buildMMS2RHSCache(const MeshTimeData& mesh,
                               const bgc::models::tsom::Constants& constants) {
    const bgc::Grid1D grid = makeGrid(mesh);
    const auto boundaryRHS = bgc::models::tsom::buildBoundaryRHS(
        grid,
        bgc::models::tsom::computeBoundaryRHS(grid, constants, makeMMS2Boundaries(), 0.0));
    const auto d = bgc::models::tsom::computeDiffusionCoefficients(constants);

    MMS2RHSCache cache {
        .uProfile = std::vector<PetscReal>(static_cast<std::size_t>(mesh.nx), 0.0),
        .vProfile = std::vector<PetscReal>(static_cast<std::size_t>(mesh.nx), 0.0),
        .boundaryEntries = boundaryRHS.entries,
    };

    const PetscReal uReaction = -1.0 + constants.lambdaC - constants.lambdaR;
    const PetscReal vReaction = -1.0 - constants.lambdaC + constants.lambdaR;
    const PetscReal uDiffusion = d.d11 + d.d12;
    const PetscReal vDiffusion = d.d21 + d.d22;

    for (PetscInt i = 0; i < mesh.nx; ++i) {
        const auto index = static_cast<std::size_t>(i);
        const PetscReal x = (static_cast<PetscReal>(i) + 0.5) * mesh.h;
        const PetscReal g = gMMS2(x);
        const PetscReal gpp = gppMMS2(x);
        cache.uProfile[index] = uReaction * g - uDiffusion * gpp;
        cache.vProfile[index] = vReaction * g - vDiffusion * gpp;
    }

    return cache;
}

std::vector<PetscReal> buildRHS(const MeshTimeData& mesh,
                                const MMS2RHSCache& cache,
                                const PetscReal time,
                                const std::vector<PetscReal>& previousState) {
    const PetscReal hdt = mesh.h / mesh.dt;
    const PetscReal timeFactor = PetscExpReal(-time);
    std::vector<PetscReal> rhs(static_cast<std::size_t>(2 * mesh.nx), 0.0);

    for (PetscInt i = 0; i < mesh.nx; ++i) {
        const auto index = static_cast<std::size_t>(i);
        const auto vIndex = static_cast<std::size_t>(mesh.nx + i);
        rhs[index] = hdt * previousState[index] +
                     mesh.h * timeFactor * cache.uProfile[index];
        rhs[vIndex] = hdt * previousState[vIndex] +
                      mesh.h * timeFactor * cache.vProfile[index];
    }

    for (const bgc::RHSEntry& entry : cache.boundaryEntries) {
        rhs[static_cast<std::size_t>(entry.row)] += entry.value;
    }
    return rhs;
}

bgc::DiscreteOperator buildTSOMOperator(const MeshTimeData& mesh,
                                        const InputData& input,
                                        const bgc::models::tsom::Constants& constants) {
    const bgc::Grid1D grid = makeGrid(mesh);
    const auto coefficients = bgc::models::tsom::computeCoefficients(
        grid,
        makeTime(mesh, input),
        constants,
        makeMMS2Boundaries());
    return bgc::models::tsom::buildOperator(grid, coefficients);
}

PetscErrorCode solveTransient(const MeshTimeData& mesh,
                              const InputData& input,
                              const bgc::models::tsom::Constants& constants,
                              std::vector<PetscReal>& state,
                              KSPConvergedReason& reason,
                              PetscInt& iterations) {
    PetscFunctionBeginUser;

    const auto op = buildTSOMOperator(mesh, input, constants);
    const MMS2RHSCache rhsCache = buildMMS2RHSCache(mesh, constants);
    state = buildExactState(mesh, 0.0);
    reason = KSP_CONVERGED_ITERATING;
    iterations = 0;

    bgc::models::bgc::MMSLinearSolver solver(op, 0.0);

    for (PetscInt step = 1; step <= mesh.nTimes; ++step) {
        const PetscReal time = static_cast<PetscReal>(step) * mesh.dt;
        const std::vector<PetscReal> rhsValues = buildRHS(mesh, rhsCache, time, state);
        PetscCall(solver.solve(rhsValues, state, reason, iterations));
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

std::vector<PetscReal> computeTruncationError(const MeshTimeData& mesh,
                                              const InputData& input,
                                              const bgc::models::tsom::Constants& constants) {
    const auto op = buildTSOMOperator(mesh, input, constants);
    const MMS2RHSCache rhsCache = buildMMS2RHSCache(mesh, constants);
    const std::vector<PetscReal> exactNow = buildExactState(mesh, input.tf);
    const std::vector<PetscReal> exactPrevious = buildExactState(mesh, input.tf - mesh.dt);
    const std::vector<PetscReal> rhs = buildRHS(mesh, rhsCache, input.tf, exactPrevious);
    std::vector<PetscReal> tau;

    PetscCallAbort(PETSC_COMM_SELF,
                   bgc::computeLocalTruncationError(op, exactNow, rhs, tau));
    return tau;
}

NormRecord computeNorms(const MeshTimeData& mesh,
                        const std::vector<PetscReal>& numerical,
                        const std::vector<PetscReal>& exact,
                        const std::vector<PetscReal>& tau,
                        const KSPConvergedReason reason,
                        const PetscInt iterations) {
    NormRecord record {
        .nx = mesh.nx,
        .h = mesh.h,
        .dt = mesh.dt,
        .nTimes = mesh.nTimes,
        .reason = reason,
        .iterations = iterations,
    };

    const auto uNumerical = std::span<const PetscReal>(numerical).first(
        static_cast<std::size_t>(mesh.nx));
    const auto vNumerical = std::span<const PetscReal>(numerical).last(
        static_cast<std::size_t>(mesh.nx));
    const auto uExactValues = std::span<const PetscReal>(exact).first(
        static_cast<std::size_t>(mesh.nx));
    const auto vExactValues = std::span<const PetscReal>(exact).last(
        static_cast<std::size_t>(mesh.nx));
    const auto tauU = std::span<const PetscReal>(tau).first(
        static_cast<std::size_t>(mesh.nx));
    const auto tauV = std::span<const PetscReal>(tau).last(
        static_cast<std::size_t>(mesh.nx));

    std::vector<PetscReal> psiNumerical(static_cast<std::size_t>(mesh.nx), 0.0);
    std::vector<PetscReal> psiExact(static_cast<std::size_t>(mesh.nx), 0.0);
    std::vector<PetscReal> tauPsi(static_cast<std::size_t>(mesh.nx), 0.0);
    for (PetscInt i = 0; i < mesh.nx; ++i) {
        const auto index = static_cast<std::size_t>(i);
        psiNumerical[index] = uNumerical[index] + vNumerical[index];
        psiExact[index] = uExactValues[index] + vExactValues[index];
        tauPsi[index] = tauU[index] + tauV[index];
    }

    const bgc::ErrorNorms psiNorms =
        bgc::computeErrorNorms(psiNumerical, psiExact, mesh.h);
    const bgc::ErrorNorms uNorms =
        bgc::computeErrorNorms(uNumerical, uExactValues, mesh.h);
    const bgc::ErrorNorms vNorms =
        bgc::computeErrorNorms(vNumerical, vExactValues, mesh.h);
    const bgc::ErrorNorms tauPsiNorms = bgc::computeVectorNorms(tauPsi, mesh.h);
    const bgc::ErrorNorms tauUNorms = bgc::computeVectorNorms(tauU, mesh.h);
    const bgc::ErrorNorms tauVNorms = bgc::computeVectorNorms(tauV, mesh.h);

    record.l1Psi = psiNorms.l1;
    record.l2Psi = psiNorms.l2;
    record.linfPsi = psiNorms.linf;
    record.l1U = uNorms.l1;
    record.l2U = uNorms.l2;
    record.linfU = uNorms.linf;
    record.l1V = vNorms.l1;
    record.l2V = vNorms.l2;
    record.linfV = vNorms.linf;
    record.lteL2Psi = tauPsiNorms.l2;
    record.lteLinfPsi = tauPsiNorms.linf;
    record.lteL2U = tauUNorms.l2;
    record.lteLinfU = tauUNorms.linf;
    record.lteL2V = tauVNorms.l2;
    record.lteLinfV = tauVNorms.linf;
    return record;
}

void writeFields(const std::filesystem::path& path,
                 const MeshTimeData& mesh,
                 const std::vector<PetscReal>& numerical,
                 const std::vector<PetscReal>& exact) {
    std::ofstream file = bgc::openOutputFile(path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open output file: " + path.string());
    }

    file << std::scientific << std::setprecision(16);
    file << "# P x u_num u_exact u_err v_num v_exact v_err psi_num psi_exact psi_err\n";
    for (PetscInt i = 0; i < mesh.nx; ++i) {
        const auto uIndex = static_cast<std::size_t>(i);
        const auto vIndex = static_cast<std::size_t>(mesh.nx + i);
        const PetscReal x = (static_cast<PetscReal>(i) + 0.5) * mesh.h;
        const PetscReal psiNum = numerical[uIndex] + numerical[vIndex];
        const PetscReal psiEx = exact[uIndex] + exact[vIndex];
        file << i + 1 << ' ' << x << ' '
             << numerical[uIndex] << ' ' << exact[uIndex] << ' '
             << numerical[uIndex] - exact[uIndex] << ' '
             << numerical[vIndex] << ' ' << exact[vIndex] << ' '
             << numerical[vIndex] - exact[vIndex] << ' '
             << psiNum << ' ' << psiEx << ' ' << psiNum - psiEx << '\n';
    }
}

void writeTruncation(const std::filesystem::path& path,
                     const MeshTimeData& mesh,
                     const std::vector<PetscReal>& tau) {
    std::ofstream file = bgc::openOutputFile(path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open output file: " + path.string());
    }

    file << std::scientific << std::setprecision(16);
    file << "# P x_center tau_U tau_V tau_Psi\n";
    for (PetscInt i = 0; i < mesh.nx; ++i) {
        const auto uIndex = static_cast<std::size_t>(i);
        const auto vIndex = static_cast<std::size_t>(mesh.nx + i);
        const PetscReal x = (static_cast<PetscReal>(i) + 0.5) * mesh.h;
        file << i + 1 << ' ' << x << ' ' << tau[uIndex] << ' ' << tau[vIndex]
             << ' ' << tau[uIndex] + tau[vIndex] << '\n';
    }
}

void writeConvergence(const std::filesystem::path& path,
                      const std::vector<NormRecord>& records) {
    std::ofstream file = bgc::openOutputFile(path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open output file: " + path.string());
    }

    file << std::scientific << std::setprecision(16);
    file << "nx,h,dt,nTimes,L1_Psi,L2_Psi,Linf_Psi,L1_U,L2_U,Linf_U,L1_V,L2_V,Linf_V,"
            "LTE_L2_Psi,LTE_Linf_Psi,LTE_L2_U,LTE_Linf_U,LTE_L2_V,LTE_Linf_V,"
            "ksp_reason,ksp_iterations,status\n";
    for (const NormRecord& r : records) {
        file << r.nx << ',' << r.h << ',' << r.dt << ',' << r.nTimes
             << ',' << r.l1Psi << ',' << r.l2Psi << ',' << r.linfPsi
             << ',' << r.l1U << ',' << r.l2U << ',' << r.linfU
             << ',' << r.l1V << ',' << r.l2V << ',' << r.linfV
             << ',' << r.lteL2Psi << ',' << r.lteLinfPsi
             << ',' << r.lteL2U << ',' << r.lteLinfU
             << ',' << r.lteL2V << ',' << r.lteLinfV
             << ',' << static_cast<int>(r.reason) << ',' << r.iterations
             << ',' << bgc::convergenceStatus(r.reason) << '\n';
    }
}

void writeSetup(const std::filesystem::path& path,
                const std::vector<NormRecord>& records,
                const bgc::models::tsom::Constants& constants) {
    std::ofstream file = bgc::openOutputFile(path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open output file: " + path.string());
    }

    file << std::scientific << std::setprecision(16);
    file << "alpha,rho,theta,lambdaC,lambdaR,vBoundary,sc,nx,h,dt,nTimes\n";
    const bool usesSc =
        constants.vBoundary.type == bgc::models::tsom::VBoundaryCondition::ScaledPsi;
    for (const NormRecord& r : records) {
        file << constants.alpha << ',' << constants.rho << ',' << constants.theta << ','
             << constants.lambdaC << ',' << constants.lambdaR << ','
             << bgc::models::tsom::toString(constants.vBoundary.type) << ',';
        if (usesSc) {
            file << constants.vBoundary.sc;
        }
        file << ','
             << r.nx << ',' << r.h << ',' << r.dt << ',' << r.nTimes << '\n';
    }
}

void runCases(const InputData& input,
              const bgc::models::tsom::Constants& constants,
              const std::filesystem::path& outputDir) {
    std::vector<NormRecord> records;
    bgc::ensureDirectory(outputDir);

    for (const PetscInt nx : input.nxList) {
        const MeshTimeData mesh = bgc::models::bgc::makeMMSMeshTimeData(input, nx);
        const std::string nxText = std::to_string(static_cast<int>(nx));
        std::vector<PetscReal> numerical;
        KSPConvergedReason reason = KSP_CONVERGED_ITERATING;
        PetscInt iterations = 0;

        PetscCallAbort(PETSC_COMM_SELF,
                       solveTransient(mesh, input, constants, numerical, reason, iterations));

        const std::vector<PetscReal> exact = buildExactState(mesh, input.tf);
        const std::vector<PetscReal> tau = computeTruncationError(mesh, input, constants);
        const NormRecord record = computeNorms(mesh, numerical, exact, tau, reason, iterations);

        writeFields(outputDir / ("mms2tsom_fields_N" + nxText + ".dat"),
                    mesh,
                    numerical,
                    exact);
        writeTruncation(outputDir / ("mms2tsom_lte_N" + nxText + ".dat"), mesh, tau);
        records.push_back(record);
    }

    writeConvergence(outputDir / "mms2tsom_convergence.csv", records);
    writeSetup(outputDir / "mms2tsom_setup.csv", records, constants);
}

} // namespace

int main(int argc, char** argv) {
    PetscCallAbort(PETSC_COMM_WORLD,
                   PetscInitialize(&argc,
                                   &argv,
                                   nullptr,
                                   "MMS2TSOM -- no-flux manufactured solution for the TSOM model"));

    PetscInt exitCode = 0;
    try {
        const std::filesystem::path caseDir = caseDirectory();
        const std::filesystem::path dataDir = caseDir / "Dados";
        const InputData input = bgc::models::bgc::readMMSInputData(dataDir, defaultInput());
        const bgc::models::tsom::Constants constants = readTSOMConstants(dataDir);
        const std::filesystem::path outputDir =
            bgc::models::bgc::resolveMMSOutputDirectory(caseDir, input.outputDir);

        const bgc::models::tsom::DiffusionCoefficients d =
            bgc::models::tsom::computeDiffusionCoefficients(constants);
        const std::string vBoundaryText {
            bgc::models::tsom::toString(constants.vBoundary.type)
        };
        const bool usesSc =
            constants.vBoundary.type == bgc::models::tsom::VBoundaryCondition::ScaledPsi;
        const std::string scText =
            usesSc ? " sc=" + std::to_string(static_cast<PetscReal>(constants.vBoundary.sc))
                   : " (sc unused)";

        PetscCallAbort(PETSC_COMM_WORLD, bgc::models::bgc::printCpuUsageSummary());
        PetscPrintf(PETSC_COMM_WORLD,
                    "\nMMS2TSOM -- TSOM MMS with homogeneous no-flux Psi boundaries\n"
                    "  U(x,t)   = exp(-t) * x^2*(1-x)^2\n"
                    "  V(x,t)   = exp(-t) * x^2*(1-x)^2\n"
                    "  Psi(x,t) = U + V\n"
                    "  alpha=%.6e rho=%.6e theta=%.6e lambdaC=%.6e lambdaR=%.6e\n"
                    "  diffusion: d11=%.6e d12=%.6e d21=%.6e d22=%.6e\n"
                    "  V boundary: %s%s\n"
                    "  input : %s\n"
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
                    vBoundaryText.c_str(),
                    scText.c_str(),
                    dataDir.string().c_str(),
                    outputDir.string().c_str());

        runCases(input, constants, outputDir);
    } catch (const std::exception& ex) {
        PetscPrintf(PETSC_COMM_WORLD, "\nERROR: %s\n", ex.what());
        exitCode = 1;
    }

    PetscCallAbort(PETSC_COMM_WORLD, PetscFinalize());
    return exitCode;
}
