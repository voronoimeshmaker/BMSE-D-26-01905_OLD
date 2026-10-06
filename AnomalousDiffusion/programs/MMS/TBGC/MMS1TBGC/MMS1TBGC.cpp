#include <petsc.h>
#include <bgclib/BGCLib.hpp>

#include <span>

namespace {

// See README.md in this directory for the mathematical statement, output
// contract, and the mapping between this driver and the new bgclib TBGC API.
//
// MMS1TBGC uses the new bgclib TBGC model only. The executable and output
// names follow the requested TBGC spelling, while the model namespace remains
// bgc::models::tbgc because that is the canonical library API.

using InputData = bgc::models::bgc::MMSInputData;
using MeshTimeData = bgc::models::bgc::MMSMeshTimeData;

struct NormRecord {
    PetscReal bv {0.0};
    PetscInt nx {0};
    PetscReal h {0.0};
    PetscReal dt {0.0};
    PetscInt nTimes {0};
    PetscReal l1Phi {0.0};
    PetscReal l2Phi {0.0};
    PetscReal linfPhi {0.0};
    PetscReal l1Mu {0.0};
    PetscReal l2Mu {0.0};
    PetscReal linfMu {0.0};
    PetscReal l1Tau {0.0};
    PetscReal l2Tau {0.0};
    PetscReal linfTau {0.0};
    KSPConvergedReason reason {KSP_CONVERGED_ITERATING};
    PetscInt iterations {0};
};

std::filesystem::path caseDirectory() {
    return bgc::caseDirectoryFromSource(__FILE__);
}

InputData defaultInput() {
    return {
        .tf = 1.0e-3,
        .cdt = 1.024e-1,
        .nxList = {8, 16, 32, 64, 128, 256, 512},
        .bvList = {1.0e-2, 5.0e-1, 1.0},
        .outputDir = "Saida",
        .verbose = true,
        .debug = false,
    };
}

bgc::BoundarySet makeMMS1Boundaries() {
    return {
        .west = {
            .conditions = {
                bgc::BoundaryCondition::neumann(0.0),
                bgc::BoundaryCondition::dirichlet(0.0),
            },
        },
        .east = {
            .conditions = {
                bgc::BoundaryCondition::neumann(0.0),
                bgc::BoundaryCondition::dirichlet(0.0),
            },
        },
    };
}

PetscReal phiExact(const PetscReal x, const PetscReal t) {
    const PetscReal oneMinusX = 1.0 - x;
    return PetscExpReal(-t) * x * x * oneMinusX * oneMinusX;
}

PetscReal phiXX(const PetscReal x, const PetscReal t) {
    return PetscExpReal(-t) * (12.0 * x * x - 12.0 * x + 2.0);
}

PetscReal muExact(const bgc::models::tbgc::Constants& constants,
                  const PetscReal x,
                  const PetscReal t) {
    return phiExact(x, t) - constants.bv * phiXX(x, t);
}

PetscReal sourceExact(const bgc::models::tbgc::Constants& constants,
                      const PetscReal x,
                      const PetscReal t) {
    const PetscReal x2 = x * x;
    const PetscReal x3 = x2 * x;
    const PetscReal x4 = x2 * x2;
    return PetscExpReal(-t) *
           (24.0 * constants.bv - x4 + 2.0 * x3 - 13.0 * x2 + 12.0 * x - 2.0);
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
                                       const bgc::models::tbgc::Constants& constants,
                                       const PetscReal time) {
    std::vector<PetscReal> values(static_cast<std::size_t>(2 * mesh.nx), 0.0);
    for (PetscInt i = 0; i < mesh.nx; ++i) {
        const PetscReal x = (static_cast<PetscReal>(i) + 0.5) * mesh.h;
        values[static_cast<std::size_t>(i)] = phiExact(x, time);
        values[static_cast<std::size_t>(mesh.nx + i)] = muExact(constants, x, time);
    }
    return values;
}

std::vector<PetscReal> buildRHS(const MeshTimeData& mesh,
                                const bgc::models::tbgc::Constants& constants,
                                const PetscReal time,
                                const std::vector<PetscReal>& previousPhi) {
    const bgc::Grid1D grid = makeGrid(mesh);
    const auto boundaryRHS = bgc::models::tbgc::buildBoundaryRHS(
        grid,
        bgc::models::tbgc::computeBoundaryRHS(grid, constants, makeMMS1Boundaries(), time));
    const PetscReal hdt = mesh.h / mesh.dt;
    std::vector<PetscReal> rhs(static_cast<std::size_t>(2 * mesh.nx), 0.0);

    for (PetscInt i = 0; i < mesh.nx; ++i) {
        const PetscReal x = (static_cast<PetscReal>(i) + 0.5) * mesh.h;
        rhs[static_cast<std::size_t>(i)] =
            mesh.h * sourceExact(constants, x, time) +
            hdt * previousPhi[static_cast<std::size_t>(i)];
    }

    for (const bgc::RHSEntry& entry : boundaryRHS.entries) {
        rhs[static_cast<std::size_t>(entry.row)] += entry.value;
    }
    return rhs;
}

PetscErrorCode solveTransient(const MeshTimeData& mesh,
                              const InputData& input,
                              const bgc::models::tbgc::Constants& constants,
                              std::vector<PetscReal>& state,
                              KSPConvergedReason& reason,
                              PetscInt& iterations) {
    PetscFunctionBeginUser;

    const bgc::Grid1D grid = makeGrid(mesh);
    const auto coefficients = bgc::models::tbgc::computeCoefficients(
        grid,
        makeTime(mesh, input),
        constants);
    const auto op = bgc::models::tbgc::flattenBlockOperator(
        bgc::models::tbgc::buildBlockOperator(grid, coefficients));

    state = buildExactState(mesh, constants, 0.0);
    reason = KSP_CONVERGED_ITERATING;
    iterations = 0;

    bgc::models::bgc::MMSLinearSolver solver(op, 0.0);
    for (PetscInt step = 1; step <= mesh.nTimes; ++step) {
        const PetscReal time = static_cast<PetscReal>(step) * mesh.dt;
        const std::vector<PetscReal> rhs = buildRHS(mesh, constants, time, state);

        PetscCall(solver.solve(rhs, state, reason, iterations));
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

std::vector<PetscReal> computeTruncationError(const MeshTimeData& mesh,
                                              const InputData& input,
                                              const bgc::models::tbgc::Constants& constants) {
    const bgc::Grid1D grid = makeGrid(mesh);
    const auto coefficients = bgc::models::tbgc::computeCoefficients(
        grid,
        makeTime(mesh, input),
        constants);
    const auto op = bgc::models::tbgc::flattenBlockOperator(
        bgc::models::tbgc::buildBlockOperator(grid, coefficients));
    const std::vector<PetscReal> exactNow = buildExactState(mesh, constants, input.tf);
    const PetscReal previousTime = input.tf - mesh.dt;
    const std::vector<PetscReal> exactPrevious =
        buildExactState(mesh, constants, previousTime);
    const std::vector<PetscReal> rhs = buildRHS(mesh, constants, input.tf, exactPrevious);
    std::vector<PetscReal> tau;

    PetscCallAbort(PETSC_COMM_SELF,
                   bgc::computeLocalTruncationError(op, exactNow, rhs, tau));
    return tau;
}

NormRecord computeNorms(const MeshTimeData& mesh,
                        const PetscReal bv,
                        const std::vector<PetscReal>& numerical,
                        const std::vector<PetscReal>& exact,
                        const std::vector<PetscReal>& tau,
                        const KSPConvergedReason reason,
                        const PetscInt iterations) {
    NormRecord record {
        .bv = bv,
        .nx = mesh.nx,
        .h = mesh.h,
        .dt = mesh.dt,
        .nTimes = mesh.nTimes,
        .reason = reason,
        .iterations = iterations,
    };

    const auto phiNumerical = std::span<const PetscReal>(numerical).first(
        static_cast<std::size_t>(mesh.nx));
    const auto muNumerical = std::span<const PetscReal>(numerical).last(
        static_cast<std::size_t>(mesh.nx));
    const auto phiExactValues = std::span<const PetscReal>(exact).first(
        static_cast<std::size_t>(mesh.nx));
    const auto muExactValues = std::span<const PetscReal>(exact).last(
        static_cast<std::size_t>(mesh.nx));

    const bgc::ErrorNorms phiNorms =
        bgc::computeErrorNorms(phiNumerical, phiExactValues, mesh.h);
    const bgc::ErrorNorms muNorms =
        bgc::computeErrorNorms(muNumerical, muExactValues, mesh.h);
    const bgc::ErrorNorms tauNorms = bgc::computeVectorNorms(tau, mesh.h);

    record.l1Phi = phiNorms.l1;
    record.l2Phi = phiNorms.l2;
    record.linfPhi = phiNorms.linf;
    record.l1Mu = muNorms.l1;
    record.l2Mu = muNorms.l2;
    record.linfMu = muNorms.linf;
    record.l1Tau = tauNorms.l1;
    record.l2Tau = tauNorms.l2;
    record.linfTau = tauNorms.linf;
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
    file << "# P x phi_num phi_exact phi_err mu_num mu_exact mu_err\n";
    for (PetscInt i = 0; i < mesh.nx; ++i) {
        const auto phiIndex = static_cast<std::size_t>(i);
        const auto muIndex = static_cast<std::size_t>(mesh.nx + i);
        const PetscReal x = (static_cast<PetscReal>(i) + 0.5) * mesh.h;
        file << i + 1 << ' ' << x << ' '
             << numerical[phiIndex] << ' ' << exact[phiIndex] << ' '
             << numerical[phiIndex] - exact[phiIndex] << ' '
             << numerical[muIndex] << ' ' << exact[muIndex] << ' '
             << numerical[muIndex] - exact[muIndex] << '\n';
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
    file << "# P x_center tau_phi tau_mu\n";
    for (PetscInt i = 0; i < mesh.nx; ++i) {
        const PetscReal x = (static_cast<PetscReal>(i) + 0.5) * mesh.h;
        file << i + 1 << ' ' << x << ' '
             << tau[static_cast<std::size_t>(i)] << ' '
             << tau[static_cast<std::size_t>(mesh.nx + i)] << '\n';
    }
}

void writeConvergence(const std::filesystem::path& path,
                      const std::vector<NormRecord>& records) {
    std::ofstream file = bgc::openOutputFile(path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open output file: " + path.string());
    }

    file << std::scientific << std::setprecision(16);
    file << "bv,nx,h,dt,nTimes,L1_phi,L2_phi,Linf_phi,L1_mu,L2_mu,Linf_mu,"
            "LTE_L1,LTE_L2,LTE_Linf,ksp_reason,ksp_iterations,status\n";
    for (const NormRecord& r : records) {
        file << r.bv << ',' << r.nx << ',' << r.h << ',' << r.dt << ',' << r.nTimes
             << ',' << r.l1Phi << ',' << r.l2Phi << ',' << r.linfPhi
             << ',' << r.l1Mu << ',' << r.l2Mu << ',' << r.linfMu
             << ',' << r.l1Tau << ',' << r.l2Tau << ',' << r.linfTau
             << ',' << static_cast<int>(r.reason) << ',' << r.iterations
             << ',' << bgc::convergenceStatus(r.reason) << '\n';
    }
}

void writeSetup(const std::filesystem::path& path,
                const std::vector<NormRecord>& records) {
    std::ofstream file = bgc::openOutputFile(path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open output file: " + path.string());
    }

    file << std::scientific << std::setprecision(16);
    file << "bv,nx,h,dt,nTimes\n";
    for (const NormRecord& r : records) {
        file << r.bv << ',' << r.nx << ',' << r.h << ',' << r.dt << ',' << r.nTimes << '\n';
    }
}

void runCases(const InputData& input, const std::filesystem::path& outputDir) {
    std::vector<NormRecord> setupRecords;

    for (const PetscReal bv : input.bvList) {
        const bgc::models::tbgc::Constants constants {.bv = bv};
        const std::filesystem::path outDir =
            bgc::models::bgc::mmsBvOutputDirectory(outputDir, bv);
        std::vector<NormRecord> norms;
        bgc::ensureDirectory(outDir);

        for (const PetscInt nx : input.nxList) {
            const MeshTimeData mesh = bgc::models::bgc::makeMMSMeshTimeData(input, nx);
            const std::string nxText = std::to_string(static_cast<int>(nx));
            std::vector<PetscReal> numerical;
            KSPConvergedReason reason = KSP_CONVERGED_ITERATING;
            PetscInt iterations = 0;

            PetscCallAbort(PETSC_COMM_SELF,
                           solveTransient(mesh, input, constants, numerical, reason, iterations));

            const std::vector<PetscReal> exact = buildExactState(mesh, constants, input.tf);
            const std::vector<PetscReal> tau = computeTruncationError(mesh, input, constants);
            const NormRecord record =
                computeNorms(mesh, bv, numerical, exact, tau, reason, iterations);

            writeFields(outDir / ("mms1tbgc_fields_N" + nxText + ".dat"),
                        mesh,
                        numerical,
                        exact);
            writeTruncation(outDir / ("mms1tbgc_lte_N" + nxText + ".dat"), mesh, tau);
            norms.push_back(record);
            setupRecords.push_back(record);
        }

        writeConvergence(outDir / "mms1tbgc_convergence.csv", norms);
    }

    writeSetup(outputDir / "mms1tbgc_setup.csv", setupRecords);
}

} // namespace

int main(int argc, char** argv) {
    PetscCallAbort(PETSC_COMM_WORLD,
                   PetscInitialize(&argc,
                                   &argv,
                                   nullptr,
                                   "MMS1TBGC -- homogeneous manufactured solution for the TBGC model"));

    PetscInt exitCode = 0;
    try {
        const std::filesystem::path caseDir = caseDirectory();
        const std::filesystem::path dataDir = caseDir / "Dados";
        const InputData input = bgc::models::bgc::readMMSInputData(dataDir, defaultInput());
        const std::filesystem::path outputDir =
            bgc::models::bgc::resolveMMSOutputDirectory(caseDir, input.outputDir);
        bgc::ensureDirectory(outputDir);
        runCases(input, outputDir);

        PetscCallAbort(PETSC_COMM_WORLD, bgc::models::bgc::printCpuUsageSummary());
        PetscPrintf(PETSC_COMM_WORLD,
                    "\nMMS1TBGC -- TBGC MMS with homogeneous boundary conditions\n"
                    "  phi(x,t) = exp(-t) * x^2 * (1 - x)^2\n"
                    "  mu(x,t)  = phi(x,t) - Bv * phi_xx(x,t)\n"
                    "  source   = exp(-t) * (24*Bv - x^4 + 2*x^3 - 13*x^2 + 12*x - 2)\n"
                    "  input : %s\n"
                    "  output: %s\n",
                    dataDir.string().c_str(),
                    outputDir.string().c_str());
    } catch (const std::exception& ex) {
        PetscPrintf(PETSC_COMM_WORLD, "\nERROR: %s\n", ex.what());
        exitCode = 1;
    }

    PetscCallAbort(PETSC_COMM_WORLD, PetscFinalize());
    return exitCode;
}
