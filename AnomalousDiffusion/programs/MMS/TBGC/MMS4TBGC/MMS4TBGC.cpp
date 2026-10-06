#include <petsc.h>

#include <bgclib/BGCLib.hpp>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// MMS4TBGC is a physical Gaussian-pulse campaign rather than an MMS
// convergence test. It uses only the new bgclib TBGC API and stores the
// unknowns in flattened order:
//
//   [ phi_0 ... phi_{N-1}  mu_0 ... mu_{N-1} ]

using InputData = bgc::models::bgc::MMSInputData;
using MeshTimeData = bgc::models::bgc::MMSMeshTimeData;

constexpr PetscReal kMuCenter = 0.5;
constexpr PetscReal kSigma0 = 0.04;
constexpr PetscInt kProfileSamples = 5;
constexpr PetscReal kBoundarySaturationRatio = 1.0e-4;

struct Diagnostics {
    PetscReal sigma2 {0.0};
    PetscReal sigma2Raw {0.0};
    PetscReal energy {0.0};
    PetscReal mass {0.0};
    PetscReal phiMin {0.0};
    PetscReal phiMax {0.0};
    PetscReal phiBoundaryWest {0.0};
    PetscReal phiBoundaryEast {0.0};
};

std::filesystem::path caseDirectory() {
    return bgc::caseDirectoryFromSource(__FILE__);
}

InputData defaultInput() {
    return {
        .tf = 1.0e-3,
        .cdt = 1.024e-1,
        .nxList = {256},
        .bvList = {1.0e-2, 5.0e-1},
        .outputDir = "Saida",
        .verbose = true,
        .debug = false,
    };
}

bgc::BoundarySet makeHomogeneousBoundaries() {
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

PetscReal gaussianInitialPhi(const PetscReal x) {
    const PetscReal pi = std::numbers::pi_v<PetscReal>;
    const PetscReal norm = 1.0 / (kSigma0 * PetscSqrtReal(2.0 * pi));
    const PetscReal arg = (x - kMuCenter) / kSigma0;
    return norm * PetscExpReal(-0.5 * arg * arg);
}

PetscReal gaussianInitialPhiXX(const PetscReal x, const PetscReal phi0) {
    const PetscReal dx = x - kMuCenter;
    const PetscReal sigma2 = kSigma0 * kSigma0;
    const PetscReal sigma4 = sigma2 * sigma2;
    return phi0 * (dx * dx / sigma4 - 1.0 / sigma2);
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

std::vector<PetscReal> buildInitialState(const MeshTimeData& mesh,
                                         const bgc::models::tbgc::Constants& constants) {
    std::vector<PetscReal> state(static_cast<std::size_t>(2 * mesh.nx), 0.0);
    for (PetscInt i = 0; i < mesh.nx; ++i) {
        const PetscReal x = (static_cast<PetscReal>(i) + 0.5) * mesh.h;
        const PetscReal phi = gaussianInitialPhi(x);
        const PetscReal mu = phi - constants.bv * gaussianInitialPhiXX(x, phi);
        state[static_cast<std::size_t>(i)] = phi;
        state[static_cast<std::size_t>(mesh.nx + i)] = mu;
    }
    return state;
}

std::vector<PetscReal> buildHomogeneousRHS(const MeshTimeData& mesh,
                                           const bgc::models::tbgc::Constants& constants,
                                           const PetscReal time,
                                           const std::vector<PetscReal>& previousState) {
    const bgc::Grid1D grid = makeGrid(mesh);
    const auto boundaryRHS = bgc::models::tbgc::buildBoundaryRHS(
        grid,
        bgc::models::tbgc::computeBoundaryRHS(
            grid,
            constants,
            makeHomogeneousBoundaries(),
            time));
    const PetscReal hdt = mesh.h / mesh.dt;
    std::vector<PetscReal> rhs(static_cast<std::size_t>(2 * mesh.nx), 0.0);

    for (PetscInt i = 0; i < mesh.nx; ++i) {
        rhs[static_cast<std::size_t>(i)] =
            hdt * previousState[static_cast<std::size_t>(i)];
    }
    for (const bgc::RHSEntry& entry : boundaryRHS.entries) {
        rhs[static_cast<std::size_t>(entry.row)] += entry.value;
    }
    return rhs;
}

Diagnostics computeDiagnostics(const MeshTimeData& mesh,
                               const bgc::models::tbgc::Constants& constants,
                               const std::vector<PetscReal>& state) {
    Diagnostics diag;
    if (state.empty()) {
        return diag;
    }

    const bgc::CenteredMomentDiagnostics moments =
        bgc::computeCenteredMomentDiagnostics(
            std::span<const PetscReal> {state.data(), static_cast<std::size_t>(mesh.nx)},
            mesh.h,
            0.0,
            kMuCenter);
    diag.sigma2Raw = moments.moment2Raw;
    diag.mass = moments.mass;
    diag.sigma2 = moments.moment2;
    diag.energy = bgc::computeQuadraticFreeEnergy(
        std::span<const PetscReal> {state.data(), static_cast<std::size_t>(mesh.nx)},
        mesh.h,
        constants.bv);
    diag.phiMin = moments.minimum;
    diag.phiMax = moments.maximum;
    diag.phiBoundaryWest = moments.boundaryWest;
    diag.phiBoundaryEast = moments.boundaryEast;
    return diag;
}

std::vector<PetscInt> makeProfileSteps(const PetscInt nTimes) {
    return bgc::makeUniformSampleSteps(nTimes, kProfileSamples);
}

std::filesystem::path bvNxOutputDirectory(const std::filesystem::path& outputDir,
                                          const PetscReal bv,
                                          const PetscInt nx) {
    return bgc::bvNxOutputDirectory(outputDir, bv, nx);
}

void writeProfile(const std::filesystem::path& path,
                  const MeshTimeData& mesh,
                  const std::vector<PetscReal>& state,
                  const PetscReal time,
                  const PetscReal bv,
                  const PetscInt sampleIndex) {
    std::ofstream file = bgc::openOutputFile(path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open profile file: " + path.string());
    }

    file << std::scientific << std::setprecision(16);
    file << "# MMS4TBGC Gaussian-pulse profile\n";
    file << "# sample = " << sampleIndex << "  tau = " << time << "  bv = " << bv
         << "  nx = " << mesh.nx << '\n';
    file << "# xi phi\n";
    file << 0.0 << ' ' << 0.0 << '\n';
    for (PetscInt i = 0; i < mesh.nx; ++i) {
        const PetscReal x = (static_cast<PetscReal>(i) + 0.5) * mesh.h;
        file << x << ' ' << state[static_cast<std::size_t>(i)] << '\n';
    }
    file << 1.0 << ' ' << 0.0 << '\n';
}

void appendDiagnostics(std::ofstream& file,
                       const PetscInt step,
                       const PetscReal time,
                       const Diagnostics& diag) {
    file << step << ' ' << time << ' ' << diag.sigma2 << ' ' << diag.sigma2Raw
         << ' ' << diag.energy << ' ' << diag.mass << ' ' << diag.phiMin
         << ' ' << diag.phiMax << ' ' << diag.phiBoundaryWest << ' '
         << diag.phiBoundaryEast << '\n';
}

PetscErrorCode runGaussianCampaign(const InputData& input,
                                   const std::filesystem::path& outputDir,
                                   const PetscReal bv,
                                   const PetscInt nx) {
    PetscFunctionBeginUser;

    const bgc::models::tbgc::Constants constants {.bv = bv};
    const MeshTimeData mesh = bgc::models::bgc::makeMMSMeshTimeData(input, nx);
    const bgc::Grid1D grid = makeGrid(mesh);
    const auto coefficients = bgc::models::tbgc::computeCoefficients(
        grid,
        makeTime(mesh, input),
        constants);
    const auto op = bgc::models::tbgc::flattenBlockOperator(
        bgc::models::tbgc::buildBlockOperator(grid, coefficients));
    const std::filesystem::path outDir = bvNxOutputDirectory(outputDir, bv, nx);
    std::vector<PetscReal> state = buildInitialState(mesh, constants);
    KSPConvergedReason reason = KSP_CONVERGED_ITERATING;
    PetscInt iterations = 0;

    bgc::ensureDirectory(outDir);

    std::ofstream series = bgc::openOutputFile(outDir / "mms4tbgc_timeseries.dat");
    if (!series.is_open()) {
        throw std::runtime_error("Could not open time-series file in " + outDir.string());
    }

    series << std::scientific << std::setprecision(16);
    series << "# MMS4TBGC time series -- TBGC Gaussian-pulse spreading\n";
    series << "# bv = " << bv << "  nx = " << mesh.nx << "  h = " << mesh.h
           << "  dt = " << mesh.dt << "  nTimes = " << mesh.nTimes << '\n';
    series << "# sigma2 is the mass-normalised centred second moment.\n";
    series << "# step tau sigma2 sigma2_raw F_h mass phi_min phi_max phi_bdyW phi_bdyE\n";

    const std::vector<PetscInt> profileSteps = makeProfileSteps(mesh.nTimes);
    Diagnostics initial = computeDiagnostics(mesh, constants, state);
    appendDiagnostics(series, 0, 0.0, initial);
    writeProfile(outDir / "mms4tbgc_profile_0.dat", mesh, state, 0.0, bv, 0);

    const PetscReal saturationLimit = kBoundarySaturationRatio * initial.phiMax;
    PetscBool saturationFlagged = PETSC_FALSE;
    PetscInt nextProfile = 1;

    bgc::models::bgc::MMSLinearSolver solver(op, 0.0);
    for (PetscInt step = 1; step <= mesh.nTimes; ++step) {
        const PetscReal time = static_cast<PetscReal>(step) * mesh.dt;
        const std::vector<PetscReal> rhs = buildHomogeneousRHS(mesh, constants, time, state);

        PetscCall(solver.solve(rhs, state, reason, iterations));

        const Diagnostics diag = computeDiagnostics(mesh, constants, state);
        appendDiagnostics(series, step, time, diag);

        if (!saturationFlagged &&
            (PetscAbsReal(diag.phiBoundaryWest) > saturationLimit ||
             PetscAbsReal(diag.phiBoundaryEast) > saturationLimit)) {
            saturationFlagged = PETSC_TRUE;
        }

        if (nextProfile < kProfileSamples &&
            step == profileSteps[static_cast<std::size_t>(nextProfile)]) {
            writeProfile(outDir / ("mms4tbgc_profile_" +
                                   std::to_string(static_cast<int>(nextProfile)) + ".dat"),
                         mesh,
                         state,
                         time,
                         bv,
                         nextProfile);
            ++nextProfile;
        }
    }

    const Diagnostics final = computeDiagnostics(mesh, constants, state);
    const PetscReal fickianRef = initial.sigma2 + 2.0 * input.tf;
    const PetscReal massLoss =
        PetscAbsReal(final.mass - initial.mass) / PetscMax(PetscAbsReal(initial.mass), 1.0e-30);

    if (input.verbose) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                              "  MMS4TBGC Bv=%.6e nx=%d: sigma2(tf)=%.6e, "
                              "fickian_ref=%.6e, mass_drift=%.3e, saturation=%s\n",
                              static_cast<double>(bv),
                              static_cast<int>(nx),
                              static_cast<double>(final.sigma2),
                              static_cast<double>(fickianRef),
                              static_cast<double>(massLoss),
                              saturationFlagged ? "yes" : "no"));
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

int main(int argc, char** argv) {
    PetscCallAbort(PETSC_COMM_WORLD,
                   PetscInitialize(&argc,
                                   &argv,
                                   nullptr,
                                   "MMS4TBGC -- Gaussian-pulse spreading under the TBGC scheme"));

    PetscInt exitCode = 0;
    try {
        const std::filesystem::path caseDir = caseDirectory();
        const std::filesystem::path dataDir = caseDir / "Dados";
        const InputData input = bgc::models::bgc::readMMSInputData(dataDir, defaultInput());
        const std::filesystem::path outputDir =
            bgc::models::bgc::resolveMMSOutputDirectory(caseDir, input.outputDir);
        bgc::ensureDirectory(outputDir);

        PetscCallAbort(PETSC_COMM_WORLD, bgc::models::bgc::printCpuUsageSummary());
        PetscPrintf(PETSC_COMM_WORLD,
                    "\nMMS4TBGC -- TBGC Gaussian-pulse spreading\n"
                    "  phi(x,0) = [1/(sigma0*sqrt(2*pi))] * exp[-(x-mu)^2/(2*sigma0^2)]\n"
                    "  mu(x,0)  = phi(x,0) - Bv * phi_xx(x,0)\n"
                    "  mu_center = %.6e  sigma0 = %.6e\n"
                    "  boundary: homogeneous Neumann + Dirichlet at west/east\n"
                    "  input : %s\n"
                    "  output: %s\n",
                    static_cast<double>(kMuCenter),
                    static_cast<double>(kSigma0),
                    dataDir.string().c_str(),
                    outputDir.string().c_str());

        for (const PetscReal bv : input.bvList) {
            for (const PetscInt nx : input.nxList) {
                PetscCallAbort(PETSC_COMM_WORLD,
                               runGaussianCampaign(input, outputDir, bv, nx));
            }
        }
    } catch (const std::exception& ex) {
        PetscPrintf(PETSC_COMM_WORLD, "\nERROR: %s\n", ex.what());
        exitCode = 1;
    }

    PetscCallAbort(PETSC_COMM_WORLD, PetscFinalize());
    return exitCode;
}
