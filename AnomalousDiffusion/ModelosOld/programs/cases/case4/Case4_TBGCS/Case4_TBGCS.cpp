#include <bgclib/Core/BoundaryCondition.hpp>
#include <bgclib/Models/TBGCS/AssemblyTBGCS.hpp>
#include <bgclib/Models/TBGCS/CoeffTBGCS.hpp>
#include <bgclib/SimConfig.hpp>
#include <bgclib/SimState.hpp>
#include <bgclib/Solver.hpp>
#include <bgclib/PostProcess.hpp>

#include <petsc.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Input data
// ---------------------------------------------------------------------------

struct InputData {
    PetscReal tf    = 5.0e-3;
    PetscReal cdt   = 1.024e-1;
    PetscInt  nx    = 512;
    PetscReal bv    = 0.0;
    PetscReal sigma = 4.0e-2;   // sigma_0 in the paper
    PetscReal mu0   = 5.0e-1;   // centre of the Gaussian
    std::vector<PetscReal> saveTimes = {
        0.0,
        1.0e-4,
        5.0e-4,
        1.0e-3,
        5.0e-3
    };
};

// ---------------------------------------------------------------------------
// String utilities
// ---------------------------------------------------------------------------

std::string trim(const std::string& text) {
    const auto begin = std::find_if_not(text.begin(),
                                        text.end(),
                                        [](unsigned char ch) {
                                            return std::isspace(ch) != 0;
                                        });
    if (begin == text.end()) {
        return "";
    }

    const auto end = std::find_if_not(text.rbegin(),
                                      text.rend(),
                                      [](unsigned char ch) {
                                          return std::isspace(ch) != 0;
                                      }).base();

    return std::string(begin, end);
}

std::string toLower(std::string text) {
    std::transform(text.begin(),
                   text.end(),
                   text.begin(),
                   [](unsigned char ch) {
                       return static_cast<char>(std::tolower(ch));
                   });
    return text;
}

std::vector<std::string> splitByComma(const std::string& text) {
    std::vector<std::string> parts;
    std::stringstream ss(text);
    std::string item;

    while (std::getline(ss, item, ',')) {
        parts.push_back(trim(item));
    }

    return parts;
}

PetscReal parsePetscReal(const std::string& text) {
    const std::string cleaned = trim(text);
    const char* begin = cleaned.c_str();
    char* end = nullptr;

    const PetscReal value =
        static_cast<PetscReal>(std::strtod(begin, &end));

    if (end == begin) {
        throw std::runtime_error("Invalid PetscReal value: " + cleaned);
    }

    return value;
}

PetscInt parsePetscInt(const std::string& text) {
    const std::string cleaned = trim(text);
    const char* begin = cleaned.c_str();
    char* end = nullptr;

    const long value = std::strtol(begin, &end, 10);

    if (end == begin) {
        throw std::runtime_error("Invalid PetscInt value: " + cleaned);
    }

    return static_cast<PetscInt>(value);
}

std::vector<PetscReal> parsePetscRealList(const std::string& text) {
    const std::vector<std::string> parts = splitByComma(text);
    std::vector<PetscReal> values;
    values.reserve(parts.size());

    for (const std::string& part : parts) {
        if (!part.empty()) {
            values.push_back(parsePetscReal(part));
        }
    }

    if (values.empty()) {
        throw std::runtime_error("Empty PetscReal list.");
    }

    return values;
}

// ---------------------------------------------------------------------------
// File I/O helpers
// ---------------------------------------------------------------------------

std::unordered_map<std::string, std::string>
readKeyValueFile(const std::filesystem::path& filePath) {
    std::ifstream file(filePath);

    if (!file.is_open()) {
        throw std::runtime_error(
            "Could not open input file: " + filePath.string());
    }

    std::unordered_map<std::string, std::string> data;
    std::string line;
    PetscInt lineNumber = 0;

    while (std::getline(file, line)) {
        ++lineNumber;

        const std::size_t commentPosHash  = line.find('#');
        const std::size_t commentPosSlash = line.find("//");
        std::size_t commentPos = std::string::npos;

        if (commentPosHash != std::string::npos) {
            commentPos = commentPosHash;
        }
        if (commentPosSlash != std::string::npos) {
            if (commentPos == std::string::npos ||
                commentPosSlash < commentPos) {
                commentPos = commentPosSlash;
            }
        }

        if (commentPos != std::string::npos) {
            line = line.substr(0, commentPos);
        }

        line = trim(line);

        if (line.empty()) {
            continue;
        }

        const std::size_t eqPos = line.find('=');
        if (eqPos == std::string::npos) {
            throw std::runtime_error(
                "Invalid line in file " + filePath.string() +
                " at line " + std::to_string(lineNumber) +
                ". Expected 'key = value'.");
        }

        const std::string key   = toLower(trim(line.substr(0, eqPos)));
        const std::string value = trim(line.substr(eqPos + 1));

        if (key.empty()) {
            throw std::runtime_error(
                "Empty key in file " + filePath.string() +
                " at line " + std::to_string(lineNumber) + ".");
        }

        data[key] = value;
    }

    return data;
}

std::filesystem::path findDataDirectory() {
    const std::filesystem::path dataDir =
        std::filesystem::current_path() / "Dados";

    if (std::filesystem::exists(dataDir) &&
        std::filesystem::is_directory(dataDir)) {
        return std::filesystem::canonical(dataDir);
    }

    throw std::runtime_error(
        "Could not locate the Dados directory at: " +
        dataDir.string());
}

InputData readInputData(const std::filesystem::path& dataDir) {
    InputData input;

    const std::filesystem::path simFile = dataDir / "simulation.dat";
    const auto simData = readKeyValueFile(simFile);

    if (simData.contains("tf"))         input.tf    = parsePetscReal(simData.at("tf"));
    if (simData.contains("cdt"))        input.cdt   = parsePetscReal(simData.at("cdt"));
    if (simData.contains("nx"))         input.nx    = parsePetscInt (simData.at("nx"));
    if (simData.contains("bv"))         input.bv    = parsePetscReal(simData.at("bv"));
    if (simData.contains("sigma"))      input.sigma = parsePetscReal(simData.at("sigma"));
    if (simData.contains("mu0"))        input.mu0   = parsePetscReal(simData.at("mu0"));
    if (simData.contains("save_times")) input.saveTimes = parsePetscRealList(simData.at("save_times"));

    return input;
}

// ---------------------------------------------------------------------------
// Global parameters (set from InputData before use)
// ---------------------------------------------------------------------------

PetscReal gSigma = 4.0e-2;
PetscReal gMu0   = 5.0e-1;

// ---------------------------------------------------------------------------
// Simulation configuration
// ---------------------------------------------------------------------------

bgc::SimConfig makeConfig(const InputData& input) {
    bgc::SimConfig cfg;

    cfg.model = bgc::Model::TBGC;
    cfg.nx    = input.nx;
    cfg.lx    = 1.0;
    cfg.x0    = 0.0;
    cfg.h     = cfg.lx / static_cast<PetscReal>(cfg.nx);
    cfg.bv    = input.bv;
    cfg.tf    = input.tf;
    cfg.dt    = input.cdt * cfg.h * cfg.h;
    cfg.nTimes = static_cast<PetscInt>(cfg.tf / cfg.dt + 0.5);

    if (cfg.nTimes < 1) {
        cfg.nTimes = 1;
    }

    cfg.dt = cfg.tf / static_cast<PetscReal>(cfg.nTimes);

    cfg.useDirectSolver     = true;
    cfg.verbose             = false;
    cfg.directSolverBackend = bgc::DirectSolverBackend::PetscDefault;

    // Homogeneous Dirichlet (phi=0) and Neumann (dphi/dx=0) at both ends.
    // The Neumann condition is listed first (k=1) and Dirichlet second (k=2),
    // matching the convention used throughout the BGC library.
    cfg.bcWest[0] = bgc::BoundaryCondition::neumann(0.0);
    cfg.bcWest[1] = bgc::BoundaryCondition::dirichlet(0.0);

    cfg.bcEast[0] = bgc::BoundaryCondition::neumann(0.0);
    cfg.bcEast[1] = bgc::BoundaryCondition::dirichlet(0.0);

    return cfg;
}

// ---------------------------------------------------------------------------
// Initial condition: normalised Gaussian centred at gMu0 with width gSigma
//
//   phi(x,0) = (1 / (gSigma * sqrt(2*pi))) * exp(-(x - gMu0)^2 / (2*gSigma^2))
//
// Second derivative used to initialise mu:
//
//   phi_xx(x) = phi(x) * ( (x - gMu0)^2 / gSigma^4  -  1 / gSigma^2 )
// ---------------------------------------------------------------------------

PetscReal initialPhiFn(const bgc::SimConfig&, PetscReal x, PetscReal) {
    const PetscReal sigma2  = gSigma * gSigma;
    const PetscReal q       = x - gMu0;
    const PetscReal norm    = 1.0 / (gSigma * std::sqrt(2.0 * M_PI));
    return norm * PetscExpReal(-q * q / (2.0 * sigma2));
}

PetscReal initialPhiXX(PetscReal x) {
    const PetscReal sigma2  = gSigma * gSigma;
    const PetscReal q       = x - gMu0;
    const PetscReal phi     = initialPhiFn(bgc::SimConfig{}, x, 0.0);
    return phi * (q * q / (sigma2 * sigma2) - 1.0 / sigma2);
}

PetscReal initialMuFn(const bgc::SimConfig& cfg, PetscReal x, PetscReal t) {
    (void)t;
    return initialPhiFn(cfg, x, 0.0) - cfg.bv * initialPhiXX(x);
}

PetscReal zeroSourceFn(const bgc::SimConfig&, PetscReal, PetscReal) {
    return 0.0;
}

// ---------------------------------------------------------------------------
// Vector fill helper
// ---------------------------------------------------------------------------

PetscErrorCode fillVectorFromField(const bgc::SimConfig& cfg,
                                   Vec                   v,
                                   PetscReal             t,
                                   PetscReal (*fieldFn)(const bgc::SimConfig&,
                                                        PetscReal,
                                                        PetscReal)) {
    PetscFunctionBeginUser;

    PetscInt iStart = 0;
    PetscInt iEnd   = 0;
    PetscCall(VecGetOwnershipRange(v, &iStart, &iEnd));

    for (PetscInt i = iStart; i < iEnd; ++i) {
        const PetscReal   x     = cfg.xCenter(i);
        const PetscScalar value = static_cast<PetscScalar>(fieldFn(cfg, x, t));
        PetscCall(VecSetValue(v, i, value, INSERT_VALUES));
    }

    PetscCall(VecAssemblyBegin(v));
    PetscCall(VecAssemblyEnd(v));

    PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------------------------------------------
// Save-step set
// ---------------------------------------------------------------------------

std::set<PetscInt> buildSaveSteps(const bgc::SimConfig&         cfg,
                                  const std::vector<PetscReal>& saveTimes) {
    std::set<PetscInt> steps;
    steps.insert(0);
    steps.insert(cfg.nTimes);

    for (const PetscReal t : saveTimes) {
        PetscInt step = static_cast<PetscInt>(std::llround(t / cfg.dt));

        if (step < 0)            step = 0;
        if (step > cfg.nTimes)   step = cfg.nTimes;

        steps.insert(step);
    }

    return steps;
}

std::string makeStepTag(PetscInt step) {
    std::ostringstream oss;
    oss << "step_" << std::setw(6) << std::setfill('0') << step;
    return oss.str();
}

// ---------------------------------------------------------------------------
// CSV profile writer
// ---------------------------------------------------------------------------

PetscErrorCode writeVectorCSV(const bgc::SimConfig&        cfg,
                               Vec                          v,
                               const std::filesystem::path& filePath,
                               PetscReal                    time,
                               const std::string&           columnName) {
    PetscFunctionBeginUser;

    PetscMPIInt rank = 0;
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

    VecScatter scatter = nullptr;
    Vec        seq     = nullptr;

    PetscCall(VecScatterCreateToZero(v, &scatter, &seq));
    PetscCall(VecScatterBegin(scatter, v, seq, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd  (scatter, v, seq, INSERT_VALUES, SCATTER_FORWARD));

    if (rank == 0) {
        const PetscScalar* values = nullptr;
        PetscCall(VecGetArrayRead(seq, &values));

        std::ofstream out(filePath);
        if (!out.is_open()) {
            PetscCall(VecRestoreArrayRead(seq, &values));
            PetscCall(VecScatterDestroy(&scatter));
            PetscCall(VecDestroy(&seq));
            throw std::runtime_error(
                "Could not open output file: " + filePath.string());
        }

        out << std::scientific << std::setprecision(16);
        out << "# time = " << static_cast<double>(time) << "\n";
        out << "# nx = "   << cfg.nx << "\n";
        out << "i,x," << columnName << "\n";

        // Boundary values (Dirichlet = 0) prepended/appended for phi only
        if (columnName == "phi") {
            out << -1 << ","
                << static_cast<double>(cfg.x0) << ","
                << 0.0 << "\n";
        }

        for (PetscInt i = 0; i < cfg.nx; ++i) {
            out << i << ","
                << static_cast<double>(cfg.xCenter(i)) << ","
                << static_cast<double>(PetscRealPart(values[i])) << "\n";
        }

        if (columnName == "phi") {
            out << cfg.nx << ","
                << static_cast<double>(cfg.x0 + cfg.lx) << ","
                << 0.0 << "\n";
        }

        PetscCall(VecRestoreArrayRead(seq, &values));
    }

    PetscCall(VecScatterDestroy(&scatter));
    PetscCall(VecDestroy(&seq));

    PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------------------------------------------
// Discrete free energy
//
//   F_h = 0.5 * [ h * sum(phi_i^2)
//               + Bv * sum_{i=0}^{N-2} h * ((phi_{i+1} - phi_i)/h)^2
//               + Bv * 0.5 * h * (2*phi_0/h)^2          (left ghost face)
//               + Bv * 0.5 * h * (2*phi_{N-1}/h)^2 ]    (right ghost face)
//
// The boundary-face gradient terms use the Dirichlet ghost-node values
// phi(-h/2) = 0 and phi(1+h/2) = 0, giving gradients 2*phi_0/h and
// -2*phi_{N-1}/h respectively, consistent with the energy-test benchmark.
// ---------------------------------------------------------------------------

PetscErrorCode computeDiscreteFreeEnergy(const bgc::SimConfig& cfg,
                                         Vec                   phi,
                                         PetscReal*            freeEnergy) {
    PetscFunctionBeginUser;

    PetscMPIInt rank = 0;
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

    VecScatter scatter = nullptr;
    Vec        seq     = nullptr;

    PetscCall(VecScatterCreateToZero(phi, &scatter, &seq));
    PetscCall(VecScatterBegin(scatter, phi, seq, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd  (scatter, phi, seq, INSERT_VALUES, SCATTER_FORWARD));

    PetscReal energyValue = 0.0;

    if (rank == 0) {
        const PetscScalar* values = nullptr;
        PetscCall(VecGetArrayRead(seq, &values));

        // Mass term
        PetscReal massIntegral = 0.0;
        for (PetscInt i = 0; i < cfg.nx; ++i) {
            const PetscReal p = PetscRealPart(values[i]);
            massIntegral += p * p;
        }
        massIntegral *= cfg.h;

        // Gradient term (interior faces)
        PetscReal gradIntegral = 0.0;
        for (PetscInt i = 0; i < cfg.nx - 1; ++i) {
            const PetscReal grad =
                (PetscRealPart(values[i + 1]) - PetscRealPart(values[i])) / cfg.h;
            gradIntegral += cfg.h * grad * grad;
        }

        // Boundary face contributions (ghost-node Dirichlet = 0)
        const PetscReal leftGrad  =  2.0 * PetscRealPart(values[0])          / cfg.h;
        const PetscReal rightGrad = -2.0 * PetscRealPart(values[cfg.nx - 1]) / cfg.h;
        gradIntegral += 0.5 * cfg.h * leftGrad  * leftGrad;
        gradIntegral += 0.5 * cfg.h * rightGrad * rightGrad;

        energyValue = 0.5 * (massIntegral + cfg.bv * gradIntegral);

        PetscCall(VecRestoreArrayRead(seq, &values));
    }

    PetscCallMPI(MPI_Bcast(&energyValue, 1, MPIU_REAL, 0, PETSC_COMM_WORLD));

    PetscCall(VecScatterDestroy(&scatter));
    PetscCall(VecDestroy(&seq));

    *freeEnergy = energyValue;

    PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------------------------------------------
// Normalised second centred moment
//
//   sigma2 = h * sum((xi - mu0)^2 * phi_i) / (h * sum(phi_i))
//
// Dividing by the discrete total mass makes the diagnostic robust to the
// residual mass drift introduced by the Dirichlet boundary conditions.
// ---------------------------------------------------------------------------

PetscErrorCode computeSecondMoment(const bgc::SimConfig& cfg,
                                   Vec                   phi,
                                   PetscReal             mu0,
                                   PetscReal*            sigma2) {
    PetscFunctionBeginUser;

    PetscMPIInt rank = 0;
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

    VecScatter scatter = nullptr;
    Vec        seq     = nullptr;

    PetscCall(VecScatterCreateToZero(phi, &scatter, &seq));
    PetscCall(VecScatterBegin(scatter, phi, seq, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd  (scatter, phi, seq, INSERT_VALUES, SCATTER_FORWARD));

    PetscReal result = 0.0;

    if (rank == 0) {
        const PetscScalar* values = nullptr;
        PetscCall(VecGetArrayRead(seq, &values));

        PetscReal numerator   = 0.0;
        PetscReal denominator = 0.0;

        for (PetscInt i = 0; i < cfg.nx; ++i) {
            const PetscReal xi  = cfg.xCenter(i);
            const PetscReal phi_i = PetscRealPart(values[i]);
            const PetscReal q   = xi - mu0;
            numerator   += q * q * phi_i;
            denominator += phi_i;
        }

        // Both are multiplied by h, which cancels in the ratio.
        if (std::abs(denominator) > 0.0) {
            result = numerator / denominator;
        }

        PetscCall(VecRestoreArrayRead(seq, &values));
    }

    PetscCallMPI(MPI_Bcast(&result, 1, MPIU_REAL, 0, PETSC_COMM_WORLD));

    PetscCall(VecScatterDestroy(&scatter));
    PetscCall(VecDestroy(&seq));

    *sigma2 = result;

    PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------------------------------------------
// Negative-value counter
// ---------------------------------------------------------------------------

PetscErrorCode countNegativeEntries(Vec       v,
                                    PetscReal lowerTol,
                                    PetscInt* negativeCount) {
    PetscFunctionBeginUser;

    PetscInt localSize = 0;
    PetscCall(VecGetLocalSize(v, &localSize));

    const PetscScalar* values = nullptr;
    PetscCall(VecGetArrayRead(v, &values));

    PetscInt localNeg = 0;
    for (PetscInt i = 0; i < localSize; ++i) {
        if (PetscRealPart(values[i]) < lowerTol) {
            ++localNeg;
        }
    }

    PetscCall(VecRestoreArrayRead(v, &values));

    PetscCallMPI(MPI_Allreduce(&localNeg,
                               negativeCount,
                               1,
                               MPIU_INT,
                               MPI_SUM,
                               PETSC_COMM_WORLD));

    PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------------------------------------------
// History file
// ---------------------------------------------------------------------------

PetscErrorCode initialiseHistoryFile(const std::filesystem::path& filePath) {
    PetscFunctionBeginUser;

    PetscMPIInt rank = 0;
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

    if (rank == 0) {
        std::ofstream out(filePath);
        if (!out.is_open()) {
            throw std::runtime_error(
                "Could not open history file: " + filePath.string());
        }
        out << "step,time,free_energy,free_energy_ratio,"
               "sigma2,phi_min,phi_max,negative_count\n";
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode appendHistoryLine(const std::filesystem::path& filePath,
                                 PetscInt                     step,
                                 PetscReal                    time,
                                 PetscReal                    freeEnergy,
                                 PetscReal                    freeEnergyRatio,
                                 PetscReal                    sigma2,
                                 PetscReal                    phiMin,
                                 PetscReal                    phiMax,
                                 PetscInt                     negativeCount) {
    PetscFunctionBeginUser;

    PetscMPIInt rank = 0;
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

    if (rank == 0) {
        std::ofstream out(filePath, std::ios::app);
        if (!out.is_open()) {
            throw std::runtime_error(
                "Could not append history file: " + filePath.string());
        }

        out << std::scientific << std::setprecision(16)
            << step              << ","
            << static_cast<double>(time)            << ","
            << static_cast<double>(freeEnergy)      << ","
            << static_cast<double>(freeEnergyRatio) << ","
            << static_cast<double>(sigma2)          << ","
            << static_cast<double>(phiMin)          << ","
            << static_cast<double>(phiMax)          << ","
            << negativeCount << "\n";
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------------------------------------------
// Metadata writer
// ---------------------------------------------------------------------------

PetscErrorCode writeMetadata(const bgc::SimConfig&        cfg,
                             const std::filesystem::path& outDir) {
    PetscFunctionBeginUser;

    PetscMPIInt rank = 0;
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

    if (rank == 0) {
        const std::filesystem::path filePath = outDir / "case_summary.txt";
        std::ofstream out(filePath);

        if (!out.is_open()) {
            throw std::runtime_error(
                "Could not open metadata file: " + filePath.string());
        }

        out << std::scientific << std::setprecision(16);
        out << "Model: TBGCS\n";
        out << "Case: Gaussian-pulse benchmark (Section 9)\n";
        out << "Source term: 0\n";
        out << "Bv: "      << static_cast<double>(cfg.bv)  << "\n";
        out << "nx: "      << cfg.nx                        << "\n";
        out << "h: "       << static_cast<double>(cfg.h)   << "\n";
        out << "dt: "      << static_cast<double>(cfg.dt)  << "\n";
        out << "nTimes: "  << cfg.nTimes                    << "\n";
        out << "tf: "      << static_cast<double>(cfg.tf)  << "\n";
        out << "BC west: Dirichlet = 0.0, Neumann = 0.0\n";
        out << "BC east: Dirichlet = 0.0, Neumann = 0.0\n";
        out << "Initial condition:\n";
        out << "  phi(x,0) = (1 / (sigma * sqrt(2*pi)))"
               " * exp(-(x - mu0)^2 / (2*sigma^2))\n";
        out << "  mu0   = " << static_cast<double>(gMu0)   << "\n";
        out << "  sigma = " << static_cast<double>(gSigma) << "\n";
        out << "Initial auxiliary field:\n";
        out << "  mu(x,0) = phi(x,0) - Bv * phi_xx(x,0)\n";
        out << "  phi_xx(x,0) = phi(x,0) * ((x-mu0)^2/sigma^4 - 1/sigma^2)\n";
        out << "Recorded free energy:\n";
        out << "  F_h = 0.5 * [ h*sum(phi_i^2)"
               " + Bv * integral_h((dphi/dx)^2) ]\n";
        out << "  Boundary faces use ghost-node Dirichlet = 0.\n";
        out << "Second moment:\n";
        out << "  sigma2 = sum((xi-mu0)^2 * phi_i) / sum(phi_i)\n";
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------------------------------------------
// State initialisation
// ---------------------------------------------------------------------------

PetscErrorCode initialiseState(const bgc::SimConfig& cfg,
                               bgc::SimState&        st) {
    PetscFunctionBeginUser;

    PetscCall(fillVectorFromField(cfg, st.phi, 0.0, initialPhiFn));
    PetscCall(fillVectorFromField(cfg, st.mu,  0.0, initialMuFn));
    PetscCall(VecCopy(st.phi, st.phi_0));

    PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------------------------------------------
// Time-march
// ---------------------------------------------------------------------------

PetscErrorCode solveProblem(const bgc::SimConfig&         cfg,
                            bgc::SimState&                st,
                            const std::set<PetscInt>&     saveSteps,
                            const std::filesystem::path&  outDir) {
    PetscFunctionBeginUser;

    const std::filesystem::path historyFile = outDir / "history_TBGCS.csv";

    PetscCall(initialiseState(cfg, st));
    PetscCall(initialiseHistoryFile(historyFile));

    // Initial diagnostics
    PetscReal freeEnergy0 = 0.0;
    PetscCall(computeDiscreteFreeEnergy(cfg, st.phi, &freeEnergy0));

    {
        PetscInt  posMin        = 0;
        PetscInt  posMax        = 0;
        PetscReal phiMin        = 0.0;
        PetscReal phiMax        = 0.0;
        PetscInt  negativeCount = 0;
        PetscReal sigma2        = 0.0;

        PetscCall(VecMin(st.phi, &posMin, &phiMin));
        PetscCall(VecMax(st.phi, &posMax, &phiMax));
        PetscCall(countNegativeEntries(st.phi,
                                       static_cast<PetscReal>(-1.0e-12),
                                       &negativeCount));
        PetscCall(computeSecondMoment(cfg, st.phi, gMu0, &sigma2));

        PetscCall(appendHistoryLine(historyFile,
                                    0,
                                    0.0,
                                    freeEnergy0,
                                    1.0,   // ratio at t=0 is always 1
                                    sigma2,
                                    phiMin,
                                    phiMax,
                                    negativeCount));
    }

    if (saveSteps.contains(0)) {
        PetscCall(writeVectorCSV(cfg,
                                 st.phi,
                                 outDir / (makeStepTag(0) + "_phi.csv"),
                                 0.0,
                                 "phi"));
        PetscCall(writeVectorCSV(cfg,
                                 st.mu,
                                 outDir / (makeStepTag(0) + "_mu.csv"),
                                 0.0,
                                 "mu"));
    }

    // Time-march loop
    for (PetscInt step = 1; step <= cfg.nTimes; ++step) {
        const PetscReal timeNp1 = static_cast<PetscReal>(step) * cfg.dt;

        const auto rhs = bgc::computeRHSTBGCS(cfg, timeNp1);
        PetscCall(bgc::assembleRHSTBGCS(cfg, st, rhs));

        PetscCall(bgc::computeSourceTerm(cfg, st, timeNp1, zeroSourceFn));
        PetscCall(VecAXPY(st.b1, 1.0, st.b1Source));

        PetscCall(bgc::solveLinearSystem(cfg, st));

        PetscInt  posMin        = 0;
        PetscInt  posMax        = 0;
        PetscReal phiMin        = 0.0;
        PetscReal phiMax        = 0.0;
        PetscInt  negativeCount = 0;
        PetscReal freeEnergy    = 0.0;
        PetscReal sigma2        = 0.0;

        PetscCall(VecMin(st.phi, &posMin, &phiMin));
        PetscCall(VecMax(st.phi, &posMax, &phiMax));
        PetscCall(countNegativeEntries(st.phi,
                                       static_cast<PetscReal>(-1.0e-12),
                                       &negativeCount));
        PetscCall(computeDiscreteFreeEnergy(cfg, st.phi, &freeEnergy));
        PetscCall(computeSecondMoment(cfg, st.phi, gMu0, &sigma2));

        const PetscReal ratio =
            (freeEnergy0 > 0.0) ? freeEnergy / freeEnergy0 : 0.0;

        PetscCall(appendHistoryLine(historyFile,
                                    step,
                                    timeNp1,
                                    freeEnergy,
                                    ratio,
                                    sigma2,
                                    phiMin,
                                    phiMax,
                                    negativeCount));

        if (negativeCount > 0) {
            PetscPrintf(PETSC_COMM_WORLD,
                        "[INFO] step=%" PetscInt_FMT
                        "  time=%.6e"
                        "  phi_min=%.6e"
                        "  negative_count=%" PetscInt_FMT "\n",
                        step,
                        static_cast<double>(timeNp1),
                        static_cast<double>(phiMin),
                        negativeCount);
        }

        if (saveSteps.contains(step)) {
            PetscCall(writeVectorCSV(cfg,
                                     st.phi,
                                     outDir / (makeStepTag(step) + "_phi.csv"),
                                     timeNp1,
                                     "phi"));
            PetscCall(writeVectorCSV(cfg,
                                     st.mu,
                                     outDir / (makeStepTag(step) + "_mu.csv"),
                                     timeNp1,
                                     "mu"));
        }

        if (step < cfg.nTimes) {
            PetscCall(VecCopy(st.phi, st.phi_0));
        }
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    PetscCallAbort(PETSC_COMM_WORLD,
                   PetscInitialize(&argc,
                                   &argv,
                                   nullptr,
                                   "Case3_TBGCS -- Gaussian-pulse benchmark"));

    PetscInt exitCode = 0;

    try {
        const std::filesystem::path dataDir = findDataDirectory();
        const InputData             input   = readInputData(dataDir);
        const bgc::SimConfig        cfg     = makeConfig(input);

        gSigma = input.sigma;
        gMu0   = input.mu0;

        const std::filesystem::path outDir =
            dataDir.parent_path() / "Saida";
        std::filesystem::create_directories(outDir);

        
        PetscPrintf(PETSC_COMM_WORLD,
                    "\n######################################################################\n"
                    "  Case3_TBGCS -- Gaussian-pulse benchmark (Section 9)\n"
                    "  phi(x,0) = (1/(sigma*sqrt(2*pi))) * exp(-(x-mu0)^2/(2*sigma^2))\n"
                    "  mu0   = %.6e\n"
                    "  sigma = %.6e\n"
                    "  Bv    = %.6e\n"
                    "  Output directory: %s\n"
                    "######################################################################\n",
                    static_cast<double>(gMu0),
                    static_cast<double>(gSigma),
                    static_cast<double>(cfg.bv),
                    outDir.string().c_str());

        PetscPrintf(PETSC_COMM_WORLD,
                    "nx=%" PetscInt_FMT
                    "  h=%.6e"
                    "  dt=%.6e"
                    "  nTimes=%" PetscInt_FMT
                    "  tf=%.6e\n",
                    cfg.nx,
                    static_cast<double>(cfg.h),
                    static_cast<double>(cfg.dt),
                    cfg.nTimes,
                    static_cast<double>(cfg.tf));

        bgc::SimState st;
        PetscCallAbort(PETSC_COMM_WORLD, bgc::createStateTBGC(cfg, st));

        const auto sc = bgc::computeCoefficientsTBGCS(cfg);
        PetscCallAbort(PETSC_COMM_WORLD, bgc::assembleMatrixTBGCS(cfg, st, sc));
        PetscCallAbort(PETSC_COMM_WORLD, bgc::configureSolver(cfg, st));

        const std::set<PetscInt> saveSteps = buildSaveSteps(cfg, input.saveTimes);

        PetscCallAbort(PETSC_COMM_WORLD, writeMetadata(cfg, outDir));
        PetscCallAbort(PETSC_COMM_WORLD,
                       solveProblem(cfg, st, saveSteps, outDir));

        PetscCallAbort(PETSC_COMM_WORLD, bgc::destroyState(st));

        PetscPrintf(PETSC_COMM_WORLD,
                    "\nSimulation completed successfully.\n");

    } catch (const std::exception& ex) {
        PetscPrintf(PETSC_COMM_WORLD,
                    "\nERROR: %s\n",
                    ex.what());
        exitCode = 1;
    }

    PetscCallAbort(PETSC_COMM_WORLD, PetscFinalize());
    return exitCode;
}