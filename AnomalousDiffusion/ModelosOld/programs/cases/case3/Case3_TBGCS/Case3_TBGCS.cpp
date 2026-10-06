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

struct InputData {
    PetscReal tf = 1.0e-2;
    PetscReal cdt = 1.024e-1;
    PetscInt nx = 16;
    PetscReal bv = 10.0;
    PetscReal sigma = 1.0e-1;
    std::vector<PetscReal> saveTimes = {
        0.0,
        1.0e-5,
        5.0e-5,
        1.0e-4,
        5.0e-4,
        1.0e-3,
        5.0e-3,
        1.0e-2
    };
};

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

        const std::size_t commentPosHash = line.find('#');
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

        const std::string key = toLower(trim(line.substr(0, eqPos)));
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

    if (simData.contains("tf")) {
        input.tf = parsePetscReal(simData.at("tf"));
    }
    if (simData.contains("cdt")) {
        input.cdt = parsePetscReal(simData.at("cdt"));
    }
    if (simData.contains("nx")) {
        input.nx = parsePetscInt(simData.at("nx"));
    }
    if (simData.contains("bv")) {
        input.bv = parsePetscReal(simData.at("bv"));
    }
    if (simData.contains("sigma")) {
        input.sigma = parsePetscReal(simData.at("sigma"));
    }
    if (simData.contains("save_times")) {
        input.saveTimes = parsePetscRealList(simData.at("save_times"));
    }

    return input;
}

constexpr PetscReal kDefaultSigma = 2.0e-2;
constexpr PetscReal kXiC = 1.0e-1;
PetscReal gSigma = kDefaultSigma;

bgc::SimConfig makeConfig(const InputData& input) {
    bgc::SimConfig cfg;

    cfg.model = bgc::Model::TBGC;
    cfg.nx = input.nx;
    cfg.lx = 1.0;
    cfg.x0 = 0.0;
    cfg.h = cfg.lx / static_cast<PetscReal>(cfg.nx);
    cfg.bv = input.bv;
    cfg.tf = input.tf;
    cfg.dt = input.cdt * cfg.h * cfg.h;
    cfg.nTimes = static_cast<PetscInt>(cfg.tf / cfg.dt + 0.5);

    if (cfg.nTimes < 1) {
        cfg.nTimes = 1;
    }

    cfg.dt = cfg.tf / static_cast<PetscReal>(cfg.nTimes);

    cfg.useDirectSolver = true;
    cfg.verbose = false;
    cfg.directSolverBackend = bgc::DirectSolverBackend::PetscDefault;

    cfg.bcWest[0] = bgc::BoundaryCondition::neumann(0.0);
    cfg.bcWest[1] = bgc::BoundaryCondition::dirichlet(0.0);

    cfg.bcEast[0] = bgc::BoundaryCondition::neumann(0.0);
    cfg.bcEast[1] = bgc::BoundaryCondition::dirichlet(0.0);

    return cfg;
}

PetscReal initialPhiFn(const bgc::SimConfig&, PetscReal x, PetscReal) {
    const PetscReal s = std::sin(M_PI * x);
    return s * s;    
    // const PetscReal q = x - kXiC;
    // const PetscReal polynomial = x * x * (1.0 - x) * (1.0 - x);
    // const PetscReal gaussian =
    //     PetscExpReal(-(q * q) / (2.0 * gSigma * gSigma));
    // return polynomial * gaussian;
}

PetscReal initialPhiXX(PetscReal x) {
    return 2.0 * M_PI * M_PI * std::cos(2.0 * M_PI * x);
//     const PetscReal sigma2 = gSigma * gSigma;
//     const PetscReal invSigma2 = 1.0 / sigma2;
//     const PetscReal q = x - kXiC;
//     const PetscReal g = x * x * (1.0 - x) * (1.0 - x);
//     const PetscReal gp =
//         2.0 * x - 6.0 * x * x + 4.0 * x * x * x;
//     const PetscReal gpp =
//         2.0 - 12.0 * x + 12.0 * x * x;
//     const PetscReal gaussian =
//         PetscExpReal(-(q * q) / (2.0 * sigma2));
//     return gaussian *
//            (gpp
//             - 2.0 * gp * q * invSigma2
//             + g * (q * q * invSigma2 * invSigma2 - invSigma2));
}

PetscReal initialMuFn(const bgc::SimConfig& cfg, PetscReal x, PetscReal t) {
    (void)t;
    return initialPhiFn(cfg, x, 0.0) - cfg.bv * initialPhiXX(x);
}

PetscReal zeroSourceFn(const bgc::SimConfig&, PetscReal, PetscReal) {
    return 0.0;
}

PetscErrorCode fillVectorFromField(const bgc::SimConfig& cfg,
                                   Vec                   v,
                                   PetscReal             t,
                                   PetscReal (*fieldFn)(const bgc::SimConfig&,
                                                        PetscReal,
                                                        PetscReal)) {
    PetscFunctionBeginUser;

    PetscInt iStart = 0;
    PetscInt iEnd = 0;
    PetscCall(VecGetOwnershipRange(v, &iStart, &iEnd));

    for (PetscInt i = iStart; i < iEnd; ++i) {
        const PetscReal x = cfg.xCenter(i);
        const PetscScalar value =
            static_cast<PetscScalar>(fieldFn(cfg, x, t));
        PetscCall(VecSetValue(v, i, value, INSERT_VALUES));
    }

    PetscCall(VecAssemblyBegin(v));
    PetscCall(VecAssemblyEnd(v));

    PetscFunctionReturn(PETSC_SUCCESS);
}

std::set<PetscInt> buildSaveSteps(const bgc::SimConfig&         cfg,
                                  const std::vector<PetscReal>& saveTimes) {
    std::set<PetscInt> steps;
    steps.insert(0);
    steps.insert(cfg.nTimes);

    for (const PetscReal t : saveTimes) {
        PetscInt step =
            static_cast<PetscInt>(std::llround(t / cfg.dt));

        if (step < 0) {
            step = 0;
        }
        if (step > cfg.nTimes) {
            step = cfg.nTimes;
        }

        steps.insert(step);
    }

    return steps;
}

std::string makeStepTag(PetscInt step) {
    std::ostringstream oss;
    oss << "step_" << std::setw(6) << std::setfill('0') << step;
    return oss.str();
}

PetscErrorCode writeVectorCSV(const bgc::SimConfig&         cfg,
                              Vec                           v,
                              const std::filesystem::path&  filePath,
                              PetscReal                     time,
                              const std::string&            columnName) {
    PetscFunctionBeginUser;

    PetscMPIInt rank = 0;
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

    VecScatter scatter = nullptr;
    Vec seq = nullptr;

    PetscCall(VecScatterCreateToZero(v, &scatter, &seq));
    PetscCall(VecScatterBegin(scatter, v, seq, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(scatter, v, seq, INSERT_VALUES, SCATTER_FORWARD));

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
        out << "# nx = " << cfg.nx << "\n";
        out << "i,x," << columnName << "\n";

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

PetscErrorCode countNonPhysicalEntries(Vec        v,
                                       PetscReal  lowerTol,
                                       PetscReal  upperTol,
                                       PetscInt*  negativeCount,
                                       PetscInt*  aboveOneCount) {
    PetscFunctionBeginUser;

    PetscInt localSize = 0;
    PetscCall(VecGetLocalSize(v, &localSize));

    const PetscScalar* values = nullptr;
    PetscCall(VecGetArrayRead(v, &values));

    PetscInt localNeg = 0;
    PetscInt localAbove = 0;

    for (PetscInt i = 0; i < localSize; ++i) {
        const PetscReal value = PetscRealPart(values[i]);

        if (value < lowerTol) {
            ++localNeg;
        }
        if (value > upperTol) {
            ++localAbove;
        }
    }

    PetscCall(VecRestoreArrayRead(v, &values));

    PetscCallMPI(MPI_Allreduce(&localNeg,
                               negativeCount,
                               1,
                               MPIU_INT,
                               MPI_SUM,
                               PETSC_COMM_WORLD));

    PetscCallMPI(MPI_Allreduce(&localAbove,
                               aboveOneCount,
                               1,
                               MPIU_INT,
                               MPI_SUM,
                               PETSC_COMM_WORLD));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode computeDiscreteFreeEnergy(const bgc::SimConfig& cfg,
                                         Vec                   phi,
                                         PetscReal*            freeEnergy) {
    PetscFunctionBeginUser;

    PetscMPIInt rank = 0;
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

    VecScatter scatter = nullptr;
    Vec seq = nullptr;

    PetscCall(VecScatterCreateToZero(phi, &scatter, &seq));
    PetscCall(VecScatterBegin(scatter, phi, seq, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(scatter, phi, seq, INSERT_VALUES, SCATTER_FORWARD));

    PetscReal energyValue = 0.0;

    if (rank == 0) {
        const PetscScalar* values = nullptr;
        PetscCall(VecGetArrayRead(seq, &values));

        PetscReal massIntegral = 0.0;
        for (PetscInt i = 0; i < cfg.nx; ++i) {
            const PetscReal p = PetscRealPart(values[i]);
            massIntegral += p * p;
        }
        massIntegral *= cfg.h;

        PetscReal gradIntegral = 0.0;

        if (cfg.nx > 0) {
            const PetscReal leftGrad =
                2.0 * PetscRealPart(values[0]) / cfg.h;
            gradIntegral += 0.5 * cfg.h * leftGrad * leftGrad;

            for (PetscInt i = 0; i < cfg.nx - 1; ++i) {
                const PetscReal grad =
                    (PetscRealPart(values[i + 1]) -
                     PetscRealPart(values[i])) / cfg.h;
                gradIntegral += cfg.h * grad * grad;
            }

            const PetscReal rightGrad =
                -2.0 * PetscRealPart(values[cfg.nx - 1]) / cfg.h;
            gradIntegral += 0.5 * cfg.h * rightGrad * rightGrad;
        }

        energyValue = 0.5 * (massIntegral + cfg.bv * gradIntegral);

        PetscCall(VecRestoreArrayRead(seq, &values));
    }

    PetscCallMPI(MPI_Bcast(&energyValue,
                           1,
                           MPIU_REAL,
                           0,
                           PETSC_COMM_WORLD));

    PetscCall(VecScatterDestroy(&scatter));
    PetscCall(VecDestroy(&seq));

    *freeEnergy = energyValue;

    PetscFunctionReturn(PETSC_SUCCESS);
}

struct DissipationDiag {
    PetscReal positive_fraction;
    PetscReal positive_l1;
    PetscReal max_positive;
};

PetscErrorCode computeDissipationTBGCS(const bgc::SimConfig& cfg,
                                       Vec                   phi,
                                       DissipationDiag*      diag) {
    PetscFunctionBeginUser;

    PetscMPIInt rank = 0;
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

    VecScatter scatter = nullptr;
    Vec seq = nullptr;
    PetscCall(VecScatterCreateToZero(phi, &scatter, &seq));
    PetscCall(VecScatterBegin(scatter, phi, seq, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(scatter, phi, seq, INSERT_VALUES, SCATTER_FORWARD));

    DissipationDiag result{0.0, 0.0, 0.0};

    if (rank == 0) {
        const PetscScalar* p = nullptr;
        PetscCall(VecGetArrayRead(seq, &p));

        const PetscInt N = cfg.nx;
        const PetscReal h = cfg.h;
        const PetscReal Bv = cfg.bv;
        const PetscReal h2 = h * h;

        std::vector<PetscReal> mu(N);
        for (PetscInt i = 0; i < N; ++i) {
            const PetscReal phi_i = PetscRealPart(p[i]);
            PetscReal phi_xx = 0.0;
            if (i == 0) {
                phi_xx = (2.0 * PetscRealPart(p[1])
                          - 2.0 * phi_i) / h2;
            } else if (i == N - 1) {
                phi_xx = (2.0 * PetscRealPart(p[N - 2])
                          - 2.0 * phi_i) / h2;
            } else {
                phi_xx = (PetscRealPart(p[i + 1])
                          - 2.0 * phi_i
                          + PetscRealPart(p[i - 1])) / h2;
            }
            mu[i] = phi_i - Bv * phi_xx;
        }

        PetscInt nFaces = 0;
        PetscInt nPositive = 0;
        PetscReal sumPositive = 0.0;
        PetscReal maxPositive = 0.0;

        for (PetscInt f = 1; f < N - 1; ++f) {
            const PetscReal mu_x = (mu[f + 1] - mu[f]) / h;
            const PetscReal D = -(mu_x * mu_x);

            ++nFaces;
            if (D > 0.0) {
                ++nPositive;
                sumPositive += D * h;
                if (D > maxPositive) maxPositive = D;
            }
        }

        result.positive_fraction =
            (nFaces > 0) ? static_cast<PetscReal>(nPositive) /
                           static_cast<PetscReal>(nFaces) : 0.0;
        result.positive_l1 = sumPositive;
        result.max_positive = maxPositive;

        PetscCall(VecRestoreArrayRead(seq, &p));
    }

    PetscReal buf[3] = {result.positive_fraction,
                        result.positive_l1,
                        result.max_positive};
    PetscCallMPI(MPI_Bcast(buf, 3, MPIU_REAL, 0, PETSC_COMM_WORLD));
    diag->positive_fraction = buf[0];
    diag->positive_l1       = buf[1];
    diag->max_positive      = buf[2];

    PetscCall(VecScatterDestroy(&scatter));
    PetscCall(VecDestroy(&seq));

    PetscFunctionReturn(PETSC_SUCCESS);
}

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

        out << "step,time,free_energy,phi_min,phi_max,negative_count,above_one_count,pos_fraction,pos_l1,max_pos\n";
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode appendHistoryLine(const std::filesystem::path& filePath,
                                 PetscInt                     step,
                                 PetscReal                    time,
                                 PetscReal                    freeEnergy,
                                 PetscReal                    phiMin,
                                 PetscReal                    phiMax,
                                 PetscInt                     negativeCount,
                                 PetscInt                     aboveOneCount,
                                 const DissipationDiag&       diag) {
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
            << step << ","
            << static_cast<double>(time) << ","
            << static_cast<double>(freeEnergy) << ","
            << static_cast<double>(phiMin) << ","
            << static_cast<double>(phiMax) << ","
            << negativeCount << ","
            << aboveOneCount << ","
            << static_cast<double>(diag.positive_fraction) << ","
            << static_cast<double>(diag.positive_l1) << ","
            << static_cast<double>(diag.max_positive) << "\n";
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode writeMetadata(const bgc::SimConfig&         cfg,
                             const std::filesystem::path&  outDir) {
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
        out << "Bv: " << static_cast<double>(cfg.bv) << "\n";
        out << "nx: " << cfg.nx << "\n";
        out << "h: " << static_cast<double>(cfg.h) << "\n";
        out << "dt: " << static_cast<double>(cfg.dt) << "\n";
        out << "nTimes: " << cfg.nTimes << "\n";
        out << "tf: " << static_cast<double>(cfg.tf) << "\n";
        // out << "BC west: Dirichlet = 0.0, Neumann = 0.0\n";
        // out << "BC east: Dirichlet = 0.0, Neumann = 0.0\n";
        // out << "sigma: " << static_cast<double>(gSigma) << "\n";
        // out << "Initial condition:\n";
        // out << "phi(x,0) = x^2 (1 - x)^2 exp(-(x - xi_c)^2 / (2 sigma^2))\n";
        // out << "with xi_c = " << static_cast<double>(kXiC) << "\n";
        // out << "Initial auxiliary field:\n";
        // out << "mu(x,0) = phi(x,0) - Bv * phi_xx(x,0)\n";
        // out << "Recorded free energy:\n";
        // out << "F_h = 0.5 * [ h * sum(phi_i^2) + Bv * integral_h((dphi/dx)^2) ]\n";
        // out << "The gradient term is approximated by a piecewise-linear\n";
        // out << "reconstruction using phi(0)=phi(1)=0 for this homogeneous case.\n";
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode initialiseState(const bgc::SimConfig& cfg,
                               bgc::SimState&       st) {
    PetscFunctionBeginUser;

    PetscCall(fillVectorFromField(cfg, st.phi, 0.0, initialPhiFn));
    PetscCall(fillVectorFromField(cfg, st.mu, 0.0, initialMuFn));
    PetscCall(VecCopy(st.phi, st.phi_0));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode solveProblem(const bgc::SimConfig&         cfg,
                            bgc::SimState&               st,
                            const std::set<PetscInt>&    saveSteps,
                            const std::filesystem::path& outDir) {
    PetscFunctionBeginUser;

    const std::filesystem::path historyFile = outDir / "history_TBGCS.csv";

    PetscCall(initialiseState(cfg, st));
    PetscCall(initialiseHistoryFile(historyFile));

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

    {
        PetscInt posMin = 0;
        PetscInt posMax = 0;
        PetscReal phiMin = 0.0;
        PetscReal phiMax = 0.0;
        PetscInt negativeCount = 0;
        PetscInt aboveOneCount = 0;
        PetscReal freeEnergy = 0.0;

        PetscCall(VecMin(st.phi, &posMin, &phiMin));
        PetscCall(VecMax(st.phi, &posMax, &phiMax));
        PetscCall(countNonPhysicalEntries(st.phi,
                                          static_cast<PetscReal>(-1.0e-12),
                                          static_cast<PetscReal>(1.0 + 1.0e-12),
                                          &negativeCount,
                                          &aboveOneCount));
        PetscCall(computeDiscreteFreeEnergy(cfg, st.phi, &freeEnergy));

        DissipationDiag diag0{};
        PetscCall(computeDissipationTBGCS(cfg, st.phi, &diag0));

        PetscCall(appendHistoryLine(historyFile,
                                    0,
                                    0.0,
                                    freeEnergy,
                                    phiMin,
                                    phiMax,
                                    negativeCount,
                                    aboveOneCount,
                                    diag0));
    }

    for (PetscInt step = 1; step <= cfg.nTimes; ++step) {
        const PetscReal timeNp1 = static_cast<PetscReal>(step) * cfg.dt;

        const auto rhs = bgc::computeRHSTBGCS(cfg, timeNp1);
        PetscCall(bgc::assembleRHSTBGCS(cfg, st, rhs));

        PetscCall(bgc::computeSourceTerm(cfg, st, timeNp1, zeroSourceFn));
        PetscCall(VecAXPY(st.b1, 1.0, st.b1Source));

        PetscCall(bgc::solveLinearSystem(cfg, st));

        PetscInt posMin = 0;
        PetscInt posMax = 0;
        PetscReal phiMin = 0.0;
        PetscReal phiMax = 0.0;
        PetscInt negativeCount = 0;
        PetscInt aboveOneCount = 0;
        PetscReal freeEnergy = 0.0;

        PetscCall(VecMin(st.phi, &posMin, &phiMin));
        PetscCall(VecMax(st.phi, &posMax, &phiMax));
        PetscCall(countNonPhysicalEntries(st.phi,
                                          static_cast<PetscReal>(-1.0e-12),
                                          static_cast<PetscReal>(1.0 + 1.0e-12),
                                          &negativeCount,
                                          &aboveOneCount));
        PetscCall(computeDiscreteFreeEnergy(cfg, st.phi, &freeEnergy));

        DissipationDiag diag{};
        PetscCall(computeDissipationTBGCS(cfg, st.phi, &diag));

        PetscCall(appendHistoryLine(historyFile,
                                    step,
                                    timeNp1,
                                    freeEnergy,
                                    phiMin,
                                    phiMax,
                                    negativeCount,
                                    aboveOneCount,
                                    diag));

        if (negativeCount > 0 || aboveOneCount > 0) {
            PetscPrintf(PETSC_COMM_WORLD,
                        "[WARNING] step=%" PetscInt_FMT
                        "  time=%.16e"
                        "  phi_min=%.16e"
                        "  phi_max=%.16e"
                        "  negative_count=%" PetscInt_FMT
                        "  above_one_count=%" PetscInt_FMT "\n",
                        step,
                        timeNp1,
                        phiMin,
                        phiMax,
                        negativeCount,
                        aboveOneCount);
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

int main(int argc, char** argv) {
    PetscCallAbort(PETSC_COMM_WORLD,
                   PetscInitialize(&argc,
                                   &argv,
                                   nullptr,
                                   "Case3_TBGCS -- coarse-mesh energy test"));

    PetscInt exitCode = 0;

    try {
        const std::filesystem::path dataDir = findDataDirectory();
        const InputData input = readInputData(dataDir);
        const bgc::SimConfig cfg = makeConfig(input);

        gSigma = input.sigma;

        const std::filesystem::path outDir =
            dataDir.parent_path() / "Saida";
        std::filesystem::create_directories(outDir);

        PetscPrintf(PETSC_COMM_WORLD,
                    "\n######################################################################\n"
                    "  Case3_TBGCS -- coarse-mesh energy test\n"
                    "  Source term: 0\n"
                    "  BCs: phi = 0 and dphi/dx = 0 at both ends\n"
                    "  Initial condition: phi(x,0) = x^2 (1 - x)^2 exp(-(x - xi_c)^2 / (2 sigma^2))\n  xi_c = 5.0e-1\n"
                    "  sigma = %.16e\n"
                    "  Output directory: %s\n"
                    "######################################################################\n",
                    gSigma,
                    outDir.string().c_str());

        PetscPrintf(PETSC_COMM_WORLD,
                    "nx=%" PetscInt_FMT
                    "  h=%.16e"
                    "  dt=%.16e"
                    "  nTimes=%" PetscInt_FMT
                    "  Bv=%.16e\n",
                    cfg.nx,
                    cfg.h,
                    cfg.dt,
                    cfg.nTimes,
                    cfg.bv);

        bgc::SimState st;
        PetscCallAbort(PETSC_COMM_WORLD, bgc::createStateTBGC(cfg, st));

        const auto sc = bgc::computeCoefficientsTBGCS(cfg);
        PetscCallAbort(PETSC_COMM_WORLD, bgc::assembleMatrixTBGCS(cfg, st, sc));
        PetscCallAbort(PETSC_COMM_WORLD, bgc::configureSolver(cfg, st));

        const std::set<PetscInt> saveSteps = buildSaveSteps(cfg, input.saveTimes);

        PetscCallAbort(PETSC_COMM_WORLD, writeMetadata(cfg, outDir));
        PetscCallAbort(PETSC_COMM_WORLD, solveProblem(cfg, st, saveSteps, outDir));

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