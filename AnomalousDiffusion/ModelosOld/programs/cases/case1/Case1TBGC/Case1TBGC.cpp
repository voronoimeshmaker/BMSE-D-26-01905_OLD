#include <bgclib/Core/BoundaryCondition.hpp>
#include <bgclib/Models/TBGC/AssemblyTBGC.hpp>
#include <bgclib/Models/TBGC/CoeffTBGC.hpp>
#include <bgclib/SimConfig.hpp>
#include <bgclib/SimState.hpp>
#include <bgclib/Solver.hpp>
#include <bgclib/PostProcess.hpp>

#include <petsc.h>

#include <algorithm>
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
    PetscReal tf = 1.0e-3;
    PetscReal cdt = 1.024e-1;
    PetscInt nx = 256;
    PetscReal bv = 1.0e-2;
    std::vector<PetscReal> saveTimes = {
        0.0,
        2.5e-4,
        5.0e-4,
        7.5e-4,
        1.0e-3
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
    if (simData.contains("save_times")) {
        input.saveTimes = parsePetscRealList(simData.at("save_times"));
    }

    return input;
}

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

    cfg.bcWest[0] = bgc::BoundaryCondition::neumann(0.2);
    cfg.bcWest[1] = bgc::BoundaryCondition::dirichlet(1.0);

    cfg.bcEast[0] = bgc::BoundaryCondition::neumann(-0.1);
    cfg.bcEast[1] = bgc::BoundaryCondition::dirichlet(1.2);

    return cfg;
}

PetscReal initialPhiFn(const bgc::SimConfig&, PetscReal x, PetscReal) {
    const PetscReal x2 = x * x;
    const PetscReal x3 = x2 * x;
    const PetscReal bubble = 0.05 * x2 * (1.0 - x) * (1.0 - x);

    return 1.0 + 0.2 * x + 0.3 * x2 - 0.3 * x3 + bubble;
}

PetscReal initialPhiXXFn(const bgc::SimConfig&, PetscReal x, PetscReal) {
    const PetscReal x2 = x * x;
    return 0.7 - 2.4 * x + 0.6 * x2;
}

PetscReal initialMuFn(const bgc::SimConfig& cfg, PetscReal x, PetscReal t) {
    return initialPhiFn(cfg, x, t) - cfg.bv * initialPhiXXFn(cfg, x, t);
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

PetscErrorCode writeVectorCSV(const bgc::SimConfig&          cfg,
                              Vec                            v,
                              const std::filesystem::path&   filePath,
                              PetscReal                      time,
                              const std::string&             columnName) {
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

        for (PetscInt i = 0; i < cfg.nx; ++i) {
            out << i << ","
                << static_cast<double>(cfg.xCenter(i)) << ","
                << static_cast<double>(PetscRealPart(values[i])) << "\n";
        }

        PetscCall(VecRestoreArrayRead(seq, &values));
    }

    PetscCall(VecScatterDestroy(&scatter));
    PetscCall(VecDestroy(&seq));

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
        out << "Model: TBGC\n";
        out << "Case: Comparative case 1\n";
        out << "Source term: 0\n";
        out << "Bv: " << static_cast<double>(cfg.bv) << "\n";
        out << "nx: " << cfg.nx << "\n";
        out << "h: " << static_cast<double>(cfg.h) << "\n";
        out << "dt: " << static_cast<double>(cfg.dt) << "\n";
        out << "nTimes: " << cfg.nTimes << "\n";
        out << "tf: " << static_cast<double>(cfg.tf) << "\n";
        out << "BC west: Dirichlet = 1.0, Neumann = 0.2\n";
        out << "BC east: Dirichlet = 1.2, Neumann = -0.1\n";
        out << "Initial condition:\n";
        out << "phi(x,0) = 1 + 0.2*x + 0.3*x^2 - 0.3*x^3 + 0.05*x^2*(1-x)^2\n";
        out << "Initial auxiliary field:\n";
        out << "mu(x,0) = phi(x,0) - Bv*(0.7 - 2.4*x + 0.6*x^2)\n";
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

PetscErrorCode solveProblem(const bgc::SimConfig&        cfg,
                            bgc::SimState&              st,
                            const std::set<PetscInt>&   saveSteps,
                            const std::filesystem::path& outDir) {
    PetscFunctionBeginUser;

    PetscCall(initialiseState(cfg, st));

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

    for (PetscInt step = 1; step <= cfg.nTimes; ++step) {
        const PetscReal timeNp1 = static_cast<PetscReal>(step) * cfg.dt;

        const auto rhs = bgc::computeRHSTBGC(cfg, timeNp1);
        PetscCall(bgc::assembleRHSTBGC(cfg, st, rhs));

        PetscCall(bgc::computeSourceTerm(cfg, st, timeNp1, zeroSourceFn));
        PetscCall(VecAXPY(st.b1, 1.0, st.b1Source));

        PetscCall(bgc::solveLinearSystem(cfg, st));

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
                                   "Case1TBGC -- comparative case with non-zero Dirichlet and Neumann boundary conditions"));

    PetscInt exitCode = 0;

    try {
        const std::filesystem::path dataDir = findDataDirectory();
        const InputData input = readInputData(dataDir);
        const bgc::SimConfig cfg = makeConfig(input);

        const std::filesystem::path outDir =
            std::filesystem::current_path() / "Saidas";
        std::filesystem::create_directories(outDir);

        PetscPrintf(PETSC_COMM_WORLD,
                    "\n######################################################################\n"
                    "  Case1TBGC -- comparative case 1 for TBGC\n"
                    "  Source term: 0\n"
                    "  Non-zero Dirichlet and Neumann boundary conditions\n"
                    "  Output directory: %s\n"
                    "######################################################################\n",
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

        const auto sc = bgc::computeCoefficientsTBGC(cfg);
        PetscCallAbort(PETSC_COMM_WORLD, bgc::assembleMatrixTBGC(cfg, st, sc));
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
