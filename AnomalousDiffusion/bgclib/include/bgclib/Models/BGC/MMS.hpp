#pragma once

// Reusable BGC Method of Manufactured Solutions driver.
//
// A BGC MMS program should define only the manufactured field, source term,
// boundary conditions, and user-facing description. This layer owns the common
// campaign mechanics: input parsing, mesh/time setup, BGC operator/RHS assembly,
// sequential PETSc solve, truncation error, norms, and output files.

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <petsc.h>

#include <bgclib/Analysis/ExactSolution.hpp>
#include <bgclib/Analysis/ErrorNorms.hpp>
#include <bgclib/Analysis/ManufacturedSource.hpp>
#include <bgclib/Analysis/MMS.hpp>
#include <bgclib/Analysis/TruncationError.hpp>
#include <bgclib/Analysis/RunSummary.hpp>
#include <bgclib/Core/BoundarySet.hpp>
#include <bgclib/Core/Grid1D.hpp>
#include <bgclib/IO/ConfigReader.hpp>
#include <bgclib/IO/OutputPaths.hpp>
#include <bgclib/IO/ResultWriter.hpp>
#include <bgclib/Models/BGC/Coeff.hpp>
#include <bgclib/Models/BGC/Operator.hpp>

namespace bgc::models::bgc {

struct MMSInputData {
    PetscReal tf {1.0e-3};
    PetscReal cdt {1.024e-1};
    std::vector<PetscInt> nxList {8};
    std::vector<PetscReal> bvList {1.0e-2};
    std::filesystem::path outputDir {"Saida"};
    bool verbose {true};
    bool debug {false};
    PetscInt debugPrintMaxNx {16};
};

struct MMSMeshTimeData {
    PetscInt nx {0};
    PetscReal h {0.0};
    PetscReal dt {0.0};
    PetscInt nTimes {0};
};

struct MMSNormRecord {
    PetscReal bv {0.0};
    PetscInt nx {0};
    PetscReal h {0.0};
    PetscReal dt {0.0};
    PetscInt nTimes {0};
    PetscReal l1Phi {0.0};
    PetscReal l2Phi {0.0};
    PetscReal linfPhi {0.0};
    PetscReal l1Tau {0.0};
    PetscReal l2Tau {0.0};
    PetscReal linfTau {0.0};
    KSPConvergedReason reason {KSP_CONVERGED_ITERATING};
    PetscInt iterations {0};
};

struct MMSCaseFiles {
    std::string setupCsv;
    std::string convergenceCsv;
    std::string statusTxt;
    std::string fieldsPrefix;
    std::string truncationPrefix;
};

struct MMSCaseSpec {
    std::string caseName;
    std::string description;
    std::string fieldFormula;
    std::string sourceFormula;
    MMSCaseFiles files;
    bool writeStatusFile {false};
    bool writeSetupSamples {false};
};

namespace detail {

inline std::string trim(const std::string& text) {
    const auto first = std::find_if_not(text.begin(), text.end(), [](const unsigned char c) {
        return std::isspace(c) != 0;
    });
    if (first == text.end()) {
        return {};
    }

    const auto last = std::find_if_not(text.rbegin(), text.rend(), [](const unsigned char c) {
        return std::isspace(c) != 0;
    }).base();
    return {first, last};
}

inline std::string toLower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](const unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return text;
}

inline std::vector<std::string> splitByComma(const std::string& text) {
    std::vector<std::string> parts;
    std::stringstream stream {text};
    std::string part;
    while (std::getline(stream, part, ',')) {
        const std::string cleaned = trim(part);
        if (!cleaned.empty()) {
            parts.push_back(cleaned);
        }
    }
    return parts;
}

inline PetscReal parseReal(const std::string& text) {
    const std::string cleaned = trim(text);
    const char* begin = cleaned.c_str();
    char* end = nullptr;
    const PetscReal value = static_cast<PetscReal>(std::strtod(begin, &end));
    if (end == begin) {
        throw std::runtime_error("Invalid PetscReal value: " + cleaned);
    }
    return value;
}

inline PetscInt parseInt(const std::string& text) {
    const std::string cleaned = trim(text);
    const char* begin = cleaned.c_str();
    char* end = nullptr;
    const long value = std::strtol(begin, &end, 10);
    if (end == begin) {
        throw std::runtime_error("Invalid PetscInt value: " + cleaned);
    }
    return static_cast<PetscInt>(value);
}

inline bool parseBool(const std::string& text) {
    const std::string cleaned = toLower(trim(text));
    if (cleaned == "true" || cleaned == "yes" || cleaned == "on" || cleaned == "1") {
        return true;
    }
    if (cleaned == "false" || cleaned == "no" || cleaned == "off" || cleaned == "0") {
        return false;
    }
    throw std::runtime_error("Invalid bool value: " + cleaned);
}

inline std::vector<PetscReal> parseRealList(const std::string& text) {
    std::vector<PetscReal> values;
    for (const std::string& part : splitByComma(text)) {
        values.push_back(parseReal(part));
    }
    if (values.empty()) {
        throw std::runtime_error("Empty PetscReal list.");
    }
    return values;
}

inline std::vector<PetscInt> parseIntList(const std::string& text) {
    std::vector<PetscInt> values;
    for (const std::string& part : splitByComma(text)) {
        values.push_back(parseInt(part));
    }
    if (values.empty()) {
        throw std::runtime_error("Empty PetscInt list.");
    }
    return values;
}

inline std::unordered_map<std::string, std::string>
readKeyValueFile(const std::filesystem::path& path) {
    std::ifstream file {path};
    if (!file.is_open()) {
        throw std::runtime_error("Could not open input file: " + path.string());
    }

    std::unordered_map<std::string, std::string> data;
    std::string line;
    PetscInt lineNumber = 0;
    while (std::getline(file, line)) {
        ++lineNumber;
        const std::size_t comment = line.find('#');
        if (comment != std::string::npos) {
            line = line.substr(0, comment);
        }
        line = trim(line);
        if (line.empty()) {
            continue;
        }

        const std::size_t equals = line.find('=');
        if (equals == std::string::npos) {
            throw std::runtime_error("Invalid line " + std::to_string(lineNumber) +
                                     " in " + path.string());
        }
        data[toLower(trim(line.substr(0, equals)))] = trim(line.substr(equals + 1));
    }
    return data;
}

inline std::string sanitizeRealForPath(const PetscReal value) {
    std::ostringstream stream;
    stream << std::scientific << std::setprecision(3) << value;
    std::string text = stream.str();
    for (char& c : text) {
        if (c == '+') {
            c = 'p';
        } else if (c == '-') {
            c = 'm';
        } else if (c == '.') {
            c = 'p';
        }
    }
    return text;
}

} // namespace detail

inline MMSInputData readMMSInputData(const std::filesystem::path& dataDir,
                                     MMSInputData input = {}) {
    const auto data = ::bgc::readKeyValueFile(dataDir / "simulation.dat");

    if (data.contains("tf")) {
        input.tf = ::bgc::parseReal(data.at("tf"));
    }
    if (data.contains("cdt")) {
        input.cdt = ::bgc::parseReal(data.at("cdt"));
    }
    if (data.contains("nx_list")) {
        input.nxList = ::bgc::parseIntList(data.at("nx_list"));
    }
    if (data.contains("bv_list")) {
        input.bvList = ::bgc::parseRealList(data.at("bv_list"));
    }
    if (data.contains("output_dir")) {
        input.outputDir = ::bgc::trim(data.at("output_dir"));
    }
    if (data.contains("verbose")) {
        input.verbose = ::bgc::parseBool(data.at("verbose"));
    }
    if (data.contains("debug")) {
        input.debug = ::bgc::parseBool(data.at("debug"));
    }
    if (data.contains("print_system")) {
        input.debug = ::bgc::parseBool(data.at("print_system"));
    }
    if (data.contains("debug_print_max_nx")) {
        input.debugPrintMaxNx = ::bgc::parseInt(data.at("debug_print_max_nx"));
    }
    return input;
}

inline std::filesystem::path findMMSDataDirectory() {
    const std::filesystem::path dataDir = std::filesystem::current_path() / "Dados";
    if (std::filesystem::is_directory(dataDir)) {
        return std::filesystem::canonical(dataDir);
    }
    throw std::runtime_error("Could not locate Dados directory at: " + dataDir.string());
}

inline std::filesystem::path resolveMMSOutputDirectory(
    const std::filesystem::path& caseDir,
    const std::filesystem::path& configuredDir) {
    return ::bgc::resolveOutputDirectory(caseDir, configuredDir);
}

inline MMSMeshTimeData makeMMSMeshTimeData(const MMSInputData& input, const PetscInt nx) {
    MMSMeshTimeData data;
    data.nx = nx;
    data.h = 1.0 / static_cast<PetscReal>(nx);
    data.dt = input.cdt * data.h * data.h;
    data.nTimes = static_cast<PetscInt>(input.tf / data.dt + 0.5);
    if (data.nTimes < 1) {
        data.nTimes = 1;
    }
    data.dt = input.tf / static_cast<PetscReal>(data.nTimes);
    return data;
}

inline Grid1D makeMMSGrid(const MMSMeshTimeData& mesh) {
    return {
        .nx = mesh.nx,
        .length = 1.0,
        .x0 = 0.0,
    };
}

inline std::filesystem::path mmsBvOutputDirectory(const std::filesystem::path& outputDir,
                                                  const PetscReal bv) {
    return ::bgc::bvOutputDirectory(outputDir, bv);
}

inline bool isWorldRankZero() noexcept {
    PetscMPIInt rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    return rank == 0;
}

inline PetscMPIInt worldSize() noexcept {
    PetscMPIInt size = 1;
    MPI_Comm_size(PETSC_COMM_WORLD, &size);
    return size;
}

inline PetscErrorCode printCpuUsageSummary() {
    PetscFunctionBeginUser;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "  MPI processes/CPUs used: %d\n",
                          static_cast<int>(worldSize())));
    PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename ExactEvaluator>
std::vector<PetscReal> buildMMSExactValues(const MMSMeshTimeData& mesh,
                                           const PetscReal time,
                                           const Constants& constants,
                                           const ExactEvaluator& exactEvaluator) {
    const ExactSolution<Tag, ExactEvaluator> exact {constants, exactEvaluator};
    std::vector<PetscReal> values(static_cast<std::size_t>(mesh.nx));
    for (PetscInt i = 0; i < mesh.nx; ++i) {
        const PetscReal x = (static_cast<PetscReal>(i) + 0.5) * mesh.h;
        values[static_cast<std::size_t>(i)] = exact(x, time);
    }
    return values;
}

template <typename SourceEvaluator, typename BoundaryFactory>
std::vector<PetscReal> buildMMSTotalSourceValues(const MMSMeshTimeData& mesh,
                                                 const PetscReal time,
                                                 const Constants& constants,
                                                 const SourceEvaluator& sourceEvaluator,
                                                 const BoundaryFactory& boundaryFactory) {
    const ManufacturedSource<Tag, SourceEvaluator> source {constants, sourceEvaluator};
    const Grid1D grid = makeMMSGrid(mesh);
    const auto boundaryRHS = buildBoundaryRHS(
        grid,
        computeBoundaryRHS(grid, constants, boundaryFactory(), time));
    std::vector<PetscReal> values(static_cast<std::size_t>(mesh.nx), 0.0);

    for (PetscInt i = 0; i < mesh.nx; ++i) {
        const PetscReal x = (static_cast<PetscReal>(i) + 0.5) * mesh.h;
        values[static_cast<std::size_t>(i)] = mesh.h * source(x, time);
    }
    for (const RHSEntry& entry : boundaryRHS.entries) {
        values[static_cast<std::size_t>(entry.row)] += entry.value;
    }
    return values;
}

inline PetscErrorCode solveMMSLinearSystem(const DiscreteOperator& op,
                                           const std::vector<PetscReal>& rhsValues,
                                           const PetscReal matrixShift,
                                           std::vector<PetscReal>& solutionValues,
                                           KSPConvergedReason& reason,
                                           PetscInt& iterations) {
    PetscFunctionBeginUser;

    class LinearSolveContext {
    public:
        LinearSolveContext(const DiscreteOperator& op, const PetscReal shift) {
            PetscCallAbort(PETSC_COMM_WORLD, createSequentialMatrix(op, matrix_));
            if (shift != 0.0) {
                PetscCallAbort(PETSC_COMM_WORLD, MatShift(matrix_, shift));
            }
            PetscCallAbort(PETSC_COMM_WORLD, MatCreateVecs(matrix_, &solution_, &rhs_));
            PetscCallAbort(PETSC_COMM_WORLD, KSPCreate(PETSC_COMM_WORLD, &ksp_));
            PetscCallAbort(PETSC_COMM_WORLD, KSPSetOperators(ksp_, matrix_, matrix_));

            PetscMPIInt commSize = 1;
            MPI_Comm_size(PETSC_COMM_WORLD, &commSize);
            PC pc = nullptr;
            PetscCallAbort(PETSC_COMM_WORLD, KSPGetPC(ksp_, &pc));
            if (commSize == 1) {
                PetscCallAbort(PETSC_COMM_WORLD, KSPSetType(ksp_, KSPPREONLY));
                PetscCallAbort(PETSC_COMM_WORLD, PCSetType(pc, PCLU));
            } else {
                PetscCallAbort(PETSC_COMM_WORLD, KSPSetType(ksp_, KSPGMRES));
                PetscCallAbort(PETSC_COMM_WORLD, PCSetType(pc, PCBJACOBI));
                PetscCallAbort(PETSC_COMM_WORLD,
                               KSPSetTolerances(ksp_, 1.0e-12, PETSC_DEFAULT, PETSC_DEFAULT, 2000));
            }
            PetscCallAbort(PETSC_COMM_WORLD, KSPSetFromOptions(ksp_));
        }

        ~LinearSolveContext() {
            PetscCallAbort(PETSC_COMM_WORLD, KSPDestroy(&ksp_));
            PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&solution_));
            PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&rhs_));
            PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&matrix_));
        }

        PetscErrorCode solve(const std::vector<PetscReal>& rhsValues,
                             std::vector<PetscReal>& solutionValues,
                             KSPConvergedReason& reason,
                             PetscInt& iterations) {
            PetscFunctionBeginUser;
            PetscCall(copyValuesToSequentialVector(rhsValues, rhs_));
            PetscCall(KSPSolve(ksp_, rhs_, solution_));
            PetscCall(KSPGetConvergedReason(ksp_, &reason));
            PetscCall(KSPGetIterationNumber(ksp_, &iterations));
            PetscCall(copySequentialVector(solution_, solutionValues));
            PetscFunctionReturn(PETSC_SUCCESS);
        }

    private:
        Mat matrix_ {nullptr};
        Vec rhs_ {nullptr};
        Vec solution_ {nullptr};
        KSP ksp_ {nullptr};
    };

    LinearSolveContext context(op, matrixShift);
    PetscCall(context.solve(rhsValues, solutionValues, reason, iterations));

    PetscFunctionReturn(PETSC_SUCCESS);
}

class MMSLinearSolver {
public:
    MMSLinearSolver(const DiscreteOperator& op, const PetscReal matrixShift) {
        PetscCallAbort(PETSC_COMM_WORLD, createSequentialMatrix(op, matrix_));
        if (matrixShift != 0.0) {
            PetscCallAbort(PETSC_COMM_WORLD, MatShift(matrix_, matrixShift));
        }
        PetscCallAbort(PETSC_COMM_WORLD, MatCreateVecs(matrix_, &solution_, &rhs_));
        PetscCallAbort(PETSC_COMM_WORLD, KSPCreate(PETSC_COMM_WORLD, &ksp_));
        PetscCallAbort(PETSC_COMM_WORLD, KSPSetOperators(ksp_, matrix_, matrix_));

        PetscMPIInt commSize = 1;
        MPI_Comm_size(PETSC_COMM_WORLD, &commSize);
        PC pc = nullptr;
        PetscCallAbort(PETSC_COMM_WORLD, KSPGetPC(ksp_, &pc));
        if (commSize == 1) {
            PetscCallAbort(PETSC_COMM_WORLD, KSPSetType(ksp_, KSPPREONLY));
            PetscCallAbort(PETSC_COMM_WORLD, PCSetType(pc, PCLU));
        } else {
            PetscCallAbort(PETSC_COMM_WORLD, KSPSetType(ksp_, KSPGMRES));
            PetscCallAbort(PETSC_COMM_WORLD, PCSetType(pc, PCBJACOBI));
            PetscCallAbort(PETSC_COMM_WORLD,
                           KSPSetTolerances(ksp_, 1.0e-12, PETSC_DEFAULT, PETSC_DEFAULT, 2000));
        }
        PetscCallAbort(PETSC_COMM_WORLD, KSPSetFromOptions(ksp_));
    }

    MMSLinearSolver(const MMSLinearSolver&) = delete;
    MMSLinearSolver& operator=(const MMSLinearSolver&) = delete;

    ~MMSLinearSolver() {
        PetscCallAbort(PETSC_COMM_WORLD, KSPDestroy(&ksp_));
        PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&solution_));
        PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&rhs_));
        PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&matrix_));
    }

    PetscErrorCode solve(const std::vector<PetscReal>& rhsValues,
                         std::vector<PetscReal>& solutionValues,
                         KSPConvergedReason& reason,
                         PetscInt& iterations) {
        PetscFunctionBeginUser;
        PetscCall(copyValuesToSequentialVector(rhsValues, rhs_));
        PetscCall(KSPSolve(ksp_, rhs_, solution_));
        PetscCall(KSPGetConvergedReason(ksp_, &reason));
        PetscCall(KSPGetIterationNumber(ksp_, &iterations));
        PetscCall(copySequentialVector(solution_, solutionValues));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

private:
    Mat matrix_ {nullptr};
    Vec rhs_ {nullptr};
    Vec solution_ {nullptr};
    KSP ksp_ {nullptr};
};

template <typename ExactEvaluator, typename SourceEvaluator, typename BoundaryFactory>
PetscErrorCode solveTransientMMSField(const MMSMeshTimeData& mesh,
                                      const Constants& constants,
                                      const ExactEvaluator& exactEvaluator,
                                      const SourceEvaluator& sourceEvaluator,
                                      const BoundaryFactory& boundaryFactory,
                                      std::vector<PetscReal>& phiNumerical,
                                      KSPConvergedReason& reason,
                                      PetscInt& iterations) {
    PetscFunctionBeginUser;

    const Grid1D grid = makeMMSGrid(mesh);
    const auto coefficients = computeCoefficients(grid, constants);
    const auto op = buildOperator(grid, coefficients);
    const PetscReal hdt = mesh.h / mesh.dt;

    phiNumerical = buildMMSExactValues(mesh, 0.0, constants, exactEvaluator);
    reason = KSP_CONVERGED_ITERATING;
    iterations = 0;

    MMSLinearSolver solver(op, hdt);
    for (PetscInt step = 1; step <= mesh.nTimes; ++step) {
        const PetscReal time = static_cast<PetscReal>(step) * mesh.dt;
        std::vector<PetscReal> rhsValues =
            buildMMSTotalSourceValues(mesh, time, constants, sourceEvaluator, boundaryFactory);

        for (PetscInt i = 0; i < mesh.nx; ++i) {
            rhsValues[static_cast<std::size_t>(i)] +=
                hdt * phiNumerical[static_cast<std::size_t>(i)];
        }

        PetscCall(solver.solve(rhsValues, phiNumerical, reason, iterations));
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

inline MMSNormRecord computeMMSNorms(const MMSMeshTimeData& mesh,
                                     const PetscReal bv,
                                     const std::vector<PetscReal>& numerical,
                                     const std::vector<PetscReal>& exact,
                                     const std::vector<PetscReal>& tau,
                                     const KSPConvergedReason reason,
                                     const PetscInt iterations) {
    MMSNormRecord record {
        .bv = bv,
        .nx = mesh.nx,
        .h = mesh.h,
        .dt = mesh.dt,
        .nTimes = mesh.nTimes,
        .reason = reason,
        .iterations = iterations,
    };

    const ErrorNorms phiNorms = computeErrorNorms(numerical, exact, mesh.h);
    const ErrorNorms tauNorms = computeVectorNorms(tau, mesh.h);

    record.l1Phi = phiNorms.l1;
    record.l2Phi = phiNorms.l2;
    record.linfPhi = phiNorms.linf;
    record.l1Tau = tauNorms.l1;
    record.l2Tau = tauNorms.l2;
    record.linfTau = tauNorms.linf;
    return record;
}

inline void writeMMSFields(const std::filesystem::path& path,
                           const MMSMeshTimeData& mesh,
                           const std::vector<PetscReal>& numerical,
                           const std::vector<PetscReal>& exact) {
    std::ofstream file = ::bgc::openOutputFile(path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open output file: " + path.string());
    }

    file << std::scientific << std::setprecision(16);
    file << "# P x phi_num phi_exact phi_err\n";
    for (PetscInt i = 0; i < mesh.nx; ++i) {
        const auto index = static_cast<std::size_t>(i);
        const PetscReal x = (static_cast<PetscReal>(i) + 0.5) * mesh.h;
        file << i + 1 << ' ' << x << ' ' << numerical[index] << ' ' << exact[index]
             << ' ' << numerical[index] - exact[index] << '\n';
    }
}

inline void writeMMSTruncation(const std::filesystem::path& path,
                               const MMSMeshTimeData& mesh,
                               const std::vector<PetscReal>& tau) {
    std::ofstream file = ::bgc::openOutputFile(path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open output file: " + path.string());
    }

    file << std::scientific << std::setprecision(16);
    file << "# P x_center tau_phi\n";
    for (PetscInt i = 0; i < mesh.nx; ++i) {
        const PetscReal x = (static_cast<PetscReal>(i) + 0.5) * mesh.h;
        file << i + 1 << ' ' << x << ' ' << tau[static_cast<std::size_t>(i)] << '\n';
    }
}

inline void writeMMSConvergence(const std::filesystem::path& path,
                                const std::vector<MMSNormRecord>& records) {
    std::ofstream file = ::bgc::openOutputFile(path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open output file: " + path.string());
    }

    file << std::scientific << std::setprecision(16);
    file << "bv,nx,h,dt,nTimes,L1_phi,L2_phi,Linf_phi,LTE_L1_phi,LTE_L2_phi,LTE_Linf_phi,ksp_reason,ksp_iterations,status\n";
    for (const MMSNormRecord& r : records) {
        file << r.bv << ',' << r.nx << ',' << r.h << ',' << r.dt << ',' << r.nTimes
             << ',' << r.l1Phi << ',' << r.l2Phi << ',' << r.linfPhi << ','
             << r.l1Tau << ',' << r.l2Tau << ',' << r.linfTau << ','
             << static_cast<int>(r.reason) << ',' << r.iterations << ','
             << ::bgc::convergenceStatus(r.reason) << '\n';
    }
}

template <typename ExactEvaluator, typename SourceEvaluator>
void writeMMSSetup(const std::filesystem::path& path,
                   const std::vector<MMSNormRecord>& records,
                   const ExactEvaluator& exactEvaluator,
                   const SourceEvaluator& sourceEvaluator,
                   const PetscReal finalTime,
                   const bool writeSamples) {
    std::ofstream file = ::bgc::openOutputFile(path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open output file: " + path.string());
    }

    file << std::scientific << std::setprecision(16);
    file << "bv,nx,h,dt,nTimes";
    if (writeSamples) {
        file << ",phi_0p5_tf,source_0p5_tf";
    }
    file << '\n';

    for (const MMSNormRecord& r : records) {
        file << r.bv << ',' << r.nx << ',' << r.h << ',' << r.dt << ',' << r.nTimes;
        if (writeSamples) {
            const Constants constants {.bv = r.bv};
            file << ',' << exactEvaluator(constants, 0.5, finalTime)
                 << ',' << sourceEvaluator(constants, 0.5, finalTime);
        }
        file << '\n';
    }
}

inline void writeMMSStatusFile(const std::filesystem::path& path,
                               const MMSCaseSpec& spec,
                               const MMSCampaign& campaign) {
    std::ofstream file = ::bgc::openOutputFile(path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open output file: " + path.string());
    }

    file
        << spec.caseName << " full numerical run status\n"
        << "======================================\n\n"
        << "Objective: " << campaign.objective() << ".\n\n"
        << "Analytical fields, PETSc numerical fields, analytical-vs-numerical\n"
        << "errors, truncation-error fields, and norm summaries are written per Bv/N.\n"
        << "The numerical solve follows the BGC implicit transient path:\n"
        << "(A + h/dt I) * phi_np1 = h*source(t_np1) + boundary_rhs(t_np1) + (h/dt)*phi_n.\n\n"
        << "Output files:\n"
        << "- " << spec.files.convergenceCsv << '\n'
        << "- " << spec.files.fieldsPrefix << "N*.dat\n"
        << "- " << spec.files.truncationPrefix << "N*.dat\n";
}

template <typename ExactEvaluator, typename SourceEvaluator, typename BoundaryFactory>
class MMSRunner {
public:
    MMSRunner(MMSCaseSpec spec,
              MMSInputData input,
              std::filesystem::path outputDirectory,
              ExactEvaluator exactEvaluator,
              SourceEvaluator sourceEvaluator,
              BoundaryFactory boundaryFactory)
        : spec_ {std::move(spec)},
          input_ {std::move(input)},
          outputDirectory_ {std::move(outputDirectory)},
          exactEvaluator_ {std::move(exactEvaluator)},
          sourceEvaluator_ {std::move(sourceEvaluator)},
          boundaryFactory_ {std::move(boundaryFactory)} {}

    [[nodiscard]] MMSCampaign campaign() const {
        return {
            spec_.caseName,
            "BGC",
            outputDirectory_,
            {
                .setupCsv = spec_.files.setupCsv,
                .convergenceCsv = spec_.files.convergenceCsv,
                .statusTxt = spec_.files.statusTxt,
                .summaryPrefix = {},
                .fieldsPrefix = spec_.files.fieldsPrefix,
                .truncationPrefix = spec_.files.truncationPrefix,
            },
        };
    }

    void run() const {
        if (isWorldRankZero()) {
            ::bgc::ensureDirectory(outputDirectory_);
        }
        std::vector<MMSNormRecord> setupRecords;

        for (const PetscReal bv : input_.bvList) {
            const Constants constants {.bv = bv};
            const std::filesystem::path outDir = mmsBvOutputDirectory(outputDirectory_, bv);
            std::vector<MMSNormRecord> norms;
            if (isWorldRankZero()) {
                ::bgc::ensureDirectory(outDir);
            }

            for (const PetscInt nx : input_.nxList) {
                const MMSMeshTimeData mesh = makeMMSMeshTimeData(input_, nx);
                const std::string nxText = std::to_string(static_cast<int>(nx));
                const Grid1D grid = makeMMSGrid(mesh);
                const auto coefficients = computeCoefficients(grid, constants);
                const auto op = buildOperator(grid, coefficients);
                const std::vector<PetscReal> exact =
                    buildMMSExactValues(mesh, input_.tf, constants, exactEvaluator_);
                const std::vector<PetscReal> source =
                    buildMMSTotalSourceValues(mesh,
                                              input_.tf,
                                              constants,
                                              sourceEvaluator_,
                                              boundaryFactory_);
                std::vector<PetscReal> phiNumerical;
                std::vector<PetscReal> tau;
                KSPConvergedReason reason = KSP_CONVERGED_ITERATING;
                PetscInt iterations = 0;

                PetscCallAbort(PETSC_COMM_WORLD,
                               solveTransientMMSField(mesh,
                                                      constants,
                                                      exactEvaluator_,
                                                      sourceEvaluator_,
                                                      boundaryFactory_,
                                                      phiNumerical,
                                                      reason,
                                                      iterations));
                PetscCallAbort(PETSC_COMM_WORLD,
                               computeLocalTruncationError(op, exact, source, tau));

                if (isWorldRankZero()) {
                    if (input_.verbose) {
                        PetscPrintf(PETSC_COMM_SELF,
                                    "  %s Bv=%.6e nx=%d: steps=%d status=%s\n",
                                    spec_.caseName.c_str(),
                                    static_cast<double>(bv),
                                    static_cast<int>(nx),
                                    static_cast<int>(mesh.nTimes),
                                    ::bgc::convergenceStatus(reason));
                    }
                    if (input_.debug && worldSize() == 1 && nx <= input_.debugPrintMaxNx) {
                        writePetscDenseMatrixView(
                            outDir / (spec_.files.fieldsPrefix + nxText + "_matrix.dat"),
                            op);
                    }
                    writeMMSFields(outDir / (spec_.files.fieldsPrefix + nxText + ".dat"),
                                   mesh,
                                   phiNumerical,
                                   exact);
                    writeMMSTruncation(outDir / (spec_.files.truncationPrefix + nxText + ".dat"),
                                       mesh,
                                       tau);
                }

                norms.push_back(computeMMSNorms(mesh,
                                                bv,
                                                phiNumerical,
                                                exact,
                                                tau,
                                                reason,
                                                iterations));
                setupRecords.push_back(norms.back());
            }

            if (isWorldRankZero()) {
                writeMMSConvergence(outDir / spec_.files.convergenceCsv, norms);
            }
        }

        if (!isWorldRankZero()) {
            return;
        }

        writeMMSSetup(outputDirectory_ / spec_.files.setupCsv,
                      setupRecords,
                      exactEvaluator_,
                      sourceEvaluator_,
                      input_.tf,
                      spec_.writeSetupSamples);

        if (spec_.writeStatusFile) {
            writeMMSStatusFile(outputDirectory_ / spec_.files.statusTxt, spec_, campaign());
        }
    }

    PetscErrorCode printSummary(const std::filesystem::path& dataDir) const {
        PetscFunctionBeginUser;
        PetscCall(printCpuUsageSummary());
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                              "\n%s -- %s\n"
                              "  phi(x,t) = %s\n"
                              "  source   = %s\n"
                              "  input : %s\n"
                              "  output: %s\n",
                              spec_.caseName.c_str(),
                              spec_.description.c_str(),
                              spec_.fieldFormula.c_str(),
                              spec_.sourceFormula.c_str(),
                              dataDir.string().c_str(),
                              outputDirectory_.string().c_str()));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

private:
    MMSCaseSpec spec_;
    MMSInputData input_;
    std::filesystem::path outputDirectory_;
    ExactEvaluator exactEvaluator_;
    SourceEvaluator sourceEvaluator_;
    BoundaryFactory boundaryFactory_;
};

template <typename ExactEvaluator, typename SourceEvaluator, typename BoundaryFactory>
MMSRunner(MMSCaseSpec,
          MMSInputData,
          std::filesystem::path,
          ExactEvaluator,
          SourceEvaluator,
          BoundaryFactory)
    -> MMSRunner<ExactEvaluator, SourceEvaluator, BoundaryFactory>;

} // namespace bgc::models::bgc
