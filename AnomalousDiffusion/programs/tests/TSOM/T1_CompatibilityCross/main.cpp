#include <bgclib/BGCLib.hpp>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using Constants = bgc::models::tsom::Constants;

struct Mesh {
    PetscInt nx;
    PetscReal h;
    PetscReal dt;
    PetscInt steps;
};

struct ExactPoint {
    PetscReal u;
    PetscReal v;
    PetscReal uxx;
    PetscReal vxx;
};

struct Case {
    std::string name;
    bgc::models::tsom::VBoundaryCondition vBoundary;
    PetscReal sc;
};

struct Result {
    std::string name;
    PetscInt nx;
    PetscReal h;
    PetscReal dt;
    PetscInt steps;
    PetscReal l2Psi;
    PetscReal l2U;
    PetscReal l2V;
    PetscReal ltePsi;
    PetscReal lteU;
    PetscReal lteV;
    KSPConvergedReason reason;
    PetscInt iterations;
};

using InputData = bgc::models::bgc::MMSInputData;

PetscReal lambda = 1.0;
PetscReal psiAmplitude = 2.0;
PetscReal scPartition = 0.3;

std::filesystem::path caseDirectory() {
    return bgc::caseDirectoryFromSource(__FILE__);
}

InputData defaultInput() {
    return {
        .tf = 0.1,
        .cdt = 0.1024,
        .nxList = {4, 8, 16, 32, 64},
        .bvList = {1.0},
        .outputDir = "Saida",
        .verbose = false,
        .debug = false,
    };
}

InputData readInput(const std::filesystem::path& dataDir) {
    InputData input = bgc::models::bgc::readMMSInputData(dataDir, defaultInput());
    const std::unordered_map<std::string, std::string> data =
        bgc::readKeyValueFile(dataDir / "simulation.dat");
    if (data.contains("lambda")) {
        lambda = bgc::parseReal(data.at("lambda"));
    }
    if (data.contains("mms_lambda")) {
        lambda = bgc::parseReal(data.at("mms_lambda"));
    }
    if (data.contains("psi_amplitude")) {
        psiAmplitude = bgc::parseReal(data.at("psi_amplitude"));
    }
    if (data.contains("sc")) {
        scPartition = bgc::parseReal(data.at("sc"));
    }
    if (data.contains("f")) {
        const PetscReal fAlias = bgc::parseReal(data.at("f"));
        if (data.contains("sc") && PetscAbsReal(fAlias - scPartition) > 1.0e-14) {
            throw std::runtime_error("Parameters f and sc are aliases and must have the same value.");
        }
        scPartition = fAlias;
    }
    if (data.contains("partition_f")) {
        const PetscReal fAlias = bgc::parseReal(data.at("partition_f"));
        if (data.contains("sc") && PetscAbsReal(fAlias - scPartition) > 1.0e-14) {
            throw std::runtime_error("Parameters partition_f and sc are aliases and must have the same value.");
        }
        scPartition = fAlias;
    }
    return input;
}

Constants readConstants(const std::filesystem::path& dataDir) {
    Constants constants {
        .alpha = 0.5,
        .rho = 0.5,
        .theta = 0.5,
        .lambdaC = 5.0,
        .lambdaR = 3.0,
        .vBoundary = {.type = bgc::models::tsom::VBoundaryCondition::ZeroValue, .sc = 0.0},
    };
    const std::unordered_map<std::string, std::string> data =
        bgc::readKeyValueFile(dataDir / "simulation.dat");
    if (data.contains("alpha")) constants.alpha = bgc::parseReal(data.at("alpha"));
    if (data.contains("rho")) constants.rho = bgc::parseReal(data.at("rho"));
    if (data.contains("theta")) constants.theta = bgc::parseReal(data.at("theta"));
    if (data.contains("lambda_c")) constants.lambdaC = bgc::parseReal(data.at("lambda_c"));
    if (data.contains("lambda_r")) constants.lambdaR = bgc::parseReal(data.at("lambda_r"));
    if (data.contains("v_boundary")) {
        constants.vBoundary.type = bgc::models::tsom::parseVBoundaryCondition(data.at("v_boundary"));
    }
    constants.vBoundary.sc = scPartition;
    if (!bgc::ModelTraits<bgc::models::tsom::Tag>::constantsAreValid(constants)) {
        throw std::runtime_error("Invalid TSOM constants in simulation.dat.");
    }
    return constants;
}

Mesh makeMesh(const PetscInt nx, const InputData& input) {
    const PetscReal h = 1.0 / static_cast<PetscReal>(nx);
    const PetscReal requestedDt = input.cdt * h * h;
    const PetscInt steps = static_cast<PetscInt>(PetscCeilReal(input.tf / requestedDt));
    return {.nx = nx, .h = h, .dt = input.tf / static_cast<PetscReal>(steps), .steps = steps};
}

bgc::Grid1D makeGrid(const Mesh& mesh) {
    return {.nx = mesh.nx, .length = 1.0, .x0 = 0.0};
}

bgc::TimeConfig makeTime(const Mesh& mesh, const InputData& input) {
    return {.dt = mesh.dt, .finalTime = input.tf, .initialTime = 0.0};
}

bgc::BoundarySet makeBoundaries() {
    return {
        .west = {.conditions = {bgc::BoundaryCondition::neumann(0.0),
                                bgc::BoundaryCondition::neumann(0.0)}},
        .east = {.conditions = {bgc::BoundaryCondition::neumann(0.0),
                                bgc::BoundaryCondition::neumann(0.0)}},
    };
}

Constants makeConstants(Constants base, const Case& c) {
    base.vBoundary.type = c.vBoundary;
    base.vBoundary.sc = c.sc;
    return base;
}

std::string safeCaseName(std::string name) {
    for (char& ch : name) {
        if (ch == ' ' || ch == '/' || ch == '\\') {
            ch = '_';
        }
    }
    return name;
}

PetscReal uFractionForBoundary(const Constants& constants) {
    switch (constants.vBoundary.type) {
    case bgc::models::tsom::VBoundaryCondition::ZeroValue:
        return 1.0;
    case bgc::models::tsom::VBoundaryCondition::ConstantGradient:
        return constants.vBoundary.sc;
    case bgc::models::tsom::VBoundaryCondition::ScaledPsi:
        return constants.vBoundary.sc;
    }
    return scPartition;
}

ExactPoint exactAt(const PetscReal, const Constants& constants) {
    const PetscReal uFraction = uFractionForBoundary(constants);
    return {
        .u = uFraction * psiAmplitude,
        .v = (1.0 - uFraction) * psiAmplitude,
        .uxx = 0.0,
        .vxx = 0.0,
    };
}

std::vector<PetscReal> exactState(const Mesh& mesh, const PetscReal t, const Constants& constants) {
    const PetscReal factor = PetscExpReal(-lambda * t);
    std::vector<PetscReal> y(static_cast<std::size_t>(2 * mesh.nx), 0.0);
    for (PetscInt i = 0; i < mesh.nx; ++i) {
        const ExactPoint e = exactAt((static_cast<PetscReal>(i) + 0.5) * mesh.h, constants);
        y[static_cast<std::size_t>(i)] = factor * e.u;
        y[static_cast<std::size_t>(mesh.nx + i)] = factor * e.v;
    }
    return y;
}

bgc::DiscreteOperator makeOperator(const Mesh& mesh, const InputData& input, const Constants& constants) {
    const bgc::Grid1D grid = makeGrid(mesh);
    return bgc::models::tsom::buildOperator(
        grid,
        bgc::models::tsom::computeCoefficients(grid, makeTime(mesh, input), constants, makeBoundaries()));
}

std::vector<PetscReal> sourceProfiles(const Mesh& mesh, const Constants& constants) {
    const auto d = bgc::models::tsom::computeDiffusionCoefficients(constants);
    std::vector<PetscReal> s(static_cast<std::size_t>(2 * mesh.nx), 0.0);
    for (PetscInt i = 0; i < mesh.nx; ++i) {
        const ExactPoint e = exactAt((static_cast<PetscReal>(i) + 0.5) * mesh.h, constants);
        s[static_cast<std::size_t>(i)] =
            (-lambda + constants.lambdaC) * e.u - constants.lambdaR * e.v -
            d.d11 * e.uxx - d.d12 * e.vxx;
        s[static_cast<std::size_t>(mesh.nx + i)] =
            -constants.lambdaC * e.u + (-lambda + constants.lambdaR) * e.v -
            d.d21 * e.uxx - d.d22 * e.vxx;
    }
    return s;
}

std::vector<PetscReal> makeRHS(const Mesh& mesh,
                               const Constants& constants,
                               const std::vector<PetscReal>& sources,
                               const std::vector<PetscReal>& previous,
                               const PetscReal t) {
    const bgc::Grid1D grid = makeGrid(mesh);
    const auto boundaryRHS = bgc::models::tsom::buildBoundaryRHS(
        grid,
        bgc::models::tsom::computeBoundaryRHS(grid, constants, makeBoundaries(), t));
    const PetscReal factor = PetscExpReal(-lambda * t);
    const PetscReal hdt = mesh.h / mesh.dt;
    std::vector<PetscReal> rhs(static_cast<std::size_t>(2 * mesh.nx), 0.0);
    for (PetscInt i = 0; i < mesh.nx; ++i) {
        const auto ui = static_cast<std::size_t>(i);
        const auto vi = static_cast<std::size_t>(mesh.nx + i);
        rhs[ui] = hdt * previous[ui] + mesh.h * factor * sources[ui];
        rhs[vi] = hdt * previous[vi] + mesh.h * factor * sources[vi];
    }
    for (const bgc::RHSEntry& entry : boundaryRHS.entries) {
        rhs[static_cast<std::size_t>(entry.row)] += entry.value;
    }
    return rhs;
}

void writeFields(const std::filesystem::path& outputDir,
                 const Case& c,
                 const Constants& constants,
                 const Mesh& mesh,
                 const std::vector<PetscReal>& numerical,
                 const std::vector<PetscReal>& exact);

Result runCase(const Case& c,
               const PetscInt nx,
               const InputData& input,
               const Constants& baseConstants,
               const std::filesystem::path& outputDir) {
    const Mesh mesh = makeMesh(nx, input);
    const Constants constants = makeConstants(baseConstants, c);
    const bgc::DiscreteOperator op = makeOperator(mesh, input, constants);
    const std::vector<PetscReal> sources = sourceProfiles(mesh, constants);
    std::vector<PetscReal> y = exactState(mesh, 0.0, constants);
    bgc::models::bgc::MMSLinearSolver solver(op, 0.0);
    KSPConvergedReason reason = KSP_CONVERGED_ITERATING;
    PetscInt iterations = 0;
    for (PetscInt step = 1; step <= mesh.steps; ++step) {
        const PetscReal t = static_cast<PetscReal>(step) * mesh.dt;
        const std::vector<PetscReal> rhs = makeRHS(mesh, constants, sources, y, t);
        PetscCallAbort(PETSC_COMM_WORLD, solver.solve(rhs, y, reason, iterations));
    }

    const std::vector<PetscReal> exact = exactState(mesh, input.tf, constants);
    writeFields(outputDir, c, constants, mesh, y, exact);
    const std::vector<PetscReal> exactPrevious = exactState(mesh, input.tf - mesh.dt, constants);
    const std::vector<PetscReal> rhs = makeRHS(mesh, constants, sources, exactPrevious, input.tf);
    std::vector<PetscReal> tau;
    PetscCallAbort(PETSC_COMM_WORLD, bgc::computeLocalTruncationError(op, exact, rhs, tau));

    std::vector<PetscReal> u(mesh.nx), v(mesh.nx), psi(mesh.nx);
    std::vector<PetscReal> ue(mesh.nx), ve(mesh.nx), psie(mesh.nx);
    std::vector<PetscReal> tu(mesh.nx), tv(mesh.nx), tpsi(mesh.nx);
    for (PetscInt i = 0; i < mesh.nx; ++i) {
        const auto ui = static_cast<std::size_t>(i);
        const auto vi = static_cast<std::size_t>(mesh.nx + i);
        u[ui] = y[ui]; v[ui] = y[vi]; psi[ui] = y[ui] + y[vi];
        ue[ui] = exact[ui]; ve[ui] = exact[vi]; psie[ui] = exact[ui] + exact[vi];
        tu[ui] = tau[ui]; tv[ui] = tau[vi]; tpsi[ui] = tau[ui] + tau[vi];
    }
    return {
        .name = c.name, .nx = nx, .h = mesh.h, .dt = mesh.dt, .steps = mesh.steps,
        .l2Psi = bgc::computeErrorNorms(psi, psie, mesh.h).l2,
        .l2U = bgc::computeErrorNorms(u, ue, mesh.h).l2,
        .l2V = bgc::computeErrorNorms(v, ve, mesh.h).l2,
        .ltePsi = bgc::computeVectorNorms(tpsi, mesh.h).l2,
        .lteU = bgc::computeVectorNorms(tu, mesh.h).l2,
        .lteV = bgc::computeVectorNorms(tv, mesh.h).l2,
        .reason = reason, .iterations = iterations,
    };
}

void writeFields(const std::filesystem::path& outputDir,
                 const Case& c,
                 const Constants& constants,
                 const Mesh& mesh,
                 const std::vector<PetscReal>& numerical,
                 const std::vector<PetscReal>& exact) {
    std::filesystem::create_directories(outputDir);
    const std::filesystem::path path =
        outputDir / ("t1_fields_" + safeCaseName(c.name) + "_N" +
                     std::to_string(static_cast<int>(mesh.nx)) + ".dat");
    std::ofstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Could not open field output file: " + path.string());
    }

    f << std::scientific << std::setprecision(16);
    f << "# test = T1_CompatibilityCross\n";
    f << "# case = " << c.name << "\n";
    f << "# alpha = " << constants.alpha << "\n";
    f << "# rho = " << constants.rho << "\n";
    f << "# theta = " << constants.theta << "\n";
    f << "# lambda_c = " << constants.lambdaC << "\n";
    f << "# lambda_r = " << constants.lambdaR << "\n";
    f << "# v_boundary = " << bgc::models::tsom::toString(constants.vBoundary.type) << "\n";
    f << "# sc = " << constants.vBoundary.sc << "\n";
    f << "# mms_lambda = " << lambda << "\n";
    f << "# psi_amplitude = " << psiAmplitude << "\n";
    f << "# nx = " << mesh.nx << "\n";
    f << "# h = " << mesh.h << "\n";
    f << "# dt = " << mesh.dt << "\n";
    f << "# steps = " << mesh.steps << "\n";
    f << "# P x u_num u_exact u_err v_num v_exact v_err psi_num psi_exact psi_err\n";
    for (PetscInt i = 0; i < mesh.nx; ++i) {
        const auto ui = static_cast<std::size_t>(i);
        const auto vi = static_cast<std::size_t>(mesh.nx + i);
        const PetscReal x = (static_cast<PetscReal>(i) + 0.5) * mesh.h;
        const PetscReal uNum = numerical[ui];
        const PetscReal vNum = numerical[vi];
        const PetscReal psiNum = uNum + vNum;
        const PetscReal uExact = exact[ui];
        const PetscReal vExact = exact[vi];
        const PetscReal psiExact = uExact + vExact;
        f << (i + 1) << ' ' << x
          << ' ' << uNum << ' ' << uExact << ' ' << PetscAbsReal(uNum - uExact)
          << ' ' << vNum << ' ' << vExact << ' ' << PetscAbsReal(vNum - vExact)
          << ' ' << psiNum << ' ' << psiExact << ' ' << PetscAbsReal(psiNum - psiExact)
          << '\n';
    }
}

PetscReal order(const PetscReal a, const PetscReal b) {
    return PetscLogReal(a / b) / PetscLogReal(2.0);
}

void writeResults(const std::filesystem::path& outputDir,
                  const Constants& constants,
                  const std::vector<Result>& results) {
    std::filesystem::create_directories(outputDir);
    std::ofstream f(outputDir / "t1_compatibility_cross.csv");
    f << std::scientific << std::setprecision(16);
    f << "# test = T1_CompatibilityCross\n";
    f << "# alpha = " << constants.alpha << "\n";
    f << "# rho = " << constants.rho << "\n";
    f << "# theta = " << constants.theta << "\n";
    f << "# lambda_c = " << constants.lambdaC << "\n";
    f << "# lambda_r = " << constants.lambdaR << "\n";
    f << "# v_boundary = " << bgc::models::tsom::toString(constants.vBoundary.type) << "\n";
    f << "# sc = " << constants.vBoundary.sc << "\n";
    f << "# mms_lambda = " << lambda << "\n";
    f << "# psi_amplitude = " << psiAmplitude << "\n";
    f << "case,nx,h,dt,steps,L2_Psi,p_Psi,L2_U,p_U,L2_V,p_V,LTE_L2_Psi,LTE_L2_U,LTE_L2_V,ksp_reason,ksp_iterations,status\n";
    const Result* prev = nullptr;
    for (const Result& r : results) {
        const bool same = prev != nullptr && prev->name == r.name;
        f << r.name << ',' << r.nx << ',' << r.h << ',' << r.dt << ',' << r.steps
          << ',' << r.l2Psi << ',';
        if (same) f << order(prev->l2Psi, r.l2Psi);
        f << ',' << r.l2U << ',';
        if (same) f << order(prev->l2U, r.l2U);
        f << ',' << r.l2V << ',';
        if (same) f << order(prev->l2V, r.l2V);
        f << ',' << r.ltePsi << ',' << r.lteU << ',' << r.lteV
          << ',' << static_cast<int>(r.reason) << ',' << r.iterations
          << ',' << bgc::convergenceStatus(r.reason) << '\n';
        prev = &r;
    }
}

} // namespace

int main(int argc, char** argv) {
    PetscCallAbort(PETSC_COMM_WORLD, PetscInitialize(&argc, &argv, nullptr, nullptr));
    int code = 0;
    try {
        const std::filesystem::path caseDir = caseDirectory();
        const InputData input = readInput(caseDir / "Dados");
        const Constants baseConstants = readConstants(caseDir / "Dados");
        const std::vector<Case> cases {
            {.name = "T1_" + std::string(bgc::models::tsom::toString(baseConstants.vBoundary.type)),
             .vBoundary = baseConstants.vBoundary.type,
             .sc = baseConstants.vBoundary.sc},
        };
        std::vector<Result> results;
        for (const Case& c : cases) {
            for (const PetscInt nx : input.nxList) {
                results.push_back(runCase(c, nx, input, baseConstants, caseDir / input.outputDir));
            }
        }
        writeResults(caseDir / input.outputDir, baseConstants, results);
        std::cout << "Wrote " << (caseDir / input.outputDir / "t1_compatibility_cross.csv") << "\n";
    } catch (const std::exception& ex) {
        PetscPrintf(PETSC_COMM_WORLD, "T1 failed: %s\n", ex.what());
        code = 1;
    }
    PetscCallAbort(PETSC_COMM_WORLD, PetscFinalize());
    return code;
}
