#include <bgclib/BGCLib.hpp>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using Constants = bgc::models::tsom::Constants;
using InputData = bgc::models::bgc::MMSInputData;

struct Mesh { PetscInt nx; PetscReal h; PetscReal dt; PetscInt steps; };
struct Exact { PetscReal u; PetscReal v; PetscReal uxx; PetscReal vxx; };
struct Case {
    std::string name;
    bgc::models::tsom::VBoundaryCondition vBoundary;
    PetscReal sc;
    PetscInt profile;
};
struct Result {
    std::string name; PetscInt nx; PetscReal h; PetscReal dt; PetscInt steps;
    PetscReal l2Psi; PetscReal l2U; PetscReal l2V;
    PetscReal ltePsi; PetscReal lteU; PetscReal lteV;
    KSPConvergedReason reason; PetscInt iterations;
};

PetscReal lambda = 1.0;
PetscReal scPartition = 0.5;

std::filesystem::path caseDirectory() { return bgc::caseDirectoryFromSource(__FILE__); }

InputData defaultInput() {
    return {.tf = 1.0, .cdt = 0.1024, .nxList = {8, 16, 32, 64, 128},
            .bvList = {1.0}, .outputDir = "T5/Caso1", .verbose = false, .debug = false};
}

InputData readInput(const std::filesystem::path& dataDir) {
    InputData input = bgc::models::bgc::readMMSInputData(dataDir, defaultInput());
    const auto data = bgc::readKeyValueFile(dataDir / "simulation.dat");
    if (data.contains("lambda")) lambda = bgc::parseReal(data.at("lambda"));
    if (data.contains("mms_lambda")) lambda = bgc::parseReal(data.at("mms_lambda"));
    if (data.contains("sc")) scPartition = bgc::parseReal(data.at("sc"));
    if (data.contains("f")) {
        const PetscReal fAlias = bgc::parseReal(data.at("f"));
        if (data.contains("sc") && PetscAbsReal(fAlias - scPartition) > 1.0e-14) {
            throw std::runtime_error("Parameters f and sc are aliases and must have the same value.");
        }
        scPartition = fAlias;
    }
    return input;
}

Constants readBaseConstants(const std::filesystem::path& dataDir) {
    Constants c {
        .alpha = 0.5, .rho = 0.5, .theta = 0.5, .lambdaC = 5.0, .lambdaR = 3.0,
        .vBoundary = {.type = bgc::models::tsom::VBoundaryCondition::ZeroValue, .sc = scPartition},
    };
    const std::unordered_map<std::string, std::string> data =
        bgc::readKeyValueFile(dataDir / "simulation.dat");
    if (data.contains("alpha")) c.alpha = bgc::parseReal(data.at("alpha"));
    if (data.contains("rho")) c.rho = bgc::parseReal(data.at("rho"));
    if (data.contains("theta")) c.theta = bgc::parseReal(data.at("theta"));
    if (data.contains("lambda_c")) c.lambdaC = bgc::parseReal(data.at("lambda_c"));
    if (data.contains("lambda_r")) c.lambdaR = bgc::parseReal(data.at("lambda_r"));
    c.vBoundary.sc = scPartition;
    if (!bgc::ModelTraits<bgc::models::tsom::Tag>::constantsAreValid(c)) {
        throw std::runtime_error("Invalid TSOM constants in simulation.dat.");
    }
    return c;
}

Constants constantsForCase(Constants base, const Case& cs) {
    base.vBoundary.type = cs.vBoundary;
    base.vBoundary.sc = cs.sc;
    return base;
}

std::string safeCaseName(std::string name) {
    for (char& ch : name) {
        if (ch == ' ' || ch == '/' || ch == '\\') ch = '_';
    }
    return name;
}

Mesh makeMesh(const PetscInt nx, const InputData& input) {
    const PetscReal h = 1.0 / static_cast<PetscReal>(nx);
    const PetscInt steps = static_cast<PetscInt>(PetscCeilReal(input.tf / (input.cdt * h * h)));
    return {.nx = nx, .h = h, .dt = input.tf / static_cast<PetscReal>(steps), .steps = steps};
}

bgc::BoundarySet boundaries() {
    return {
        .west = {.conditions = {bgc::BoundaryCondition::neumann(0.0),
                                bgc::BoundaryCondition::neumann(0.0)}},
        .east = {.conditions = {bgc::BoundaryCondition::neumann(0.0),
                                bgc::BoundaryCondition::neumann(0.0)}},
    };
}

bgc::Grid1D grid(const Mesh& m) { return {.nx = m.nx, .length = 1.0, .x0 = 0.0}; }
bgc::TimeConfig timeConfig(const Mesh& m, const InputData& input) { return {.dt = m.dt, .finalTime = input.tf, .initialTime = 0.0}; }

Exact exactAt(const PetscReal x, const Case& cs) {
    const PetscReal p = std::numbers::pi_v<PetscReal>;
    const PetscReal c = PetscCosReal(p * x);
    const PetscReal s = PetscSinReal(p * x);
    const PetscReal psi = 1.0 + c;
    const PetscReal psixx = -p * p * c;

    if (cs.profile == 1) {
        const PetscReal v = s;
        const PetscReal vxx = -p * p * s;
        return {.u = psi - v, .v = v, .uxx = psixx - vxx, .vxx = vxx};
    }
    if (cs.profile == 2) {
        return {.u = 1.0, .v = c, .uxx = 0.0, .vxx = -p * p * c};
    }
    return {.u = cs.sc * psi, .v = (1.0 - cs.sc) * psi, .uxx = cs.sc * psixx, .vxx = (1.0 - cs.sc) * psixx};
}

std::vector<PetscReal> exactState(const Mesh& m, const PetscReal t, const Case& cs) {
    const PetscReal e = PetscExpReal(-lambda * t);
    std::vector<PetscReal> y(static_cast<std::size_t>(2 * m.nx), 0.0);
    for (PetscInt i = 0; i < m.nx; ++i) {
        const Exact q = exactAt((static_cast<PetscReal>(i) + 0.5) * m.h, cs);
        y[static_cast<std::size_t>(i)] = e * q.u;
        y[static_cast<std::size_t>(m.nx + i)] = e * q.v;
    }
    return y;
}

bgc::DiscreteOperator op(const Mesh& m, const InputData& input, const Constants& c) {
    const bgc::Grid1D g = grid(m);
    return bgc::models::tsom::buildOperator(
        g, bgc::models::tsom::computeCoefficients(g, timeConfig(m, input), c, boundaries()));
}

std::vector<PetscReal> source(const Mesh& m, const Constants& c, const Case& cs) {
    const auto d = bgc::models::tsom::computeDiffusionCoefficients(c);
    std::vector<PetscReal> s(static_cast<std::size_t>(2 * m.nx), 0.0);
    for (PetscInt i = 0; i < m.nx; ++i) {
        const Exact q = exactAt((static_cast<PetscReal>(i) + 0.5) * m.h, cs);
        s[static_cast<std::size_t>(i)] =
            (-lambda + c.lambdaC) * q.u - c.lambdaR * q.v - d.d11 * q.uxx - d.d12 * q.vxx;
        s[static_cast<std::size_t>(m.nx + i)] =
            -c.lambdaC * q.u + (-lambda + c.lambdaR) * q.v - d.d21 * q.uxx - d.d22 * q.vxx;
    }
    return s;
}

std::vector<PetscReal> rhs(const Mesh& m, const Constants& c,
                           const std::vector<PetscReal>& src,
                           const std::vector<PetscReal>& prev, const PetscReal t) {
    const bgc::Grid1D g = grid(m);
    const auto brhs = bgc::models::tsom::buildBoundaryRHS(
        g, bgc::models::tsom::computeBoundaryRHS(g, c, boundaries(), t));
    const PetscReal e = PetscExpReal(-lambda * t);
    const PetscReal hdt = m.h / m.dt;
    std::vector<PetscReal> b(static_cast<std::size_t>(2 * m.nx), 0.0);
    for (PetscInt i = 0; i < m.nx; ++i) {
        const auto u = static_cast<std::size_t>(i);
        const auto v = static_cast<std::size_t>(m.nx + i);
        b[u] = hdt * prev[u] + m.h * e * src[u];
        b[v] = hdt * prev[v] + m.h * e * src[v];
    }
    for (const bgc::RHSEntry& entry : brhs.entries) b[static_cast<std::size_t>(entry.row)] += entry.value;
    return b;
}

void writeFields(const std::filesystem::path& outputDir, const Case& cs, const Constants& c, const Mesh& m,
                 const std::vector<PetscReal>& y, const std::vector<PetscReal>& ye) {
    std::filesystem::create_directories(outputDir);
    std::ofstream f(outputDir / ("t5_fields_" + safeCaseName(cs.name) + "_N" +
                                 std::to_string(static_cast<int>(m.nx)) + ".dat"));
    f << std::scientific << std::setprecision(16);
    f << "# test = T5_FixedPsi\n";
    f << "# case = " << cs.name << "\n";
    f << "# alpha = " << c.alpha << "\n";
    f << "# rho = " << c.rho << "\n";
    f << "# theta = " << c.theta << "\n";
    f << "# lambda_c = " << c.lambdaC << "\n";
    f << "# lambda_r = " << c.lambdaR << "\n";
    f << "# v_boundary = " << bgc::models::tsom::toString(c.vBoundary.type) << "\n";
    f << "# sc = " << c.vBoundary.sc << "\n";
    f << "# mms_lambda = " << lambda << "\n";
    f << "# nx = " << m.nx << "\n";
    f << "# h = " << m.h << "\n";
    f << "# dt = " << m.dt << "\n";
    f << "# steps = " << m.steps << "\n";
    f << "# P x u_num u_exact u_err v_num v_exact v_err psi_num psi_exact psi_err\n";
    for (PetscInt i = 0; i < m.nx; ++i) {
        const auto ui = static_cast<std::size_t>(i);
        const auto vi = static_cast<std::size_t>(m.nx + i);
        const PetscReal x = (static_cast<PetscReal>(i) + 0.5) * m.h;
        const PetscReal p = y[ui] + y[vi];
        const PetscReal pe = ye[ui] + ye[vi];
        f << (i + 1) << ' ' << x
          << ' ' << y[ui] << ' ' << ye[ui] << ' ' << PetscAbsReal(y[ui] - ye[ui])
          << ' ' << y[vi] << ' ' << ye[vi] << ' ' << PetscAbsReal(y[vi] - ye[vi])
          << ' ' << p << ' ' << pe << ' ' << PetscAbsReal(p - pe) << '\n';
    }
}

Result run(const Case& cs, const PetscInt nx, const InputData& input, const Constants& base,
           const std::filesystem::path& outputDir) {
    const Mesh m = makeMesh(nx, input);
    const Constants c = constantsForCase(base, cs);
    const auto a = op(m, input, c);
    const auto src = source(m, c, cs);
    std::vector<PetscReal> y = exactState(m, 0.0, cs);
    bgc::models::bgc::MMSLinearSolver solver(a, 0.0);
    KSPConvergedReason reason = KSP_CONVERGED_ITERATING;
    PetscInt iterations = 0;
    for (PetscInt step = 1; step <= m.steps; ++step) {
        PetscCallAbort(PETSC_COMM_WORLD,
                       solver.solve(rhs(m, c, src, y, static_cast<PetscReal>(step) * m.dt),
                                    y, reason, iterations));
    }
    const auto ye = exactState(m, input.tf, cs);
    writeFields(outputDir, cs, c, m, y, ye);
    const auto yp = exactState(m, input.tf - m.dt, cs);
    std::vector<PetscReal> tau;
    PetscCallAbort(PETSC_COMM_WORLD, bgc::computeLocalTruncationError(a, ye, rhs(m, c, src, yp, input.tf), tau));

    std::vector<PetscReal> u(m.nx), v(m.nx), p(m.nx), ue(m.nx), ve(m.nx), pe(m.nx), tu(m.nx), tv(m.nx), tp(m.nx);
    for (PetscInt i = 0; i < m.nx; ++i) {
        const auto ui = static_cast<std::size_t>(i);
        const auto vi = static_cast<std::size_t>(m.nx + i);
        u[ui] = y[ui]; v[ui] = y[vi]; p[ui] = y[ui] + y[vi];
        ue[ui] = ye[ui]; ve[ui] = ye[vi]; pe[ui] = ye[ui] + ye[vi];
        tu[ui] = tau[ui]; tv[ui] = tau[vi]; tp[ui] = tau[ui] + tau[vi];
    }
    return {.name = cs.name, .nx = nx, .h = m.h, .dt = m.dt, .steps = m.steps,
            .l2Psi = bgc::computeErrorNorms(p, pe, m.h).l2,
            .l2U = bgc::computeErrorNorms(u, ue, m.h).l2,
            .l2V = bgc::computeErrorNorms(v, ve, m.h).l2,
            .ltePsi = bgc::computeVectorNorms(tp, m.h).l2,
            .lteU = bgc::computeVectorNorms(tu, m.h).l2,
            .lteV = bgc::computeVectorNorms(tv, m.h).l2,
            .reason = reason, .iterations = iterations};
}

PetscReal rate(const PetscReal a, const PetscReal b) { return PetscLogReal(a / b) / PetscLogReal(2.0); }

void write(const std::filesystem::path& outputDir, const Constants& base, const std::vector<Result>& rs) {
    std::filesystem::create_directories(outputDir);
    std::ofstream f(outputDir / "t5_fixed_psi.csv");
    f << std::scientific << std::setprecision(16);
    f << "# test = T5_FixedPsi\n";
    f << "# alpha = " << base.alpha << "\n";
    f << "# rho = " << base.rho << "\n";
    f << "# theta = " << base.theta << "\n";
    f << "# lambda_c = " << base.lambdaC << "\n";
    f << "# lambda_r = " << base.lambdaR << "\n";
    f << "# sc = " << scPartition << "\n";
    f << "# mms_lambda = " << lambda << "\n";
    f << "case,nx,h,dt,steps,L2_Psi,p_Psi,L2_U,p_U,L2_V,p_V,LTE_L2_Psi,LTE_L2_U,LTE_L2_V,ksp_reason,ksp_iterations,status\n";
    const Result* prev = nullptr;
    for (const Result& r : rs) {
        const bool same = prev != nullptr && prev->name == r.name;
        f << r.name << ',' << r.nx << ',' << r.h << ',' << r.dt << ',' << r.steps << ',' << r.l2Psi << ',';
        if (same) f << rate(prev->l2Psi, r.l2Psi);
        f << ',' << r.l2U << ',';
        if (same) f << rate(prev->l2U, r.l2U);
        f << ',' << r.l2V << ',';
        if (same) f << rate(prev->l2V, r.l2V);
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
        const Constants base = readBaseConstants(caseDir / "Dados");
        const std::vector<Case> cases {
            {.name = "T5_zero_value", .vBoundary = bgc::models::tsom::VBoundaryCondition::ZeroValue, .sc = scPartition, .profile = 1},
            {.name = "T5_constant_gradient", .vBoundary = bgc::models::tsom::VBoundaryCondition::ConstantGradient, .sc = scPartition, .profile = 2},
            {.name = "T5_scaled_psi", .vBoundary = bgc::models::tsom::VBoundaryCondition::ScaledPsi, .sc = scPartition, .profile = 3},
        };
        std::vector<Result> rs;
        for (const Case& c : cases) {
            for (const PetscInt nx : input.nxList) rs.push_back(run(c, nx, input, base, caseDir / input.outputDir));
        }
        write(caseDir / input.outputDir, base, rs);
        std::cout << "Wrote " << (caseDir / input.outputDir / "t5_fixed_psi.csv") << "\n";
    } catch (const std::exception& ex) {
        PetscPrintf(PETSC_COMM_WORLD, "T5 failed: %s\n", ex.what());
        code = 1;
    }
    PetscCallAbort(PETSC_COMM_WORLD, PetscFinalize());
    return code;
}
