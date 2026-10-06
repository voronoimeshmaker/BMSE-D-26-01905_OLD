#pragma once

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numbers>
#include <string>
#include <string_view>
#include <vector>

#include <petsc.h>

#include <bgclib/BGCLib.hpp>

namespace paper_tsompsiv {

struct Mesh {
    PetscInt n {0};
    PetscReal h {0.0};
    PetscReal dt {0.0};
    PetscReal tf {0.0};
    PetscInt steps {0};
};

struct Sample {
    PetscReal time {0.0};
    std::vector<PetscReal> psi;
    std::vector<PetscReal> v;
    PetscReal centerRatio {0.0};
    PetscReal mass {0.0};
    PetscReal energy {0.0};
    PetscReal variance {0.0};
};

struct TsomRun {
    Mesh mesh;
    std::vector<PetscReal> psi;
    std::vector<PetscReal> v;
    std::vector<Sample> samples;
    PetscReal minPsi {0.0};
    PetscReal mass0 {0.0};
    PetscReal massFinal {0.0};
    PetscReal energy0 {0.0};
    PetscReal energyFinal {0.0};
};

struct ScalarRun {
    Mesh mesh;
    std::vector<PetscReal> phi;
    std::vector<Sample> samples;
    PetscReal minPhi {0.0};
    PetscReal mass0 {0.0};
    PetscReal massFinal {0.0};
    PetscReal varianceFinal {0.0};
};

inline std::filesystem::path caseDirFromSource(const std::string_view source) {
    return bgc::caseDirectoryFromSource(source);
}

inline std::filesystem::path outputDir(const std::string_view source) {
    auto dir = caseDirFromSource(source) / "Saida";
    bgc::ensureDirectory(dir);
    return dir;
}

inline Mesh makeMesh(const PetscInt n, const PetscReal tf, const PetscReal cdt = 0.1024) {
    const PetscReal h = 1.0 / static_cast<PetscReal>(n);
    const PetscReal requested = cdt * h * h;
    const PetscInt steps = static_cast<PetscInt>(std::ceil(tf / requested));
    const PetscReal dt = tf / static_cast<PetscReal>(steps);
    return {.n = n, .h = h, .dt = dt, .tf = tf, .steps = steps};
}

inline PetscReal xCell(const Mesh& mesh, const PetscInt i) {
    return (static_cast<PetscReal>(i) + 0.5) * mesh.h;
}

inline PetscReal sin2Profile(const PetscReal x) {
    const PetscReal s = PetscSinReal(std::numbers::pi_v<PetscReal> * x);
    return s * s;
}

inline PetscReal polyProfile(const PetscReal x) {
    const PetscReal y = 1.0 - x;
    return x * x * y * y;
}

inline PetscReal polySecond(const PetscReal x) {
    return 2.0 * (1.0 - 6.0 * x + 6.0 * x * x);
}

inline PetscReal gaussianProfile(const PetscReal x, const PetscReal sigma = 0.04) {
    const PetscReal z = (x - 0.5) / sigma;
    return PetscExpReal(-0.5 * z * z) /
           (sigma * PetscSqrtReal(2.0 * std::numbers::pi_v<PetscReal>));
}

inline PetscReal mass(const Mesh& mesh, const std::vector<PetscReal>& psi) {
    PetscReal sum = 0.0;
    for (const PetscReal value : psi) {
        sum += value;
    }
    return mesh.h * sum;
}

inline PetscReal minValue(const std::vector<PetscReal>& values) {
    return *std::min_element(values.begin(), values.end());
}

inline PetscReal variance(const Mesh& mesh, const std::vector<PetscReal>& psi) {
    const PetscReal m = mass(mesh, psi);
    if (m == 0.0) {
        return 0.0;
    }
    PetscReal sum = 0.0;
    for (PetscInt i = 0; i < mesh.n; ++i) {
        const PetscReal dx = xCell(mesh, i) - 0.5;
        sum += dx * dx * psi[static_cast<std::size_t>(i)];
    }
    return mesh.h * sum / m;
}

inline PetscReal freeEnergy(const Mesh& mesh,
                            const bgc::models::tsompsiv::Constants& constants,
                            const std::vector<PetscReal>& psi,
                            const std::vector<PetscReal>& v) {
    const PetscReal eta = constants.rho / constants.alpha;
    PetscReal sum = 0.0;
    for (PetscInt i = 0; i < mesh.n; ++i) {
        const auto idx = static_cast<std::size_t>(i);
        const PetscReal u = psi[idx] - v[idx];
        sum += u * u + eta * v[idx] * v[idx];
    }
    return 0.5 * mesh.h * sum;
}

inline bgc::BoundarySet noFluxBoundaries() {
    return {
        .west = {.conditions = {bgc::BoundaryCondition::neumann(0.0),
                                bgc::BoundaryCondition::neumann(0.0)}},
        .east = {.conditions = {bgc::BoundaryCondition::neumann(0.0),
                                bgc::BoundaryCondition::neumann(0.0)}},
    };
}

inline bgc::BoundarySet dirichletNeumannBoundaries() {
    return {
        .west = {.conditions = {bgc::BoundaryCondition::dirichlet(0.0),
                                bgc::BoundaryCondition::neumann(0.0)}},
        .east = {.conditions = {bgc::BoundaryCondition::dirichlet(0.0),
                                bgc::BoundaryCondition::neumann(0.0)}},
    };
}

inline bgc::models::tsompsiv::Constants tsomConstants(const PetscReal alpha,
                                                      const PetscReal rho,
                                                      const PetscReal lambdaC,
                                                      const PetscReal lambdaR,
                                                      const PetscReal theta) {
    bgc::models::tsompsiv::Constants c;
    c.alpha = alpha;
    c.rho = rho;
    c.lambdaC = lambdaC;
    c.lambdaR = lambdaR;
    c.theta = theta;
    c.vBoundary = {
        .type = bgc::models::tsompsiv::VBoundaryCondition::ConstantGradient,
        .sc = 0.0,
    };
    return c;
}

inline std::vector<PetscReal> flatten(const std::vector<PetscReal>& psi,
                                      const std::vector<PetscReal>& v) {
    std::vector<PetscReal> state(psi.size() + v.size(), 0.0);
    std::copy(psi.begin(), psi.end(), state.begin());
    std::copy(v.begin(), v.end(), state.begin() + static_cast<std::ptrdiff_t>(psi.size()));
    return state;
}

inline void splitState(const std::vector<PetscReal>& state,
                       std::vector<PetscReal>& psi,
                       std::vector<PetscReal>& v) {
    const std::size_t n = state.size() / 2;
    psi.assign(state.begin(), state.begin() + static_cast<std::ptrdiff_t>(n));
    v.assign(state.begin() + static_cast<std::ptrdiff_t>(n), state.end());
}

inline std::vector<PetscReal> tsomRhs(const Mesh& mesh,
                                      const std::vector<PetscReal>& state,
                                      const bgc::DiscreteRHS& boundaryRhs = {}) {
    const PetscReal hdt = mesh.h / mesh.dt;
    std::vector<PetscReal> rhs(state.size(), 0.0);
    for (std::size_t i = 0; i < state.size(); ++i) {
        rhs[i] = hdt * state[i];
    }
    for (const bgc::RHSEntry& entry : boundaryRhs.entries) {
        rhs[static_cast<std::size_t>(entry.row)] += entry.value;
    }
    return rhs;
}

inline void addTsomSource(const Mesh& mesh,
                          const PetscReal time,
                          const std::vector<PetscReal>& sPsi,
                          const std::vector<PetscReal>& sV,
                          std::vector<PetscReal>& rhs,
                          const PetscReal decayLambda = 0.0) {
    const std::size_t n = sPsi.size();
    const PetscReal factor = PetscExpReal(-decayLambda * time);
    for (std::size_t i = 0; i < n; ++i) {
        rhs[i] += mesh.h * factor * sPsi[i];
        rhs[n + i] += mesh.h * factor * sV[i];
    }
}

inline Sample makeTsomSample(const Mesh& mesh,
                             const bgc::models::tsompsiv::Constants& constants,
                             const PetscReal time,
                             const std::vector<PetscReal>& psi,
                             const std::vector<PetscReal>& v,
                             const bool keepFields) {
    const PetscInt center = mesh.n / 2;
    const PetscReal denom = psi[static_cast<std::size_t>(center)];
    Sample s;
    s.time = time;
    if (keepFields) {
        s.psi = psi;
        s.v = v;
    }
    s.centerRatio = denom != 0.0 ? v[static_cast<std::size_t>(center)] / denom : 0.0;
    s.mass = mass(mesh, psi);
    s.energy = freeEnergy(mesh, constants, psi, v);
    s.variance = variance(mesh, psi);
    return s;
}

inline TsomRun runTsom(const Mesh& mesh,
                       const bgc::models::tsompsiv::Constants& constants,
                       const bgc::BoundarySet& boundaries,
                       const std::vector<PetscReal>& psi0,
                       const std::vector<PetscReal>& v0,
                       const std::vector<PetscReal>& sampleTimes = {},
                       const bool keepSampleFields = false,
                       const std::vector<PetscReal>* sPsi = nullptr,
                       const std::vector<PetscReal>* sV = nullptr,
                       const PetscReal sourceDecayLambda = 0.0) {
    const bgc::Grid1D grid {.nx = mesh.n, .length = 1.0, .x0 = 0.0};
    const bgc::TimeConfig time {.dt = mesh.dt, .finalTime = mesh.tf, .initialTime = 0.0};
    const auto coefficients =
        bgc::models::tsompsiv::computeCoefficients(grid, time, constants, boundaries);
    const auto op = bgc::models::tsompsiv::buildOperator(grid, coefficients);
    const auto boundary =
        bgc::models::tsompsiv::buildBoundaryRHS(
            grid,
            bgc::models::tsompsiv::computeBoundaryRHS(grid, constants, boundaries, 0.0));

    std::vector<PetscReal> state = flatten(psi0, v0);
    std::vector<PetscReal> psi = psi0;
    std::vector<PetscReal> v = v0;
    TsomRun run {.mesh = mesh, .psi = psi0, .v = v0};
    run.mass0 = mass(mesh, psi0);
    run.energy0 = freeEnergy(mesh, constants, psi0, v0);

    std::size_t nextSample = 0;
    auto maybeSample = [&](const PetscReal t) {
        while (nextSample < sampleTimes.size() && t + 0.5 * mesh.dt >= sampleTimes[nextSample]) {
            run.samples.push_back(makeTsomSample(mesh, constants, t, psi, v, keepSampleFields));
            ++nextSample;
        }
    };
    maybeSample(0.0);

    bgc::models::bgc::MMSLinearSolver solver(op, 0.0);
    KSPConvergedReason reason = KSP_CONVERGED_ITERATING;
    PetscInt iterations = 0;
    for (PetscInt step = 1; step <= mesh.steps; ++step) {
        std::vector<PetscReal> rhs = tsomRhs(mesh, state, boundary);
        if (sPsi != nullptr && sV != nullptr) {
            addTsomSource(mesh,
                          static_cast<PetscReal>(step) * mesh.dt,
                          *sPsi,
                          *sV,
                          rhs,
                          sourceDecayLambda);
        }
        PetscCallAbort(PETSC_COMM_WORLD, solver.solve(rhs, state, reason, iterations));
        splitState(state, psi, v);
        maybeSample(static_cast<PetscReal>(step) * mesh.dt);
    }

    run.psi = psi;
    run.v = v;
    run.minPsi = minValue(psi);
    run.massFinal = mass(mesh, psi);
    run.energyFinal = freeEnergy(mesh, constants, psi, v);
    return run;
}

inline bgc::DiscreteOperator fickianOperator(const Mesh& mesh,
                                             const PetscReal gamma,
                                             const bool dirichlet) {
    bgc::DiscreteOperator op {
        .layout = {.kind = bgc::MatrixLayoutKind::Scalar,
                   .rows = mesh.n,
                   .cols = mesh.n,
                   .blockSize = 1},
        .entries = {},
    };
    const PetscReal hdt = mesh.h / mesh.dt;
    const PetscReal a = gamma / mesh.h;
    for (PetscInt i = 0; i < mesh.n; ++i) {
        if (i == 0) {
            op.entries.push_back({.row = i, .col = i, .value = hdt + (dirichlet ? 3.0 * a : a)});
            op.entries.push_back({.row = i, .col = i + 1, .value = -a});
        } else if (i == mesh.n - 1) {
            op.entries.push_back({.row = i, .col = i - 1, .value = -a});
            op.entries.push_back({.row = i, .col = i, .value = hdt + (dirichlet ? 3.0 * a : a)});
        } else {
            op.entries.push_back({.row = i, .col = i - 1, .value = -a});
            op.entries.push_back({.row = i, .col = i, .value = hdt + 2.0 * a});
            op.entries.push_back({.row = i, .col = i + 1, .value = -a});
        }
    }
    return op;
}

inline ScalarRun runFickian(const Mesh& mesh,
                            const PetscReal gamma,
                            const bool dirichlet,
                            const std::vector<PetscReal>& phi0,
                            const std::vector<PetscReal>& sampleTimes = {}) {
    std::vector<PetscReal> phi = phi0;
    ScalarRun run {.mesh = mesh, .phi = phi0};
    run.mass0 = mass(mesh, phi0);
    const auto op = fickianOperator(mesh, gamma, dirichlet);
    bgc::models::bgc::MMSLinearSolver solver(op, 0.0);
    KSPConvergedReason reason = KSP_CONVERGED_ITERATING;
    PetscInt iterations = 0;
    std::size_t nextSample = 0;
    auto maybeSample = [&](const PetscReal t) {
        while (nextSample < sampleTimes.size() && t + 0.5 * mesh.dt >= sampleTimes[nextSample]) {
            Sample s;
            s.time = t;
            s.psi = phi;
            s.mass = mass(mesh, phi);
            s.variance = variance(mesh, phi);
            run.samples.push_back(std::move(s));
            ++nextSample;
        }
    };
    maybeSample(0.0);
    const PetscReal hdt = mesh.h / mesh.dt;
    for (PetscInt step = 1; step <= mesh.steps; ++step) {
        std::vector<PetscReal> rhs(phi.size(), 0.0);
        for (std::size_t i = 0; i < phi.size(); ++i) {
            rhs[i] = hdt * phi[i];
        }
        PetscCallAbort(PETSC_COMM_WORLD, solver.solve(rhs, phi, reason, iterations));
        maybeSample(static_cast<PetscReal>(step) * mesh.dt);
    }
    run.phi = phi;
    run.minPhi = minValue(phi);
    run.massFinal = mass(mesh, phi);
    run.varianceFinal = variance(mesh, phi);
    return run;
}

inline ScalarRun runBGC(const Mesh& mesh,
                        const PetscReal bv,
                        const bgc::BoundarySet& boundaries,
                        const std::vector<PetscReal>& phi0,
                        const std::vector<PetscReal>& sampleTimes = {}) {
    const bgc::Grid1D grid {.nx = mesh.n, .length = 1.0, .x0 = 0.0};
    bgc::models::bgc::Constants constants {.bv = bv};
    const auto coefficients = bgc::models::bgc::computeCoefficients(grid, constants);
    const auto op = bgc::models::bgc::buildOperator(grid, coefficients);
    const auto boundary =
        bgc::models::bgc::buildBoundaryRHS(
            grid,
            bgc::models::bgc::computeBoundaryRHS(grid, constants, boundaries, 0.0));

    std::vector<PetscReal> phi = phi0;
    ScalarRun run {.mesh = mesh, .phi = phi0};
    run.mass0 = mass(mesh, phi0);
    bgc::models::bgc::MMSLinearSolver solver(op, mesh.h / mesh.dt);
    KSPConvergedReason reason = KSP_CONVERGED_ITERATING;
    PetscInt iterations = 0;
    std::size_t nextSample = 0;
    auto maybeSample = [&](const PetscReal t) {
        while (nextSample < sampleTimes.size() && t + 0.5 * mesh.dt >= sampleTimes[nextSample]) {
            Sample s;
            s.time = t;
            s.psi = phi;
            s.mass = mass(mesh, phi);
            s.variance = variance(mesh, phi);
            run.samples.push_back(std::move(s));
            ++nextSample;
        }
    };
    maybeSample(0.0);
    const PetscReal hdt = mesh.h / mesh.dt;
    for (PetscInt step = 1; step <= mesh.steps; ++step) {
        std::vector<PetscReal> rhs(phi.size(), 0.0);
        for (std::size_t i = 0; i < phi.size(); ++i) {
            rhs[i] = hdt * phi[i];
        }
        for (const bgc::RHSEntry& entry : boundary.entries) {
            rhs[static_cast<std::size_t>(entry.row)] += entry.value;
        }
        PetscCallAbort(PETSC_COMM_WORLD, solver.solve(rhs, phi, reason, iterations));
        maybeSample(static_cast<PetscReal>(step) * mesh.dt);
    }
    run.phi = phi;
    run.minPhi = minValue(phi);
    run.massFinal = mass(mesh, phi);
    run.varianceFinal = variance(mesh, phi);
    return run;
}

inline void writeProfile(const std::filesystem::path& path,
                         const Mesh& mesh,
                         const std::vector<std::pair<std::string, std::vector<PetscReal>>>& fields) {
    std::ofstream file = bgc::openOutputFile(path);
    file << std::scientific << std::setprecision(16);
    file << "x";
    for (const auto& [name, _] : fields) {
        file << ',' << name;
    }
    file << '\n';
    for (PetscInt i = 0; i < mesh.n; ++i) {
        const auto idx = static_cast<std::size_t>(i);
        file << xCell(mesh, i);
        for (const auto& [_, values] : fields) {
            file << ',' << values[idx];
        }
        file << '\n';
    }
}

inline PetscReal l2Difference(const Mesh& mesh,
                              const std::vector<PetscReal>& a,
                              const std::vector<PetscReal>& b) {
    PetscReal sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const PetscReal d = a[i] - b[i];
        sum += d * d;
    }
    return PetscSqrtReal(mesh.h * sum);
}

template <typename Profile>
inline std::vector<PetscReal> initialField(const Mesh& mesh, const Profile& profile) {
    std::vector<PetscReal> values(static_cast<std::size_t>(mesh.n), 0.0);
    for (PetscInt i = 0; i < mesh.n; ++i) {
        values[static_cast<std::size_t>(i)] = profile(xCell(mesh, i));
    }
    return values;
}

inline std::vector<PetscReal> constantTimes(const std::vector<PetscReal>& values,
                                            const PetscReal factor) {
    std::vector<PetscReal> out = values;
    for (PetscReal& value : out) {
        value *= factor;
    }
    return out;
}

} // namespace paper_tsompsiv
