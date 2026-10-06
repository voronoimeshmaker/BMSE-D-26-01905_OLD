#include <petsc.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "../../../paper/TSOMPsiV/common/PaperTSOMPsiV.hpp"

namespace {

namespace pt = paper_tsompsiv;

using Constants = bgc::models::tsompsiv::Constants;

struct CaseParameters {
    std::string name;
    PetscReal alpha {0.0};
    PetscReal rho {0.0};
    PetscReal lambdaC {0.0};
    PetscReal lambdaR {0.0};
    PetscReal theta {1.0};
};

struct ErrorNorms {
    PetscReal l1Psi {0.0};
    PetscReal l2Psi {0.0};
    PetscReal linfPsi {0.0};
    PetscReal l1U {0.0};
    PetscReal l2U {0.0};
    PetscReal linfU {0.0};
    PetscReal l1V {0.0};
    PetscReal l2V {0.0};
    PetscReal linfV {0.0};
};

struct StepDiagnostics {
    bool completed {true};
    PetscReal stopTime {0.0};
    PetscReal finalMass {0.0};
    PetscReal finalDrift {0.0};
    PetscReal maxDrift {0.0};
    PetscReal finalEnergy {0.0};
    PetscReal energyRatio {0.0};
    PetscReal maxEnergyIncrease {0.0};
    PetscInt positiveEnergyIncrements {0};
    PetscReal finalRatioMean {0.0};
    PetscReal finalRatioMin {0.0};
    PetscReal finalRatioMax {0.0};
    PetscReal finalRatioMaxError {0.0};
};

std::filesystem::path caseDir() {
    return bgc::caseDirectoryFromSource(__FILE__);
}

std::filesystem::path outDir() {
    auto dir = caseDir() / "Saida";
    bgc::ensureDirectory(dir);
    return dir;
}

std::ofstream openCsv(const std::filesystem::path& path) {
    std::ofstream file(path);
    if (!file) {
        throw std::runtime_error("Could not open output file: " + path.string());
    }
    file << std::setprecision(16);
    return file;
}

Constants constantsFrom(const CaseParameters& p) {
    return pt::tsomConstants(p.alpha, p.rho, p.lambdaC, p.lambdaR, p.theta);
}

std::vector<PetscReal> makeProfile(const pt::Mesh& mesh,
                                   PetscReal (*profile)(PetscReal)) {
    std::vector<PetscReal> values(static_cast<std::size_t>(mesh.n), 0.0);
    for (PetscInt i = 0; i < mesh.n; ++i) {
        values[static_cast<std::size_t>(i)] = profile(pt::xCell(mesh, i));
    }
    return values;
}

PetscReal poly(const PetscReal x) {
    return pt::polyProfile(x);
}

PetscReal polySecond(const PetscReal x) {
    return pt::polySecond(x);
}

PetscReal equilibriumRetainedFraction(const Constants& c) {
    return c.alpha / (c.alpha + c.rho);
}

std::vector<PetscReal> buildExactPsi(const pt::Mesh& mesh, const PetscReal time) {
    std::vector<PetscReal> psi(static_cast<std::size_t>(mesh.n), 0.0);
    const PetscReal decay = PetscExpReal(-time);
    for (PetscInt i = 0; i < mesh.n; ++i) {
        psi[static_cast<std::size_t>(i)] = decay * poly(pt::xCell(mesh, i));
    }
    return psi;
}

std::vector<PetscReal> buildExactV(const pt::Mesh& mesh,
                                   const PetscReal time,
                                   const PetscReal retainedFraction) {
    std::vector<PetscReal> v(static_cast<std::size_t>(mesh.n), 0.0);
    const PetscReal decay = PetscExpReal(-time);
    for (PetscInt i = 0; i < mesh.n; ++i) {
        v[static_cast<std::size_t>(i)] =
            retainedFraction * decay * poly(pt::xCell(mesh, i));
    }
    return v;
}

void accumulateError(const pt::Mesh& mesh,
                     const std::vector<PetscReal>& num,
                     const std::vector<PetscReal>& exact,
                     PetscReal& l1,
                     PetscReal& l2,
                     PetscReal& linf) {
    PetscReal sumAbs = 0.0;
    PetscReal sumSq = 0.0;
    PetscReal maxAbs = 0.0;
    for (PetscInt i = 0; i < mesh.n; ++i) {
        const auto idx = static_cast<std::size_t>(i);
        const PetscReal error = PetscAbsReal(num[idx] - exact[idx]);
        sumAbs += error;
        sumSq += error * error;
        maxAbs = PetscMax(maxAbs, error);
    }
    l1 = mesh.h * sumAbs;
    l2 = PetscSqrtReal(mesh.h * sumSq);
    linf = maxAbs;
}

ErrorNorms computeErrors(const pt::Mesh& mesh,
                         const std::vector<PetscReal>& psi,
                         const std::vector<PetscReal>& v,
                         const std::vector<PetscReal>& psiExact,
                         const std::vector<PetscReal>& vExact) {
    std::vector<PetscReal> u(static_cast<std::size_t>(mesh.n), 0.0);
    std::vector<PetscReal> uExact(static_cast<std::size_t>(mesh.n), 0.0);
    for (PetscInt i = 0; i < mesh.n; ++i) {
        const auto idx = static_cast<std::size_t>(i);
        u[idx] = psi[idx] - v[idx];
        uExact[idx] = psiExact[idx] - vExact[idx];
    }

    ErrorNorms e;
    accumulateError(mesh, psi, psiExact, e.l1Psi, e.l2Psi, e.linfPsi);
    accumulateError(mesh, u, uExact, e.l1U, e.l2U, e.linfU);
    accumulateError(mesh, v, vExact, e.l1V, e.l2V, e.linfV);
    return e;
}

PetscReal order(const PetscReal previous, const PetscReal current) {
    if (previous <= 0.0 || current <= 0.0) {
        return 0.0;
    }
    return PetscLogReal(previous / current) / PetscLogReal(2.0);
}

void writeOrder(std::ofstream& file, const PetscReal previous, const PetscReal current) {
    if (previous <= 0.0) {
        file << ",nan";
    } else {
        file << "," << order(previous, current);
    }
}

void runR1(const std::filesystem::path& out) {
    const std::vector<CaseParameters> cases {
        {.name = "R1a", .alpha = 0.2, .rho = 0.3, .lambdaC = 5.0, .lambdaR = 7.5},
        {.name = "R1b", .alpha = 0.4, .rho = 0.4, .lambdaC = 10.0, .lambdaR = 10.0},
        {.name = "R1c", .alpha = 0.3, .rho = 0.6, .lambdaC = 2.0, .lambdaR = 4.0},
    };
    const std::vector<PetscInt> nList {8, 16, 32, 64, 128, 256};
    const PetscReal tf = 1.0;

    auto file = openCsv(out / "R1_mms_convergence.csv");
    file << "case,alpha,rho,lambda_c,lambda_r,n,h,dt,steps,"
         << "l1_psi,order_l1_psi,l2_psi,order_l2_psi,linf_psi,order_linf_psi,"
         << "l1_u,order_l1_u,l2_u,order_l2_u,linf_u,order_linf_u,"
         << "l1_v,order_l1_v,l2_v,order_l2_v,linf_v,order_linf_v\n";

    for (const auto& p : cases) {
        PetscReal prevL1Psi = 0.0;
        PetscReal prevL2Psi = 0.0;
        PetscReal prevLinfPsi = 0.0;
        PetscReal prevL1U = 0.0;
        PetscReal prevL2U = 0.0;
        PetscReal prevLinfU = 0.0;
        PetscReal prevL1V = 0.0;
        PetscReal prevL2V = 0.0;
        PetscReal prevLinfV = 0.0;
        for (const PetscInt n : nList) {
            const pt::Mesh mesh = pt::makeMesh(n, tf);
            const Constants c = constantsFrom(p);
            const PetscReal f = equilibriumRetainedFraction(c);
            const auto d = bgc::models::tsompsiv::computeDiffusionCoefficients(c);

            std::vector<PetscReal> sPsi(static_cast<std::size_t>(n), 0.0);
            std::vector<PetscReal> sV(static_cast<std::size_t>(n), 0.0);
            for (PetscInt i = 0; i < n; ++i) {
                const auto idx = static_cast<std::size_t>(i);
                const PetscReal x = pt::xCell(mesh, i);
                const PetscReal g = poly(x);
                const PetscReal gpp = polySecond(x);
                const PetscReal psiDiffusion = d.d11 + d.d12 * f;
                const PetscReal vReaction = -f - c.lambdaC + (c.lambdaC + c.lambdaR) * f;
                const PetscReal vDiffusion = d.d21 + d.d22 * f;
                sPsi[idx] = -g - psiDiffusion * gpp;
                sV[idx] = vReaction * g - vDiffusion * gpp;
            }

            const std::vector<PetscReal> psi0 = buildExactPsi(mesh, 0.0);
            const std::vector<PetscReal> v0 = buildExactV(mesh, 0.0, f);
            const pt::TsomRun run = pt::runTsom(mesh,
                                                c,
                                                pt::noFluxBoundaries(),
                                                psi0,
                                                v0,
                                                {},
                                                false,
                                                &sPsi,
                                                &sV,
                                                1.0);
            const std::vector<PetscReal> psiExact = buildExactPsi(mesh, tf);
            const std::vector<PetscReal> vExact = buildExactV(mesh, tf, f);
            const ErrorNorms e = computeErrors(mesh, run.psi, run.v, psiExact, vExact);

            file << p.name << "," << p.alpha << "," << p.rho << ","
                 << p.lambdaC << "," << p.lambdaR << "," << n << ","
                 << mesh.h << "," << mesh.dt << "," << mesh.steps << ","
                 << e.l1Psi;
            writeOrder(file, prevL1Psi, e.l1Psi);
            file << "," << e.l2Psi;
            writeOrder(file, prevL2Psi, e.l2Psi);
            file << "," << e.linfPsi;
            writeOrder(file, prevLinfPsi, e.linfPsi);
            file << "," << e.l1U;
            writeOrder(file, prevL1U, e.l1U);
            file << "," << e.l2U;
            writeOrder(file, prevL2U, e.l2U);
            file << "," << e.linfU;
            writeOrder(file, prevLinfU, e.linfU);
            file << "," << e.l1V;
            writeOrder(file, prevL1V, e.l1V);
            file << "," << e.l2V;
            writeOrder(file, prevL2V, e.l2V);
            file << "," << e.linfV;
            writeOrder(file, prevLinfV, e.linfV);
            file << "\n";

            prevL1Psi = e.l1Psi;
            prevL2Psi = e.l2Psi;
            prevLinfPsi = e.linfPsi;
            prevL1U = e.l1U;
            prevL2U = e.l2U;
            prevLinfU = e.linfU;
            prevL1V = e.l1V;
            prevL2V = e.l2V;
            prevLinfV = e.linfV;
        }
    }
}

StepDiagnostics runSourceFreeDiagnostics(const pt::Mesh& mesh,
                                         const Constants& c,
                                         const std::vector<PetscReal>& psi0,
                                         const std::vector<PetscReal>& v0,
                                         const std::filesystem::path* historyPath = nullptr) {
    const bgc::Grid1D grid {.nx = mesh.n, .length = 1.0, .x0 = 0.0};
    const bgc::TimeConfig time {.dt = mesh.dt, .finalTime = mesh.tf, .initialTime = 0.0};
    const auto coefficients =
        bgc::models::tsompsiv::computeCoefficients(grid, time, c, pt::noFluxBoundaries());
    const auto op = bgc::models::tsompsiv::buildOperator(grid, coefficients);
    const auto boundary = bgc::models::tsompsiv::buildBoundaryRHS(
        grid,
        bgc::models::tsompsiv::computeBoundaryRHS(grid, c, pt::noFluxBoundaries(), 0.0));

    std::vector<PetscReal> state = pt::flatten(psi0, v0);
    std::vector<PetscReal> psi = psi0;
    std::vector<PetscReal> v = v0;
    const PetscReal mass0 = pt::mass(mesh, psi);
    const PetscReal energy0 = pt::freeEnergy(mesh, c, psi, v);
    PetscReal previousEnergy = energy0;
    PetscReal maxDrift = 0.0;
    PetscReal maxEnergyIncrease = 0.0;
    PetscInt positiveEnergyIncrements = 0;

    std::ofstream history;
    if (historyPath != nullptr) {
        history = openCsv(*historyPath);
        history << "step,time,mass,mass_drift,energy,energy_ratio,delta_energy,"
                << "center_ratio\n";
        const PetscInt center = mesh.n / 2;
        history << 0 << "," << 0.0 << "," << mass0 << "," << 0.0 << ","
                << energy0 << "," << 1.0 << "," << 0.0 << ","
                << v[static_cast<std::size_t>(center)] /
                       psi[static_cast<std::size_t>(center)]
                << "\n";
    }

    bgc::models::bgc::MMSLinearSolver solver(op, 0.0);
    KSPConvergedReason reason = KSP_CONVERGED_ITERATING;
    PetscInt iterations = 0;
    for (PetscInt step = 1; step <= mesh.steps; ++step) {
        std::vector<PetscReal> rhs = pt::tsomRhs(mesh, state, boundary);
        PetscCallAbort(PETSC_COMM_WORLD, solver.solve(rhs, state, reason, iterations));
        pt::splitState(state, psi, v);
        const PetscReal currentMass = pt::mass(mesh, psi);
        const PetscReal currentEnergy = pt::freeEnergy(mesh, c, psi, v);
        const bool finiteStep = std::isfinite(currentMass) && std::isfinite(currentEnergy);
        const PetscReal drift = PetscAbsReal(currentMass - mass0);
        maxDrift = PetscMax(maxDrift, drift);
        const PetscReal deltaEnergy = currentEnergy - previousEnergy;
        if (deltaEnergy > maxEnergyIncrease) {
            maxEnergyIncrease = deltaEnergy;
        }
        if (deltaEnergy > 32.0 * PETSC_MACHINE_EPSILON * PetscMax(1.0, energy0)) {
            ++positiveEnergyIncrements;
        }
        if (history) {
            const PetscInt center = mesh.n / 2;
            history << step << "," << static_cast<PetscReal>(step) * mesh.dt << ","
                    << currentMass << "," << currentMass - mass0 << ","
                    << currentEnergy << "," << currentEnergy / energy0 << ","
                    << deltaEnergy << ","
                    << v[static_cast<std::size_t>(center)] /
                           psi[static_cast<std::size_t>(center)]
                    << "\n";
        }
        if (!finiteStep) {
            return {
                .completed = false,
                .stopTime = static_cast<PetscReal>(step) * mesh.dt,
                .finalMass = currentMass,
                .finalDrift = currentMass - mass0,
                .maxDrift = maxDrift,
                .finalEnergy = currentEnergy,
                .energyRatio = currentEnergy / energy0,
                .maxEnergyIncrease = maxEnergyIncrease,
                .positiveEnergyIncrements = positiveEnergyIncrements,
                .finalRatioMean = std::numeric_limits<PetscReal>::quiet_NaN(),
                .finalRatioMin = std::numeric_limits<PetscReal>::quiet_NaN(),
                .finalRatioMax = std::numeric_limits<PetscReal>::quiet_NaN(),
                .finalRatioMaxError = std::numeric_limits<PetscReal>::quiet_NaN(),
            };
        }
        previousEnergy = currentEnergy;
    }

    const PetscReal feq = equilibriumRetainedFraction(c);
    PetscReal ratioSum = 0.0;
    PetscReal ratioMin = PETSC_MAX_REAL;
    PetscReal ratioMax = -PETSC_MAX_REAL;
    PetscReal ratioMaxError = 0.0;
    for (PetscInt i = 0; i < mesh.n; ++i) {
        const auto idx = static_cast<std::size_t>(i);
        const PetscReal ratio = v[idx] / psi[idx];
        ratioSum += ratio;
        ratioMin = PetscMin(ratioMin, ratio);
        ratioMax = PetscMax(ratioMax, ratio);
        ratioMaxError = PetscMax(ratioMaxError, PetscAbsReal(ratio - feq));
    }

    const PetscReal finalMass = pt::mass(mesh, psi);
    const PetscReal finalEnergy = pt::freeEnergy(mesh, c, psi, v);
    return {
        .completed = true,
        .stopTime = mesh.tf,
        .finalMass = finalMass,
        .finalDrift = finalMass - mass0,
        .maxDrift = maxDrift,
        .finalEnergy = finalEnergy,
        .energyRatio = finalEnergy / energy0,
        .maxEnergyIncrease = maxEnergyIncrease,
        .positiveEnergyIncrements = positiveEnergyIncrements,
        .finalRatioMean = ratioSum / static_cast<PetscReal>(mesh.n),
        .finalRatioMin = ratioMin,
        .finalRatioMax = ratioMax,
        .finalRatioMaxError = ratioMaxError,
    };
}

void runR2(const std::filesystem::path& out) {
    const std::vector<CaseParameters> cases {
        {.name = "R2_fast_exchange", .alpha = 0.2, .rho = 0.3, .lambdaC = 20.0, .lambdaR = 20.0},
        {.name = "R2_slow_exchange", .alpha = 0.2, .rho = 0.3, .lambdaC = 0.5, .lambdaR = 0.5},
    };
    const std::vector<PetscInt> nList {32, 64, 128, 256, 512};
    auto file = openCsv(out / "R2_mass_conservation.csv");
    file << "case,alpha,rho,lambda_c,lambda_r,n,h,dt,steps,mass0,mass_final,"
         << "final_drift,max_abs_drift\n";
    for (const auto& p : cases) {
        for (const PetscInt n : nList) {
            const pt::Mesh mesh = pt::makeMesh(n, 1.0);
            const Constants c = constantsFrom(p);
            const std::vector<PetscReal> psi0 = makeProfile(mesh, pt::sin2Profile);
            std::vector<PetscReal> v0(static_cast<std::size_t>(n), 0.0);
            const StepDiagnostics d = runSourceFreeDiagnostics(mesh, c, psi0, v0);
            file << p.name << "," << p.alpha << "," << p.rho << ","
                 << p.lambdaC << "," << p.lambdaR << "," << n << ","
                 << mesh.h << "," << mesh.dt << "," << mesh.steps << ","
                 << pt::mass(mesh, psi0) << "," << d.finalMass << ","
                 << d.finalDrift << "," << d.maxDrift << "\n";
        }
    }
}

void runR3R4(const std::filesystem::path& out) {
    const std::vector<CaseParameters> cases {
        {.name = "R3a_R4a", .alpha = 0.1, .rho = 0.4, .lambdaC = 1.0, .lambdaR = 4.0},
        {.name = "R3b_R4b", .alpha = 0.4, .rho = 0.2, .lambdaC = 1.0, .lambdaR = 0.5},
    };
    const PetscInt n = 256;
    auto summary = openCsv(out / "R3_R4_partition_energy.csv");
    summary << "case,alpha,rho,lambda_c,lambda_r,equilibrium_ratio,n,h,dt,steps,"
            << "completed,stop_time,"
            << "final_ratio_mean,final_ratio_min,final_ratio_max,final_ratio_max_error,"
            << "energy_ratio_final,max_energy_increase,positive_energy_increments,"
            << "mass_final\n";
    for (const auto& p : cases) {
        const pt::Mesh mesh = pt::makeMesh(n, 5.0);
        const Constants c = constantsFrom(p);
        const std::vector<PetscReal> psi0 = makeProfile(mesh, pt::sin2Profile);
        std::vector<PetscReal> v0(static_cast<std::size_t>(n), 0.0);
        const std::filesystem::path history = out / (p.name + "_history.csv");
        const StepDiagnostics d = runSourceFreeDiagnostics(mesh, c, psi0, v0, &history);
        summary << p.name << "," << p.alpha << "," << p.rho << ","
                << p.lambdaC << "," << p.lambdaR << ","
                << equilibriumRetainedFraction(c) << "," << n << ","
                << mesh.h << "," << mesh.dt << "," << mesh.steps << ","
                << (d.completed ? 1 : 0) << "," << d.stopTime << ","
                << d.finalRatioMean << "," << d.finalRatioMin << ","
                << d.finalRatioMax << "," << d.finalRatioMaxError << ","
                << d.energyRatio << "," << d.maxEnergyIncrease << ","
                << d.positiveEnergyIncrements << "," << d.finalMass << "\n";
    }
}

void runR5(const std::filesystem::path& out) {
    const std::vector<CaseParameters> cases {
        {.name = "R5_theta_half", .alpha = 0.2, .rho = 0.3, .lambdaC = 1.0, .lambdaR = 1.5, .theta = 0.5},
        {.name = "R5_theta_one", .alpha = 0.2, .rho = 0.3, .lambdaC = 1.0, .lambdaR = 1.5, .theta = 1.0},
    };
    const PetscInt n = 256;
    const PetscReal tf = 5.0;
    auto summary = openCsv(out / "R5_energy_theta_comparison.csv");
    summary << "case,theta,alpha,rho,lambda_c,lambda_r,equilibrium_ratio,n,h,dt,steps,"
            << "completed,stop_time,"
            << "energy_ratio_final,max_energy_increase,positive_energy_increments,"
            << "final_ratio_mean,final_ratio_min,final_ratio_max,final_ratio_max_error,"
            << "mass_final\n";
    for (const auto& p : cases) {
        const pt::Mesh mesh = pt::makeMesh(n, tf, 0.1);
        const Constants c = constantsFrom(p);
        const std::vector<PetscReal> psi0 = makeProfile(mesh, pt::sin2Profile);
        std::vector<PetscReal> v0(static_cast<std::size_t>(n), 0.0);
        const std::filesystem::path history = out / (p.name + "_history.csv");
        const StepDiagnostics d = runSourceFreeDiagnostics(mesh, c, psi0, v0, &history);
        summary << p.name << "," << p.theta << "," << p.alpha << ","
                << p.rho << "," << p.lambdaC << "," << p.lambdaR << ","
                << equilibriumRetainedFraction(c) << "," << n << ","
                << mesh.h << "," << mesh.dt << "," << mesh.steps << ","
                << (d.completed ? 1 : 0) << "," << d.stopTime << ","
                << d.energyRatio << "," << d.maxEnergyIncrease << ","
                << d.positiveEnergyIncrements << "," << d.finalRatioMean << ","
                << d.finalRatioMin << "," << d.finalRatioMax << ","
                << d.finalRatioMaxError << "," << d.finalMass << "\n";
    }
}

void writeReadme(const std::filesystem::path& out) {
    std::ofstream file(out / "README.txt");
    file << "MMSPsiV verification suite\n"
         << "R1: manufactured-solution convergence for theta=1.\n"
         << "R2: source-free total-mass conservation for fast and slow exchange.\n"
         << "R3: final V/Psi equilibrium-partition check.\n"
         << "R4: source-free free-energy monotonicity for R3 parameters.\n"
         << "R5: theta=1/2 energy decay check, compared with theta=1.\n";
}

} // namespace

int main(int argc, char** argv) {
    PetscCall(PetscInitialize(&argc, &argv, nullptr, nullptr));
    try {
        const auto out = outDir();
        runR1(out);
        runR2(out);
        runR3R4(out);
        runR5(out);
        writeReadme(out);
        std::cout << "MMSPsiV suite completed. Output: " << out << "\n";
    } catch (const std::exception& ex) {
        std::cerr << "MMSPsiV failed: " << ex.what() << "\n";
        PetscCall(PetscFinalize());
        return 1;
    }
    PetscCall(PetscFinalize());
    return 0;
}
