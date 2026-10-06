#include <petsc.h>

#include <bgclib/BGCLib.hpp>

// MMS1TSOMPsiV exercises the new bgclib TSOMPsiV model path with a manufactured
// two-field solution. The driver owns only the manufactured Psi/V fields, source
// terms, and output table layout. TSOMPsiV coefficients, V boundary-condition
// variants, PETSc operator assembly, truncation error, and finite-volume norms
// come from bgclib.
//
// Unknown ordering is flattened as:
//
//   [ Psi_0 ... Psi_{N-1}  V_0 ... V_{N-1} ]

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace {

using InputData = bgc::models::bgc::MMSInputData;
using MeshTimeData = bgc::models::bgc::MMSMeshTimeData;

enum class MMS1Profile {
    Sine,
    OnePlusCosine,
};

struct MMS1Case {
    std::string name;
    bgc::models::tsompsiv::VBoundaryCondition vBoundary {
        bgc::models::tsompsiv::VBoundaryCondition::ZeroValue
    };
    MMS1Profile profile {MMS1Profile::Sine};
    PetscReal theta {0.0};
};

struct MMS1ManufacturedParameters {
    PetscReal amplitude {1.0};
    PetscReal lambda {1.0};
    PetscReal vFraction {0.5};
};

struct NormRecord {
    PetscInt nx {0};
    PetscReal h {0.0};
    PetscReal dt {0.0};
    PetscInt nTimes {0};
    PetscReal l1Psi {0.0};
    PetscReal l2Psi {0.0};
    PetscReal linfPsi {0.0};
    PetscReal l1V {0.0};
    PetscReal l2V {0.0};
    PetscReal linfV {0.0};
    PetscReal lteL2Psi {0.0};
    PetscReal lteLinfPsi {0.0};
    PetscReal lteL2V {0.0};
    PetscReal lteLinfV {0.0};
    KSPConvergedReason reason {KSP_CONVERGED_ITERATING};
    PetscInt iterations {0};
};

std::filesystem::path caseDirectory() {
    return bgc::caseDirectoryFromSource(__FILE__);
}

InputData defaultInput() {
    return {
        .tf = 1.0,
        .cdt = 1.024e-1,
        .nxList = {8, 16, 32, 64, 128, 256, 512},
        .bvList = {1.0},
        .outputDir = "Saida",
        .verbose = true,
        .debug = true,
    };
}

bgc::models::tsompsiv::Constants defaultConstants() {
    bgc::models::tsompsiv::Constants constants;
    constants.alpha = 0.75;
    constants.rho = 0.25;
    constants.theta = 0.0;
    constants.lambdaC = 1.0;
    constants.lambdaR = 1.0 / 3.0;
    constants.vBoundary = {
        .type = bgc::models::tsompsiv::VBoundaryCondition::ZeroValue,
        .sc = 0.5,
    };
    return constants;
}

bgc::models::tsompsiv::Constants readTSOMPsiVConstants(const std::filesystem::path& dataDir) {
    bgc::models::tsompsiv::Constants constants = defaultConstants();
    const std::unordered_map<std::string, std::string> data =
        ::bgc::readKeyValueFile(dataDir / "simulation.dat");

    if (data.contains("alpha")) {
        constants.alpha = ::bgc::parseReal(data.at("alpha"));
    }
    if (data.contains("alpha_tsompsiv")) {
        constants.alpha = ::bgc::parseReal(data.at("alpha_tsompsiv"));
    }
    if (data.contains("rho")) {
        constants.rho = ::bgc::parseReal(data.at("rho"));
    }
    if (data.contains("rho_tsompsiv")) {
        constants.rho = ::bgc::parseReal(data.at("rho_tsompsiv"));
    }
    if (data.contains("theta")) {
        constants.theta = ::bgc::parseReal(data.at("theta"));
    }
    if (data.contains("theta_tsompsiv")) {
        constants.theta = ::bgc::parseReal(data.at("theta_tsompsiv"));
    }
    if (data.contains("lambda_c")) {
        constants.lambdaC = ::bgc::parseReal(data.at("lambda_c"));
    }
    if (data.contains("lambdac")) {
        constants.lambdaC = ::bgc::parseReal(data.at("lambdac"));
    }
    if (data.contains("lambda_r")) {
        constants.lambdaR = ::bgc::parseReal(data.at("lambda_r"));
    }
    if (data.contains("lambdar")) {
        constants.lambdaR = ::bgc::parseReal(data.at("lambdar"));
    }
    if (data.contains("v_boundary")) {
        constants.vBoundary.type = bgc::models::tsompsiv::parseVBoundaryCondition(
            ::bgc::trim(data.at("v_boundary")));
    }
    if (data.contains("v_boundary_condition")) {
        constants.vBoundary.type = bgc::models::tsompsiv::parseVBoundaryCondition(
            ::bgc::trim(data.at("v_boundary_condition")));
    }
    if (data.contains("v_boundary_type")) {
        constants.vBoundary.type = bgc::models::tsompsiv::parseVBoundaryCondition(
            ::bgc::trim(data.at("v_boundary_type")));
    }
    if (data.contains("sc")) {
        constants.vBoundary.sc = ::bgc::parseReal(data.at("sc"));
    }
    if (data.contains("v_boundary_sc")) {
        constants.vBoundary.sc =
            ::bgc::parseReal(data.at("v_boundary_sc"));
    }

    if (!bgc::ModelTraits<bgc::models::tsompsiv::Tag>::constantsAreValid(constants)) {
        throw std::runtime_error("Invalid TSOMPsiV constants in simulation.dat.");
    }
    return constants;
}

bgc::BoundarySet makeMMS1Boundaries(const MMS1Case& testCase) {
    const bool usesPsiDirichlet =
        testCase.vBoundary == bgc::models::tsompsiv::VBoundaryCondition::ZeroValue;
    return {
        .west = {
            .conditions = {
                usesPsiDirichlet ? bgc::BoundaryCondition::dirichlet(0.0)
                                 : bgc::BoundaryCondition::neumann(0.0),
                bgc::BoundaryCondition::neumann(0.0),
            },
        },
        .east = {
            .conditions = {
                usesPsiDirichlet ? bgc::BoundaryCondition::dirichlet(0.0)
                                 : bgc::BoundaryCondition::neumann(0.0),
                bgc::BoundaryCondition::neumann(0.0),
            },
        },
    };
}

std::string_view profileName(const MMS1Profile profile) noexcept {
    switch (profile) {
    case MMS1Profile::Sine:
        return "sin(pi*x)";
    case MMS1Profile::OnePlusCosine:
        return "1+cos(pi*x)";
    }
    return "unknown";
}

PetscReal gMMS1(const MMS1Profile profile, const PetscReal x) {
    const PetscReal pi = std::numbers::pi_v<PetscReal>;
    switch (profile) {
    case MMS1Profile::Sine:
        return PetscSinReal(pi * x);
    case MMS1Profile::OnePlusCosine:
        return 1.0 + PetscCosReal(pi * x);
    }
    return 0.0;
}

PetscReal gppMMS1(const MMS1Profile profile, const PetscReal x) {
    const PetscReal pi = std::numbers::pi_v<PetscReal>;
    switch (profile) {
    case MMS1Profile::Sine:
        return -(pi * pi) * PetscSinReal(pi * x);
    case MMS1Profile::OnePlusCosine:
        return -(pi * pi) * PetscCosReal(pi * x);
    }
    return 0.0;
}

PetscReal psiExact(const MMS1ManufacturedParameters& mms,
                   const MMS1Case& testCase,
                   const PetscReal x,
                   const PetscReal t) {
    return mms.amplitude * PetscExpReal(-mms.lambda * t) * gMMS1(testCase.profile, x);
}

PetscReal vExact(const MMS1ManufacturedParameters& mms,
                 const MMS1Case& testCase,
                 const PetscReal x,
                 const PetscReal t) {
    return mms.vFraction * psiExact(mms, testCase, x, t);
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
                                       const MMS1ManufacturedParameters& mms,
                                       const MMS1Case& testCase,
                                       const PetscReal time) {
    std::vector<PetscReal> state(static_cast<std::size_t>(2 * mesh.nx), 0.0);
    for (PetscInt i = 0; i < mesh.nx; ++i) {
        const PetscReal x = (static_cast<PetscReal>(i) + 0.5) * mesh.h;
        state[static_cast<std::size_t>(i)] = psiExact(mms, testCase, x, time);
        state[static_cast<std::size_t>(mesh.nx + i)] = vExact(mms, testCase, x, time);
    }
    return state;
}

struct MMS1RHSCache {
    std::vector<PetscReal> psiProfile;
    std::vector<PetscReal> vProfile;
    std::vector<bgc::RHSEntry> boundaryEntries;
};

MMS1RHSCache buildMMS1RHSCache(const MeshTimeData& mesh,
                               const MMS1ManufacturedParameters& mms,
                               const MMS1Case& testCase,
                               const bgc::models::tsompsiv::Constants& constants) {
    const bgc::Grid1D grid = makeGrid(mesh);
    const auto boundaryRHS = bgc::models::tsompsiv::buildBoundaryRHS(
        grid,
        bgc::models::tsompsiv::computeBoundaryRHS(
            grid,
            constants,
            makeMMS1Boundaries(testCase),
            0.0));
    const auto d = bgc::models::tsompsiv::computeDiffusionCoefficients(constants);

    MMS1RHSCache cache {
        .psiProfile = std::vector<PetscReal>(static_cast<std::size_t>(mesh.nx), 0.0),
        .vProfile = std::vector<PetscReal>(static_cast<std::size_t>(mesh.nx), 0.0),
        .boundaryEntries = boundaryRHS.entries,
    };

    const PetscReal f = mms.vFraction;
    const PetscReal psiDiffusion = d.d11 + d.d12 * f;
    const PetscReal vReaction =
        constants.lambdaR * f - constants.lambdaC * (1.0 - f) - mms.lambda * f;
    const PetscReal vDiffusion = d.d21 + d.d22 * f;

    for (PetscInt i = 0; i < mesh.nx; ++i) {
        const auto index = static_cast<std::size_t>(i);
        const PetscReal x = (static_cast<PetscReal>(i) + 0.5) * mesh.h;
        const PetscReal g = gMMS1(testCase.profile, x);
        const PetscReal gpp = gppMMS1(testCase.profile, x);
        cache.psiProfile[index] = mms.amplitude * (-mms.lambda * g - psiDiffusion * gpp);
        cache.vProfile[index] = mms.amplitude * (vReaction * g - vDiffusion * gpp);
    }

    return cache;
}

std::vector<PetscReal> buildRHS(const MeshTimeData& mesh,
                                const MMS1ManufacturedParameters& mms,
                                const MMS1RHSCache& cache,
                                const PetscReal time,
                                const std::vector<PetscReal>& previousState) {
    const PetscReal hdt = mesh.h / mesh.dt;
    const PetscReal timeFactor = PetscExpReal(-mms.lambda * time);
    std::vector<PetscReal> rhs(static_cast<std::size_t>(2 * mesh.nx), 0.0);

    for (PetscInt i = 0; i < mesh.nx; ++i) {
        const auto index = static_cast<std::size_t>(i);
        const auto vIndex = static_cast<std::size_t>(mesh.nx + i);
        rhs[index] = hdt * previousState[index] +
                     mesh.h * timeFactor * cache.psiProfile[index];
        rhs[vIndex] = hdt * previousState[vIndex] +
                      mesh.h * timeFactor * cache.vProfile[index];
    }

    for (const bgc::RHSEntry& entry : cache.boundaryEntries) {
        rhs[static_cast<std::size_t>(entry.row)] += entry.value;
    }
    return rhs;
}

bgc::DiscreteOperator buildTSOMPsiVOperator(const MeshTimeData& mesh,
                                        const InputData& input,
                                        const MMS1Case& testCase,
                                        const bgc::models::tsompsiv::Constants& constants) {
    const bgc::Grid1D grid = makeGrid(mesh);
    const auto coefficients = bgc::models::tsompsiv::computeCoefficients(
        grid,
        makeTime(mesh, input),
        constants,
        makeMMS1Boundaries(testCase));
    return bgc::models::tsompsiv::buildOperator(grid, coefficients);
}

PetscErrorCode solveTransient(const MeshTimeData& mesh,
                              const InputData& input,
                              const MMS1ManufacturedParameters& mms,
                              const MMS1Case& testCase,
                              const bgc::models::tsompsiv::Constants& constants,
                              std::vector<PetscReal>& state,
                              KSPConvergedReason& reason,
                              PetscInt& iterations) {
    PetscFunctionBeginUser;

    const auto op = buildTSOMPsiVOperator(mesh, input, testCase, constants);
    const MMS1RHSCache rhsCache = buildMMS1RHSCache(mesh, mms, testCase, constants);
    state = buildExactState(mesh, mms, testCase, 0.0);
    reason = KSP_CONVERGED_ITERATING;
    iterations = 0;

    bgc::models::bgc::MMSLinearSolver solver(op, 0.0);

    for (PetscInt step = 1; step <= mesh.nTimes; ++step) {
        const PetscReal time = static_cast<PetscReal>(step) * mesh.dt;
        const std::vector<PetscReal> rhsValues = buildRHS(mesh, mms, rhsCache, time, state);
        PetscCall(solver.solve(rhsValues, state, reason, iterations));
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

std::vector<PetscReal> computeTruncationError(const MeshTimeData& mesh,
                                              const InputData& input,
                                              const MMS1ManufacturedParameters& mms,
                                              const MMS1Case& testCase,
                                              const bgc::models::tsompsiv::Constants& constants) {
    const auto op = buildTSOMPsiVOperator(mesh, input, testCase, constants);
    const MMS1RHSCache rhsCache = buildMMS1RHSCache(mesh, mms, testCase, constants);
    const std::vector<PetscReal> exactNow = buildExactState(mesh, mms, testCase, input.tf);
    const std::vector<PetscReal> exactPrevious =
        buildExactState(mesh, mms, testCase, input.tf - mesh.dt);
    const std::vector<PetscReal> rhs = buildRHS(mesh, mms, rhsCache, input.tf, exactPrevious);
    std::vector<PetscReal> tau;

    PetscCallAbort(PETSC_COMM_SELF,
                   bgc::computeLocalTruncationError(op, exactNow, rhs, tau));
    return tau;
}

NormRecord computeNorms(const MeshTimeData& mesh,
                        const std::vector<PetscReal>& numerical,
                        const std::vector<PetscReal>& exact,
                        const std::vector<PetscReal>& tau,
                        const KSPConvergedReason reason,
                        const PetscInt iterations) {
    NormRecord record {
        .nx = mesh.nx,
        .h = mesh.h,
        .dt = mesh.dt,
        .nTimes = mesh.nTimes,
        .reason = reason,
        .iterations = iterations,
    };

    const auto psiNumerical = std::span<const PetscReal>(numerical).first(
        static_cast<std::size_t>(mesh.nx));
    const auto vNumerical = std::span<const PetscReal>(numerical).last(
        static_cast<std::size_t>(mesh.nx));
    const auto psiExactValues = std::span<const PetscReal>(exact).first(
        static_cast<std::size_t>(mesh.nx));
    const auto vExactValues = std::span<const PetscReal>(exact).last(
        static_cast<std::size_t>(mesh.nx));
    const auto tauPsi = std::span<const PetscReal>(tau).first(
        static_cast<std::size_t>(mesh.nx));
    const auto tauV = std::span<const PetscReal>(tau).last(
        static_cast<std::size_t>(mesh.nx));

    const bgc::ErrorNorms psiNorms =
        bgc::computeErrorNorms(psiNumerical, psiExactValues, mesh.h);
    const bgc::ErrorNorms vNorms =
        bgc::computeErrorNorms(vNumerical, vExactValues, mesh.h);
    const bgc::ErrorNorms tauPsiNorms = bgc::computeVectorNorms(tauPsi, mesh.h);
    const bgc::ErrorNorms tauVNorms = bgc::computeVectorNorms(tauV, mesh.h);

    record.l1Psi = psiNorms.l1;
    record.l2Psi = psiNorms.l2;
    record.linfPsi = psiNorms.linf;
    record.l1V = vNorms.l1;
    record.l2V = vNorms.l2;
    record.linfV = vNorms.linf;
    record.lteL2Psi = tauPsiNorms.l2;
    record.lteLinfPsi = tauPsiNorms.linf;
    record.lteL2V = tauVNorms.l2;
    record.lteLinfV = tauVNorms.linf;
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
    file << "# P x psi_num psi_exact psi_err v_num v_exact v_err\n";
    for (PetscInt i = 0; i < mesh.nx; ++i) {
        const auto psiIndex = static_cast<std::size_t>(i);
        const auto vIndex = static_cast<std::size_t>(mesh.nx + i);
        const PetscReal x = (static_cast<PetscReal>(i) + 0.5) * mesh.h;
        file << i + 1 << ' ' << x << ' '
             << numerical[psiIndex] << ' ' << exact[psiIndex] << ' '
             << numerical[psiIndex] - exact[psiIndex] << ' '
             << numerical[vIndex] << ' ' << exact[vIndex] << ' '
             << numerical[vIndex] - exact[vIndex] << '\n';
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
    file << "# P x_center tau_Psi tau_V\n";
    for (PetscInt i = 0; i < mesh.nx; ++i) {
        const auto psiIndex = static_cast<std::size_t>(i);
        const auto vIndex = static_cast<std::size_t>(mesh.nx + i);
        const PetscReal x = (static_cast<PetscReal>(i) + 0.5) * mesh.h;
        file << i + 1 << ' ' << x << ' ' << tau[psiIndex] << ' ' << tau[vIndex] << '\n';
    }
}

void writeConvergence(const std::filesystem::path& path,
                      const std::vector<NormRecord>& records) {
    std::ofstream file = bgc::openOutputFile(path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open output file: " + path.string());
    }

    file << std::scientific << std::setprecision(16);
    file << "nx,h,dt,nTimes,L1_Psi,L2_Psi,Linf_Psi,L1_V,L2_V,Linf_V,"
            "LTE_L2_Psi,LTE_Linf_Psi,LTE_L2_V,LTE_Linf_V,"
            "ksp_reason,ksp_iterations,status\n";
    for (const NormRecord& r : records) {
        file << r.nx << ',' << r.h << ',' << r.dt << ',' << r.nTimes
             << ',' << r.l1Psi << ',' << r.l2Psi << ',' << r.linfPsi
             << ',' << r.l1V << ',' << r.l2V << ',' << r.linfV
             << ',' << r.lteL2Psi << ',' << r.lteLinfPsi
             << ',' << r.lteL2V << ',' << r.lteLinfV
             << ',' << static_cast<int>(r.reason) << ',' << r.iterations
             << ',' << bgc::convergenceStatus(r.reason) << '\n';
    }
}

void writeSetup(const std::filesystem::path& path,
                const std::vector<NormRecord>& records,
                const MMS1ManufacturedParameters& mms,
                const MMS1Case& testCase,
                const bgc::models::tsompsiv::Constants& constants) {
    std::ofstream file = bgc::openOutputFile(path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open output file: " + path.string());
    }

    file << std::scientific << std::setprecision(16);
    file << "case,profile,amplitude,lambda,f,alpha,rho,theta,lambdaC,lambdaR,"
            "vBoundary,sc,nx,h,dt,nTimes\n";
    for (const NormRecord& r : records) {
        file << testCase.name << ',' << profileName(testCase.profile) << ','
             << mms.amplitude << ',' << mms.lambda << ',' << mms.vFraction << ','
             << constants.alpha << ',' << constants.rho << ',' << constants.theta << ','
             << constants.lambdaC << ',' << constants.lambdaR << ','
             << bgc::models::tsompsiv::toString(constants.vBoundary.type) << ','
             << constants.vBoundary.sc << ','
             << r.nx << ',' << r.h << ',' << r.dt << ',' << r.nTimes << '\n';
    }
}

std::vector<MMS1Case> makeMMS1Cases() {
    using VBC = bgc::models::tsompsiv::VBoundaryCondition;
    return {
        {.name = "bc1_zero_value_theta0", .vBoundary = VBC::ZeroValue,
         .profile = MMS1Profile::Sine, .theta = 0.0},
        {.name = "bc1_zero_value_theta05", .vBoundary = VBC::ZeroValue,
         .profile = MMS1Profile::Sine, .theta = 0.5},
        {.name = "bc1_zero_value_theta1", .vBoundary = VBC::ZeroValue,
         .profile = MMS1Profile::Sine, .theta = 1.0},
        {.name = "bc2_constant_gradient_theta0", .vBoundary = VBC::ConstantGradient,
         .profile = MMS1Profile::OnePlusCosine, .theta = 0.0},
        {.name = "bc2_constant_gradient_theta05", .vBoundary = VBC::ConstantGradient,
         .profile = MMS1Profile::OnePlusCosine, .theta = 0.5},
        {.name = "bc2_constant_gradient_theta1", .vBoundary = VBC::ConstantGradient,
         .profile = MMS1Profile::OnePlusCosine, .theta = 1.0},
        {.name = "bc3_scaled_psi_theta0", .vBoundary = VBC::ScaledPsi,
         .profile = MMS1Profile::OnePlusCosine, .theta = 0.0},
        {.name = "bc3_scaled_psi_theta05", .vBoundary = VBC::ScaledPsi,
         .profile = MMS1Profile::OnePlusCosine, .theta = 0.5},
        {.name = "bc3_scaled_psi_theta1", .vBoundary = VBC::ScaledPsi,
         .profile = MMS1Profile::OnePlusCosine, .theta = 1.0},
    };
}

bgc::models::tsompsiv::Constants constantsForCase(
    bgc::models::tsompsiv::Constants constants,
    const MMS1ManufacturedParameters& mms,
    const MMS1Case& testCase) {
    constants.theta = testCase.theta;
    constants.vBoundary.type = testCase.vBoundary;
    constants.vBoundary.sc = 1.0 - mms.vFraction;
    return constants;
}

void runCases(const InputData& input,
              const MMS1ManufacturedParameters& mms,
              const MMS1Case& testCase,
              const bgc::models::tsompsiv::Constants& constants,
              const std::filesystem::path& outputDir) {
    std::vector<NormRecord> records;
    bgc::ensureDirectory(outputDir);

    for (const PetscInt nx : input.nxList) {
        const MeshTimeData mesh = bgc::models::bgc::makeMMSMeshTimeData(input, nx);
        const std::string nxText = std::to_string(static_cast<int>(nx));
        std::vector<PetscReal> numerical;
        KSPConvergedReason reason = KSP_CONVERGED_ITERATING;
        PetscInt iterations = 0;

        PetscCallAbort(PETSC_COMM_SELF,
                       solveTransient(mesh,
                                      input,
                                      mms,
                                      testCase,
                                      constants,
                                      numerical,
                                      reason,
                                      iterations));

        const std::vector<PetscReal> exact = buildExactState(mesh, mms, testCase, input.tf);
        const std::vector<PetscReal> tau =
            computeTruncationError(mesh, input, mms, testCase, constants);
        const NormRecord record = computeNorms(mesh, numerical, exact, tau, reason, iterations);

        writeFields(outputDir / ("mms1tsompsiv_fields_N" + nxText + ".dat"),
                    mesh,
                    numerical,
                    exact);
        writeTruncation(outputDir / ("mms1tsompsiv_lte_N" + nxText + ".dat"), mesh, tau);
        records.push_back(record);
    }

    writeConvergence(outputDir / "mms1tsompsiv_convergence.csv", records);
    writeSetup(outputDir / "mms1tsompsiv_setup.csv", records, mms, testCase, constants);
}

} // namespace

int main(int argc, char** argv) {
    PetscCallAbort(PETSC_COMM_WORLD,
                   PetscInitialize(&argc,
                                   &argv,
                                   nullptr,
                                   "MMS1TSOMPsiV -- manufactured solution for the TSOMPsiV model"));

    PetscInt exitCode = 0;
    try {
        const std::filesystem::path caseDir = caseDirectory();
        const std::filesystem::path dataDir = caseDir / "Dados";
        const InputData input = bgc::models::bgc::readMMSInputData(dataDir, defaultInput());
        const bgc::models::tsompsiv::Constants baseConstants = readTSOMPsiVConstants(dataDir);
        const MMS1ManufacturedParameters mms;
        const std::filesystem::path baseOutputDir =
            bgc::models::bgc::resolveMMSOutputDirectory(caseDir, input.outputDir);

        PetscCallAbort(PETSC_COMM_WORLD, bgc::models::bgc::printCpuUsageSummary());
        const std::string caseFilter = argc > 1 ? argv[1] : "";
        PetscPrintf(PETSC_COMM_WORLD,
                    "\nMMS1TSOMPsiV -- nine-case MMS sweep for the TSOMPsiV (Psi,V) model\n"
                    "  Psi(x,t) = A*exp(-lambda*t)*g(x), V(x,t) = f*Psi(x,t)\n"
                    "  A=%.6e lambda=%.6e f=%.6e sc=%.6e\n"
                    "  alpha=%.6e rho=%.6e lambdaC=%.6e lambdaR=%.6e\n"
                    "  input : %s\n"
                    "  output: %s\n",
                    static_cast<PetscReal>(mms.amplitude),
                    static_cast<PetscReal>(mms.lambda),
                    static_cast<PetscReal>(mms.vFraction),
                    static_cast<PetscReal>(1.0 - mms.vFraction),
                    static_cast<PetscReal>(baseConstants.alpha),
                    static_cast<PetscReal>(baseConstants.rho),
                    static_cast<PetscReal>(baseConstants.lambdaC),
                    static_cast<PetscReal>(baseConstants.lambdaR),
                    dataDir.string().c_str(),
                    baseOutputDir.string().c_str());

        for (const MMS1Case& testCase : makeMMS1Cases()) {
            if (!caseFilter.empty() && testCase.name != caseFilter) {
                continue;
            }

            const bgc::models::tsompsiv::Constants constants =
                constantsForCase(baseConstants, mms, testCase);
            if (!bgc::ModelTraits<bgc::models::tsompsiv::Tag>::constantsAreValid(constants)) {
                throw std::runtime_error("Invalid TSOMPsiV constants for MMS1 case " +
                                         testCase.name + ".");
            }

            const bgc::models::tsompsiv::DiffusionCoefficients d =
                bgc::models::tsompsiv::computeDiffusionCoefficients(constants);
            const std::string vBoundaryText {
                bgc::models::tsompsiv::toString(constants.vBoundary.type)
            };
            const std::filesystem::path outputDir = baseOutputDir / testCase.name;

            PetscPrintf(PETSC_COMM_WORLD,
                        "  case %-31s theta=%.6e V boundary=%s profile=%s "
                        "d=(%.6e, %.6e, %.6e, %.6e)\n",
                        testCase.name.c_str(),
                        static_cast<PetscReal>(constants.theta),
                        vBoundaryText.c_str(),
                        std::string(profileName(testCase.profile)).c_str(),
                        static_cast<PetscReal>(d.d11),
                        static_cast<PetscReal>(d.d12),
                        static_cast<PetscReal>(d.d21),
                        static_cast<PetscReal>(d.d22));

            runCases(input, mms, testCase, constants, outputDir);
        }
    } catch (const std::exception& ex) {
        PetscPrintf(PETSC_COMM_WORLD, "\nERROR: %s\n", ex.what());
        exitCode = 1;
    }

    PetscCallAbort(PETSC_COMM_WORLD, PetscFinalize());
    return exitCode;
}
