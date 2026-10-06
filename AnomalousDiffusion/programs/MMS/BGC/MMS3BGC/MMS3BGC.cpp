#include <petsc.h>

#include <bgclib/BGCLib.hpp>

#include <filesystem>

namespace {

constexpr PetscReal kAlpha = 2.0;
constexpr PetscReal kXiC = 1.0 / 3.0;
constexpr PetscReal kEps = 0.2;
constexpr PetscReal kPi = std::numbers::pi_v<PetscReal>;

std::filesystem::path caseDirectory() {
    return bgc::caseDirectoryFromSource(__FILE__);
}

PetscReal spatialArg(const PetscReal x) {
    return kAlpha * (x - kXiC);
}

PetscReal spatialProfile(const PetscReal x) {
    return PetscCoshReal(spatialArg(x)) + kEps * PetscSinReal(kPi * x);
}

PetscReal spatialSecondDerivative(const PetscReal x) {
    const PetscReal pi2 = kPi * kPi;
    return 4.0 * PetscCoshReal(spatialArg(x)) - kEps * pi2 * PetscSinReal(kPi * x);
}

PetscReal spatialFourthDerivative(const PetscReal x) {
    const PetscReal pi2 = kPi * kPi;
    const PetscReal pi4 = pi2 * pi2;
    return 16.0 * PetscCoshReal(spatialArg(x)) + kEps * pi4 * PetscSinReal(kPi * x);
}

PetscReal phiWestBC(const PetscReal t) {
    return PetscExpReal(-t) * PetscCoshReal(kAlpha * kXiC);
}

PetscReal phiEastBC(const PetscReal t) {
    return PetscExpReal(-t) * PetscCoshReal(kAlpha * (1.0 - kXiC));
}

PetscReal dPhiWestBC(const PetscReal t) {
    return PetscExpReal(-t) * (-kAlpha * PetscSinhReal(kAlpha * kXiC) + kEps * kPi);
}

PetscReal dPhiEastBC(const PetscReal t) {
    return PetscExpReal(-t) *
           (kAlpha * PetscSinhReal(kAlpha * (1.0 - kXiC)) - kEps * kPi);
}

bgc::BoundarySet makeMMS3Boundaries() {
    return {
        .west = {
            .conditions = {
                bgc::BoundaryCondition::neumann(dPhiWestBC),
                bgc::BoundaryCondition::dirichlet(phiWestBC),
            },
        },
        .east = {
            .conditions = {
                bgc::BoundaryCondition::neumann(dPhiEastBC),
                bgc::BoundaryCondition::dirichlet(phiEastBC),
            },
        },
    };
}

PetscReal mms3Phi(const bgc::models::bgc::Constants&, const PetscReal x, const PetscReal t) {
    return PetscExpReal(-t) * spatialProfile(x);
}

PetscReal mms3Source(const bgc::models::bgc::Constants& constants,
                     const PetscReal x,
                     const PetscReal t) {
    return PetscExpReal(-t) *
           (-spatialProfile(x) - spatialSecondDerivative(x) +
            constants.bv * spatialFourthDerivative(x));
}

bgc::models::bgc::MMSCaseSpec makeSpec() {
    return {
        .caseName = "MMS3BGC",
        .description = "BGC MMS with mixed hyperbolic-trigonometric profile",
        .fieldFormula = "exp(-t) * [cosh(2 * (x - 1/3)) + 0.2 * sin(pi * x)]",
        .sourceFormula = "exp(-t) * (-phi_x0 - phi_xx0 + Bv*phi_xxxx0)",
        .files = {
            .setupCsv = "mms3bgc_setup.csv",
            .convergenceCsv = "mms3bgc_convergence.csv",
            .statusTxt = "mms3bgc_full_run_status.txt",
            .fieldsPrefix = "mms3bgc_fields_N",
            .truncationPrefix = "mms3bgc_lte_N",
        },
    };
}

bgc::models::bgc::MMSInputData defaultInput() {
    return {
        .tf = 1.0e-3,
        .cdt = 1.024e-1,
        .nxList = {8},
        .bvList = {1.0e-2},
        .outputDir = "Saida",
        .verbose = true,
        .debug = false,
    };
}

} // namespace

int main(int argc, char** argv) {
    PetscCallAbort(PETSC_COMM_WORLD,
                   PetscInitialize(&argc,
                                   &argv,
                                   nullptr,
                                   "MMS3BGC -- mixed hyperbolic-trigonometric manufactured solution"));

    PetscInt exitCode = 0;
    try {
        const std::filesystem::path caseDir = caseDirectory();
        const std::filesystem::path dataDir = caseDir / "Dados";
        const auto input = bgc::models::bgc::readMMSInputData(dataDir, defaultInput());
        const std::filesystem::path outputDir =
            bgc::models::bgc::resolveMMSOutputDirectory(caseDir, input.outputDir);
        const bgc::models::bgc::MMSRunner runner {
            makeSpec(),
            input,
            outputDir,
            &mms3Phi,
            &mms3Source,
            &makeMMS3Boundaries,
        };

        PetscCallAbort(PETSC_COMM_WORLD, runner.printSummary(dataDir));
        runner.run();
    } catch (const std::exception& ex) {
        PetscPrintf(PETSC_COMM_WORLD, "\nERROR: %s\n", ex.what());
        exitCode = 1;
    }

    PetscCallAbort(PETSC_COMM_WORLD, PetscFinalize());
    return exitCode;
}
