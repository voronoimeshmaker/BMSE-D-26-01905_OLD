#include <petsc.h>

#include <bgclib/BGCLib.hpp>

#include <filesystem>

namespace {

constexpr PetscReal kAlpha = 2.0;
constexpr PetscReal kXiC = 1.0 / 3.0;

std::filesystem::path caseDirectory() {
    return bgc::caseDirectoryFromSource(__FILE__);
}

PetscReal spatialArg(const PetscReal x) {
    return kAlpha * (x - kXiC);
}

PetscReal phiWestBC(const PetscReal t) {
    return PetscExpReal(-t) * PetscCoshReal(kAlpha * kXiC);
}

PetscReal phiEastBC(const PetscReal t) {
    return PetscExpReal(-t) * PetscCoshReal(kAlpha * (1.0 - kXiC));
}

PetscReal dPhiWestBC(const PetscReal t) {
    return -kAlpha * PetscExpReal(-t) * PetscSinhReal(kAlpha * kXiC);
}

PetscReal dPhiEastBC(const PetscReal t) {
    return kAlpha * PetscExpReal(-t) * PetscSinhReal(kAlpha * (1.0 - kXiC));
}

bgc::BoundarySet makeMMS2Boundaries() {
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

PetscReal mms2Phi(const bgc::models::bgc::Constants&, const PetscReal x, const PetscReal t) {
    return PetscExpReal(-t) * PetscCoshReal(spatialArg(x));
}

PetscReal mms2Source(const bgc::models::bgc::Constants& constants,
                     const PetscReal x,
                     const PetscReal t) {
    return (16.0 * constants.bv - 5.0) * mms2Phi(constants, x, t);
}

bgc::models::bgc::MMSCaseSpec makeSpec() {
    return {
        .caseName = "MMS2BGC",
        .description = "BGC MMS with non-homogeneous boundary conditions",
        .fieldFormula = "exp(-t) * cosh(2 * (x - 1/3))",
        .sourceFormula = "(16*Bv - 5) * phi(x,t)",
        .files = {
            .setupCsv = "mms2bgc_setup.csv",
            .convergenceCsv = "mms2bgc_convergence.csv",
            .statusTxt = "mms2bgc_full_run_status.txt",
            .fieldsPrefix = "mms2bgc_fields_N",
            .truncationPrefix = "mms2bgc_lte_N",
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
                                   "MMS2BGC -- non-homogeneous manufactured solution for BGC"));

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
            &mms2Phi,
            &mms2Source,
            &makeMMS2Boundaries,
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
