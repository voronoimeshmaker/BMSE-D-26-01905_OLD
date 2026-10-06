#include <petsc.h>

#include <bgclib/BGCLib.hpp>

#include <filesystem>

namespace {

bgc::BoundarySet makeMMS1Boundaries() {
    return {
        .west = {
            .conditions = {
                bgc::BoundaryCondition::neumann(0.0),
                bgc::BoundaryCondition::dirichlet(0.0),
            },
        },
        .east = {
            .conditions = {
                bgc::BoundaryCondition::neumann(0.0),
                bgc::BoundaryCondition::dirichlet(0.0),
            },
        },
    };
}

PetscReal mms1Phi(const bgc::models::bgc::Constants&, const PetscReal x, const PetscReal t) {
    const PetscReal oneMinusX = 1.0 - x;
    return PetscExpReal(-t) * x * x * oneMinusX * oneMinusX;
}

PetscReal mms1Source(const bgc::models::bgc::Constants& constants,
                     const PetscReal x,
                     const PetscReal t) {
    const PetscReal x2 = x * x;
    const PetscReal x3 = x2 * x;
    const PetscReal x4 = x2 * x2;
    return PetscExpReal(-t) *
           (24.0 * constants.bv - x4 + 2.0 * x3 - 13.0 * x2 + 12.0 * x - 2.0);
}

bgc::models::bgc::MMSCaseSpec makeSpec() {
    return {
        .caseName = "MMS1BGC",
        .description = "BGC MMS with homogeneous boundary conditions",
        .fieldFormula = "exp(-t) * x^2 * (1 - x)^2",
        .sourceFormula = "exp(-t) * (24*Bv - x^4 + 2*x^3 - 13*x^2 + 12*x - 2)",
        .files = {
            .setupCsv = "mms1bgc_setup.csv",
            .convergenceCsv = "mms1bgc_convergence.csv",
            .statusTxt = "mms1bgc_full_run_status.txt",
            .fieldsPrefix = "mms1bgc_fields_N",
            .truncationPrefix = "mms1bgc_lte_N",
        },
        .writeStatusFile = true,
        .writeSetupSamples = true,
    };
}

bgc::models::bgc::MMSInputData defaultInput() {
    return {
        .tf = 1.0e-3,
        .cdt = 1.024e-1,
        .nxList = {8, 16, 32, 64, 128, 256, 512},
        .bvList = {1.0e-2, 5.0e-1},
        .outputDir = "Saida",
        .verbose = true,
        .debug = false,
        .debugPrintMaxNx = 16,
    };
}

} // namespace

int main(int argc, char** argv) {
    PetscCallAbort(PETSC_COMM_WORLD,
                   PetscInitialize(&argc,
                                   &argv,
                                   nullptr,
                                   "MMS1BGC -- homogeneous manufactured solution for the BGC model"));

    PetscInt exitCode = 0;
    try {
        const std::filesystem::path dataDir = bgc::models::bgc::findMMSDataDirectory();
        const auto input = bgc::models::bgc::readMMSInputData(dataDir, defaultInput());
        const std::filesystem::path caseDir = dataDir.parent_path();
        const std::filesystem::path outputDir =
            bgc::models::bgc::resolveMMSOutputDirectory(caseDir, input.outputDir);
        const bgc::models::bgc::MMSRunner runner {
            makeSpec(),
            input,
            outputDir,
            &mms1Phi,
            &mms1Source,
            &makeMMS1Boundaries,
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
