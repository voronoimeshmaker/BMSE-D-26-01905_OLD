#include "../../Common/MMSCommon.hpp"

#include <petsc.h>

#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <vector>

namespace {

bgc::SimConfig makeConfig(const paper_mms::InputData& input,
                          const PetscReal bv,
                          const PetscInt nx) {
    bgc::SimConfig cfg;
    paper_mms::configureStandardSimulation(input, cfg, bgc::Model::TBGC, bv, nx);

    cfg.bcWest[0] = bgc::BoundaryCondition::neumann(0.0);
    cfg.bcWest[1] = bgc::BoundaryCondition::dirichlet(0.0);
    cfg.bcEast[0] = bgc::BoundaryCondition::neumann(0.0);
    cfg.bcEast[1] = bgc::BoundaryCondition::dirichlet(0.0);
    return cfg;
}

PetscReal phiFn(const bgc::SimConfig&, const PetscReal x, const PetscReal t) {
    const PetscReal oneMinusX = 1.0 - x;
    return PetscExpReal(-t) * x * x * oneMinusX * oneMinusX;
}

PetscReal phiXX(const bgc::SimConfig&, const PetscReal x, const PetscReal t) {
    return PetscExpReal(-t) * (12.0 * x * x - 12.0 * x + 2.0);
}

PetscReal muFn(const bgc::SimConfig& cfg, const PetscReal x, const PetscReal t) {
    return phiFn(cfg, x, t) - cfg.bv * phiXX(cfg, x, t);
}

PetscReal sourceFn(const bgc::SimConfig& cfg, const PetscReal x, const PetscReal t) {
    const PetscReal x2 = x * x;
    const PetscReal x3 = x2 * x;
    const PetscReal x4 = x2 * x2;
    return PetscExpReal(-t) *
           (24.0 * cfg.bv - x4 + 2.0 * x3 - 13.0 * x2 + 12.0 * x - 2.0);
}

PetscReal phiX(const PetscReal x) {
    return 2.0 * x * (1.0 - x) * (1.0 - 2.0 * x);
}

PetscReal phiXXX(const PetscReal x) {
    return 12.0 * (2.0 * x - 1.0);
}

PetscErrorCode runMMSCampaign(const paper_mms::InputData& input,
                              const PetscReal bv,
                              std::vector<paper_mms::ErrorRecord>& records) {
    PetscFunctionBeginUser;

    PetscPrintf(PETSC_COMM_WORLD,
                "\n======================================================================\n"
                "  MMS campaign for TBGC  (MMS1, Bv = %.16e)\n"
                "======================================================================\n",
                bv);

    for (const PetscInt nx : input.nxList) {
        const bgc::SimConfig cfg = makeConfig(input, bv, nx);
        const auto thermo = paper_mms::analyseTBGCMixedCriterion(cfg, phiX, phiXXX);
        paper_mms::printThermoSummary("MMS1", cfg, thermo);

        bgc::SimState st;
        PetscCall(bgc::createStateTBGC(cfg, st));

        const auto coeff = bgc::computeCoefficientsTBGC(cfg);
        PetscCall(bgc::assembleMatrixTBGC(cfg, st, coeff));
        PetscCall(bgc::configureSolver(cfg, st));
        PetscCall(paper_mms::runTBGCTransientLoop(cfg, st, phiFn, muFn, sourceFn));

        bgc::ErrorNorms norms;
        PetscCall(bgc::computeErrorNorms(cfg, st, cfg.tf, phiFn, norms));
        paper_mms::appendRecord(cfg, norms, records);
        paper_mms::printSimulationSummary(cfg, norms);

        PetscCall(bgc::destroyState(st));
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace

int main(int argc, char** argv) {
    PetscCallAbort(PETSC_COMM_WORLD,
                   PetscInitialize(&argc,
                                   &argv,
                                   nullptr,
                                   "MMS1TBGC -- homogeneous manufactured solution for the TBGC model"));

    PetscInt exitCode = 0;
    try {
        const std::filesystem::path dataDir = paper_mms::findDataDirectory();
        const auto input = paper_mms::readInputData(dataDir);

        PetscPrintf(PETSC_COMM_WORLD,
                    "\n######################################################################\n"
                    "  MMS1TBGC -- TBGC MMS with homogeneous boundary conditions\n"
                    "  phi(x,t) = exp(-t) * x^2 * (1 - x)^2\n"
                    "  mu(x,t)  = phi(x,t) - Bv * phi_xx(x,t)\n"
                    "  Input directory: %s\n"
                    "######################################################################\n",
                    dataDir.string().c_str());

        for (const PetscReal bv : input.bvList) {
            std::vector<paper_mms::ErrorRecord> records;
            PetscCallAbort(PETSC_COMM_WORLD, runMMSCampaign(input, bv, records));
            paper_mms::printConvergenceTable("TBGC", bv, records);
        }
    } catch (const std::exception& ex) {
        PetscPrintf(PETSC_COMM_WORLD, "\nERROR: %s\n", ex.what());
        exitCode = 1;
    }

    PetscCallAbort(PETSC_COMM_WORLD, PetscFinalize());
    return exitCode;
}
