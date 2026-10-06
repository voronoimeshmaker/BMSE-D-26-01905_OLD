#include "../../Common/MMSCommon.hpp"

#include <petsc.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <vector>

namespace {

constexpr PetscReal kAlpha = 2.0;
constexpr PetscReal kXiC = 1.0 / 3.0;

PetscReal spatialArg(const PetscReal x) {
    return kAlpha * (x - kXiC);
}

PetscReal phiWestBC(const PetscReal t) {
    return PetscExpReal(-t) * std::cosh(kAlpha * kXiC);
}

PetscReal phiEastBC(const PetscReal t) {
    return PetscExpReal(-t) * std::cosh(kAlpha * (1.0 - kXiC));
}

PetscReal dPhiWestBC(const PetscReal t) {
    return -kAlpha * PetscExpReal(-t) * std::sinh(kAlpha * kXiC);
}

PetscReal dPhiEastBC(const PetscReal t) {
    return kAlpha * PetscExpReal(-t) * std::sinh(kAlpha * (1.0 - kXiC));
}

bgc::SimConfig makeConfig(const paper_mms::InputData& input,
                          const PetscReal bv,
                          const PetscInt nx) {
    bgc::SimConfig cfg;
    paper_mms::configureStandardSimulation(input, cfg, bgc::Model::TBGC, bv, nx);

    cfg.bcWest[0] = bgc::BoundaryCondition::neumann(dPhiWestBC);
    cfg.bcWest[1] = bgc::BoundaryCondition::dirichlet(phiWestBC);
    cfg.bcEast[0] = bgc::BoundaryCondition::neumann(dPhiEastBC);
    cfg.bcEast[1] = bgc::BoundaryCondition::dirichlet(phiEastBC);
    return cfg;
}

PetscReal phiFn(const bgc::SimConfig&, const PetscReal x, const PetscReal t) {
    return PetscExpReal(-t) * std::cosh(spatialArg(x));
}

PetscReal muFn(const bgc::SimConfig& cfg, const PetscReal x, const PetscReal t) {
    return (1.0 - 4.0 * cfg.bv) * phiFn(cfg, x, t);
}

PetscReal sourceFn(const bgc::SimConfig& cfg, const PetscReal x, const PetscReal t) {
    return (16.0 * cfg.bv - 5.0) * phiFn(cfg, x, t);
}

PetscReal phiX(const PetscReal x) {
    return kAlpha * std::sinh(spatialArg(x));
}

PetscReal phiXXX(const PetscReal x) {
    return kAlpha * kAlpha * kAlpha * std::sinh(spatialArg(x));
}

std::string makeProfileFilename(const char* modelName,
                                const PetscReal bv,
                                const PetscInt nx) {
    char buffer[256];
    PetscSNPrintf(buffer,
                  sizeof(buffer),
                  "MMS2_%s_phi_tf_bv_%0.6f_nx_%" PetscInt_FMT ".dat",
                  modelName,
                  bv,
                  nx);
    return std::string(buffer);
}

PetscErrorCode writePhiProfileAtFinalTime(const char* modelName,
                                          const bgc::SimConfig& cfg,
                                          const bgc::SimState& st) {
    PetscFunctionBeginUser;

    const PetscScalar* phiValues = nullptr;
    PetscCall(VecGetArrayRead(st.phi, &phiValues));

    const std::filesystem::path outPath =
        std::filesystem::current_path() / makeProfileFilename(modelName, cfg.bv, cfg.nx);

    std::ofstream out(outPath);
    if (!out.is_open()) {
        PetscCall(VecRestoreArrayRead(st.phi, &phiValues));
        throw std::runtime_error("Could not open output file for phi(x,tf) profile: " +
                                 outPath.string());
    }

    out << std::scientific << std::setprecision(16);
    out << "# x phi(x,tf)\n";
    out << 0.0 << " " << phiWestBC(cfg.tf) << "\n";

    for (PetscInt i = 0; i < cfg.nx; ++i) {
        const PetscReal x = (static_cast<PetscReal>(i) + 0.5) * cfg.h;
        out << x << " " << PetscRealPart(phiValues[i]) << "\n";
    }

    out << 1.0 << " " << phiEastBC(cfg.tf) << "\n";

    PetscCall(VecRestoreArrayRead(st.phi, &phiValues));

    PetscPrintf(PETSC_COMM_WORLD,
                "[profile] phi(x,tf) written to %s\n",
                outPath.string().c_str());

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode runMMSCampaign(const paper_mms::InputData& input,
                              const PetscReal bv,
                              std::vector<paper_mms::ErrorRecord>& records) {
    PetscFunctionBeginUser;

    PetscPrintf(PETSC_COMM_WORLD,
                "\n======================================================================\n"
                "  MMS campaign for TBGC  (MMS2, Bv = %.16e)\n"
                "======================================================================\n",
                bv);

    for (const PetscInt nx : input.nxList) {
        const bgc::SimConfig cfg = makeConfig(input, bv, nx);
        const auto thermo = paper_mms::analyseTBGCMixedCriterion(cfg, phiX, phiXXX);
        paper_mms::printThermoSummary("MMS2", cfg, thermo);

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
        PetscCall(writePhiProfileAtFinalTime("TBGC", cfg, st));

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
                                   "MMS2TBGC -- non-homogeneous manufactured solution for the TBGC model"));

    PetscInt exitCode = 0;
    try {
        const std::filesystem::path dataDir = paper_mms::findDataDirectory();
        const auto input = paper_mms::readInputData(dataDir);

        PetscPrintf(PETSC_COMM_WORLD,
                    "\n######################################################################\n"
                    "  MMS2TBGC -- TBGC MMS with non-homogeneous boundary conditions\n"
                    "  phi(x,t) = exp(-t) * cosh(2 * (x - 1/3))\n"
                    "  mu(x,t)  = (1 - 4 * Bv) * phi(x,t)\n"
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
