#include <bgclib/Solver.hpp>
#include <bgclib/Misc/Types.hpp>
#include <bgclib/SimConfig.hpp>
#include <bgclib/SimState.hpp>

#include <iostream>

namespace bgc {

static const char* directSolverBackendLabel(
    DirectSolverBackend backend) {
    switch (backend) {
        case DirectSolverBackend::PetscDefault:
            return "PETSc default";
        case DirectSolverBackend::Mumps:
            return "MUMPS";
    }

    return "Unknown";
}

static PetscErrorCode configureDirectLUBackend(
    const SimConfig& cfg,
    PC               pc) {
    PetscFunctionBeginUser;

    PetscCall(PCSetType(pc, PCLU));

    switch (cfg.directSolverBackend) {
        case DirectSolverBackend::PetscDefault:
            break;

        case DirectSolverBackend::Mumps:
#if defined(PETSC_HAVE_MUMPS)
            PetscCall(
                PCFactorSetMatSolverType(pc, MATSOLVERMUMPS));
#else
            PetscCheck(false,
                       PETSC_COMM_WORLD,
                       PETSC_ERR_SUP,
                       "SimConfig pede MUMPS, mas o PETSc foi "
                       "compilado sem suporte a MUMPS.");
#endif
            break;
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode configureIterativeSubKSPs(
    const SimConfig& cfg,
    PC               pc) {
    PetscFunctionBeginUser;

    PetscInt nSplits = 0;
    KSP* subKsps = nullptr;
    PetscCall(PCFieldSplitGetSubKSP(pc, &nSplits, &subKsps));

    for (PetscInt k = 0; k < nSplits; ++k) {
        PC subPc = nullptr;

        PetscCall(KSPSetType(subKsps[k], KSPGMRES));
        PetscCall(KSPSetTolerances(subKsps[k],
                                   cfg.tole,
                                   PETSC_DEFAULT,
                                   PETSC_DEFAULT,
                                   500));
        PetscCall(KSPGetPC(subKsps[k], &subPc));
        PetscCall(PCSetType(subPc, PCILU));
    }

    PetscCall(PetscFree(subKsps));
    PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode configureSolverScalarIterative(
    const SimConfig& cfg,
    SimState&        st) {
    PetscFunctionBeginUser;

    PC pc = nullptr;
    PetscCall(KSPGetPC(st.ksp, &pc));

    PetscCall(KSPSetType(st.ksp, KSPGMRES));
    PetscCall(KSPSetTolerances(st.ksp,
                               cfg.tole,
                               PETSC_DEFAULT,
                               PETSC_DEFAULT,
                               PETSC_DEFAULT));
    PetscCall(PCSetType(pc, PCILU));

    if (cfg.verbose) {
        PetscPrintf(PETSC_COMM_WORLD,
                    "[solver] %s: GMRES + ILU\n",
                    cfg.modelName().c_str());
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode configureSolverBlockIterative(
    const SimConfig& cfg,
    SimState&        st) {
    PetscFunctionBeginUser;

    PetscCheck(st.is_phi && st.is_mu,
               PETSC_COMM_WORLD,
               PETSC_ERR_ARG_NULL,
               "Solver iterativo em bloco requer is_phi e is_mu validos.");

    PC pc = nullptr;
    PetscCall(KSPGetPC(st.ksp, &pc));

    PetscCall(KSPSetType(st.ksp, KSPFGMRES));
    PetscCall(KSPSetTolerances(st.ksp,
                               cfg.tole,
                               PETSC_DEFAULT,
                               PETSC_DEFAULT,
                               PETSC_DEFAULT));

    PetscCall(PCSetType(pc, PCFIELDSPLIT));
    PetscCall(PCFieldSplitSetIS(pc, "phi", st.is_phi));
    PetscCall(PCFieldSplitSetIS(pc, "mu", st.is_mu));
    PetscCall(PCFieldSplitSetType(pc, PC_COMPOSITE_SCHUR));
    PetscCall(PCFieldSplitSetSchurFactType(
        pc, PC_FIELDSPLIT_SCHUR_FACT_UPPER));
    PetscCall(PCFieldSplitSetSchurPre(
        pc, PC_FIELDSPLIT_SCHUR_PRE_A11, nullptr));

    PetscCall(KSPSetUp(st.ksp));
    PetscCall(configureIterativeSubKSPs(cfg, pc));

    if (cfg.verbose) {
        PetscPrintf(
            PETSC_COMM_WORLD,
            "[solver] %s: FGMRES + FIELDSPLIT Schur "
            "(sub-KSPs: GMRES + ILU)\n",
            cfg.modelName().c_str());
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode configureSolverDirect(
    const SimConfig& cfg,
    SimState&        st) {
    PetscFunctionBeginUser;

    PC pc = nullptr;
    PetscCall(KSPGetPC(st.ksp, &pc));

    PetscCall(KSPSetType(st.ksp, KSPPREONLY));
    PetscCall(configureDirectLUBackend(cfg, pc));

    if (cfg.verbose) {
        PetscPrintf(
            PETSC_COMM_WORLD,
            "[solver] %s: PREONLY + LU + %s\n",
            cfg.modelName().c_str(),
            directSolverBackendLabel(
                cfg.directSolverBackend));
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode configureSolver(const SimConfig& cfg,
                               SimState&        st) {
    PetscFunctionBeginUser;

    if (cfg.useDirectSolver) {
        PetscCall(configureSolverDirect(cfg, st));
    } else {
        const bool useBlockSolver =
            (cfg.model == Model::TBGC || cfg.model == Model::TSPV) ||
            (st.is_phi && st.is_mu);

        if (useBlockSolver) {
            PetscCall(configureSolverBlockIterative(cfg, st));
        } else {
            PetscCall(configureSolverScalarIterative(cfg, st));
        }
    }

    PetscCall(KSPSetFromOptions(st.ksp));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode solveLinearSystem(const SimConfig& cfg,
                                 SimState&        st) {
    PetscFunctionBeginUser;

    // Fechar as vistas abertas de b e x ANTES do solve.
    // Se VecGetSubVector retornou cópias (IS geral), é necessário
    // VecRestoreSubVector para sincronizar b1/b2 → b e liberar phi/mu de x
    // antes que KSPSolve modifique x.  Para vistas reais (VECNEST),
    // este par Restore/Get é inofensivo.
    if (st.is_phi && st.is_mu) {
        PetscCall(VecRestoreSubVector(st.b, st.is_phi, &st.b1));
        PetscCall(VecRestoreSubVector(st.b, st.is_mu,  &st.b2));
        PetscCall(VecRestoreSubVector(st.x, st.is_phi, &st.phi));
        PetscCall(VecRestoreSubVector(st.x, st.is_mu,  &st.mu));
    }

    PetscCall(KSPSolve(st.ksp, st.b, st.x));

   

    if (cfg.verbose) {
    PetscPrintf(PETSC_COMM_WORLD,
                "\n============================================================\n");
    PetscPrintf(PETSC_COMM_WORLD,
                "Sistema linear TBGCS\n");
    PetscPrintf(PETSC_COMM_WORLD,
                "nx = %" PetscInt_FMT "\n",
                cfg.nx);
    PetscPrintf(PETSC_COMM_WORLD,
                "h  = %.16e\n",
                static_cast<double>(cfg.h));
    PetscPrintf(PETSC_COMM_WORLD,
                "dt = %.16e\n",
                static_cast<double>(cfg.dt));
    PetscPrintf(PETSC_COMM_WORLD,
                "Bv = %.16e\n",
                static_cast<double>(cfg.bv));
    PetscPrintf(PETSC_COMM_WORLD,
                "============================================================\n\n");

    Mat Amono = nullptr;

    PetscCall(MatConvert(st.A, MATAIJ, MAT_INITIAL_MATRIX, &Amono));

    PetscViewer viewer = PETSC_VIEWER_STDOUT_WORLD;

    PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB));

    PetscPrintf(PETSC_COMM_WORLD,
                "\n-------------------- Matriz global A --------------------\n\n");

    PetscCall(MatView(Amono, viewer));

    PetscPrintf(PETSC_COMM_WORLD,
                "\n-------------------- Vetor global b --------------------\n\n");

    PetscCall(VecView(st.b, viewer));

    PetscCall(PetscViewerPopFormat(viewer));

    PetscCall(MatDestroy(&Amono));

    PetscPrintf(PETSC_COMM_WORLD,
                "\n============================================================\n");
    PetscPrintf(PETSC_COMM_WORLD,
                "Fim da impressao do sistema linear\n");
    PetscPrintf(PETSC_COMM_WORLD,
                "============================================================\n\n");

            PetscCall(PetscFinalize());
        std::exit(EXIT_SUCCESS);
    }

    KSPConvergedReason reason;
    PetscInt its = 0;
    PetscReal rnorm = 0.0;

    PetscCall(KSPGetConvergedReason(st.ksp, &reason));
    PetscCall(KSPGetIterationNumber(st.ksp, &its));
    PetscCall(KSPGetResidualNorm(st.ksp, &rnorm));

    if (cfg.verbose || reason < 0) {
        PetscPrintf(PETSC_COMM_WORLD,
                    "[solver] reason=%d  its=%" PetscInt_FMT
                    "  ||r||=%.16e\n",
                    static_cast<int>(reason),
                    its,
                    rnorm);
    }

    PetscCheck(reason >= 0,
               PETSC_COMM_WORLD,
               PETSC_ERR_NOT_CONVERGED,
               "O solver linear nao convergiu. reason=%d  "
               "its=%" PetscInt_FMT "  ||r||=%.16e",
               static_cast<int>(reason),
               its,
               rnorm);

    // Obter as sub-vistas de x (agora com a solução nova) e
    // reabrir b1/b2 para o próximo passo de montagem.
    if (st.is_phi && st.is_mu) {
        PetscCall(VecGetSubVector(st.x, st.is_phi, &st.phi));
        PetscCall(VecGetSubVector(st.x, st.is_mu,  &st.mu));
        PetscCall(VecGetSubVector(st.b, st.is_phi, &st.b1));
        PetscCall(VecGetSubVector(st.b, st.is_mu,  &st.b2));
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace bgc