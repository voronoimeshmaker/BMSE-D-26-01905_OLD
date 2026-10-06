#include <bgclib/SimState.hpp>
#include <bgclib/Misc/Types.hpp>
#include <bgclib/SimConfig.hpp>

namespace bgc {

PetscErrorCode createStateBGC(const SimConfig& cfg,
                              SimState&        st) {
    PetscFunctionBeginUser;

    if (cfg.verbose) {  
        PetscPrintf(PETSC_COMM_WORLD, "%s", separator().c_str());
        PetscPrintf(PETSC_COMM_WORLD,
                "Alocando sistema BGC  (nx=%" PetscInt_FMT ")\n",
                cfg.nx);
        PetscPrintf(PETSC_COMM_WORLD, "%s", separator().c_str());
    }

    PetscCall(DMDACreate1d(PETSC_COMM_WORLD,
                           DM_BOUNDARY_NONE,
                           cfg.nx,
                           1,
                           SimState::stencilWidth,
                           nullptr,
                           &st.dm));
    PetscCall(DMSetFromOptions(st.dm));
    PetscCall(DMSetUp(st.dm));

    PetscCall(DMCreateGlobalVector(st.dm, &st.b));
    PetscCall(DMCreateGlobalVector(st.dm, &st.phi));
    PetscCall(DMCreateGlobalVector(st.dm, &st.phi_a));
    PetscCall(DMCreateGlobalVector(st.dm, &st.phi_0));
    PetscCall(DMCreateGlobalVector(st.dm, &st.b1Source));
    PetscCall(DMCreateGlobalVector(st.dm, &st.source));

    PetscCall(DMCreateMatrix(st.dm, &st.A));

    PetscCall(KSPCreate(PETSC_COMM_WORLD, &st.ksp));
    PetscCall(KSPSetOperators(st.ksp, st.A, st.A));

    PetscPrintf(PETSC_COMM_WORLD, "Sistema BGC alocado com sucesso.\n\n");
    PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode createStateTBGC(const SimConfig& cfg,
                               SimState&        st) {
    PetscFunctionBeginUser;

    if (cfg.verbose) {  
        PetscPrintf(PETSC_COMM_WORLD, "%s", separator().c_str());
        PetscPrintf(PETSC_COMM_WORLD,
                "Alocando sistema TBGC  (nx=%" PetscInt_FMT ")\n",
                cfg.nx);
        PetscPrintf(PETSC_COMM_WORLD, "%s", separator().c_str());
    }

    PetscCall(DMDACreate1d(PETSC_COMM_WORLD,
                           DM_BOUNDARY_NONE,
                           cfg.nx,
                           2,
                           1,
                           nullptr,
                           &st.dm));
    PetscCall(DMSetFromOptions(st.dm));
    PetscCall(DMSetUp(st.dm));
    PetscCall(DMDASetFieldName(st.dm, 0, "phi"));
    PetscCall(DMDASetFieldName(st.dm, 1, "mu"));

    DM dmPhi = nullptr;
    PetscCall(DMDACreateCompatibleDMDA(st.dm, 1, &dmPhi));

    DMDALocalInfo info;
    PetscCall(DMDAGetLocalInfo(dmPhi, &info));
    const PetscInt localSize = info.xm;
    const PetscInt start = info.xs;

    PetscInt* idxPhi = nullptr;
    PetscInt* idxMu = nullptr;
    PetscCall(PetscMalloc1(localSize, &idxPhi));
    PetscCall(PetscMalloc1(localSize, &idxMu));

    for (PetscInt i = 0; i < localSize; ++i) {
        idxPhi[i] = start + i;
        idxMu[i] = start + i + cfg.nx;
    }

    PetscCall(ISCreateGeneral(PETSC_COMM_WORLD,
                              localSize,
                              idxPhi,
                              PETSC_COPY_VALUES,
                              &st.is_phi));
    PetscCall(ISCreateGeneral(PETSC_COMM_WORLD,
                              localSize,
                              idxMu,
                              PETSC_COPY_VALUES,
                              &st.is_mu));
    PetscCall(PetscFree(idxPhi));
    PetscCall(PetscFree(idxMu));

    auto createBlock = [&](Mat& M) -> PetscErrorCode {
        PetscCall(MatCreate(PETSC_COMM_WORLD, &M));
        PetscCall(MatSetSizes(M, localSize, localSize, cfg.nx, cfg.nx));
        PetscCall(MatSetType(M, MATMPIAIJ));
        PetscCall(MatSetFromOptions(M));
        return PETSC_SUCCESS;
    };

    PetscCall(createBlock(st.A11));
    PetscCall(createBlock(st.A12));
    PetscCall(createBlock(st.A21));
    PetscCall(createBlock(st.A22));

    PetscInt* dA11 = nullptr;
    PetscInt* dA12 = nullptr;
    PetscInt* dA21 = nullptr;
    PetscInt* dA22 = nullptr;
    PetscCall(PetscCalloc1(localSize, &dA11));
    PetscCall(PetscCalloc1(localSize, &dA12));
    PetscCall(PetscCalloc1(localSize, &dA21));
    PetscCall(PetscCalloc1(localSize, &dA22));

    for (PetscInt il = 0; il < localSize; ++il) {
        const PetscInt g = start + il;
        dA11[il] = 1 + (g > 0 ? 1 : 0) + (g < cfg.nx - 1 ? 1 : 0);
        dA12[il] = 1 + (g > 0 ? 1 : 0) + (g < cfg.nx - 1 ? 1 : 0);
        dA21[il] = 1 + (g > 0 ? 1 : 0) + (g > 1 ? 1 : 0)
                 + (g < cfg.nx - 1 ? 1 : 0) + (g < cfg.nx - 2 ? 1 : 0);
        dA22[il] = 1;
    }

    auto prealloc = [&](Mat M, PetscInt* d) -> PetscErrorCode {
        PetscCall(MatMPIAIJSetPreallocation(M, 0, d, 0, nullptr));
        PetscCall(MatSetUp(M));
        PetscCall(MatSetOption(M, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
        return PETSC_SUCCESS;
    };

    PetscCall(prealloc(st.A11, dA11));
    PetscCall(prealloc(st.A12, dA12));
    PetscCall(prealloc(st.A21, dA21));
    PetscCall(prealloc(st.A22, dA22));

    PetscCall(PetscFree(dA11));
    PetscCall(PetscFree(dA12));
    PetscCall(PetscFree(dA21));
    PetscCall(PetscFree(dA22));

    {
        Mat sub[2][2] = {{st.A11, st.A12}, {st.A21, st.A22}};
        IS rows[2] = {st.is_phi, st.is_mu};
        IS cols[2] = {st.is_phi, st.is_mu};

        PetscCall(MatCreateNest(PETSC_COMM_WORLD,
                                2,
                                rows,
                                2,
                                cols,
                                &sub[0][0],
                                &st.A));
        PetscCall(MatSetUp(st.A));
    }

    PetscCall(MatCreateVecs(st.A, &st.x, &st.b));
    PetscCall(VecGetSubVector(st.x, st.is_phi, &st.phi));
    PetscCall(VecGetSubVector(st.x, st.is_mu, &st.mu));
    PetscCall(VecGetSubVector(st.b, st.is_phi, &st.b1));
    PetscCall(VecGetSubVector(st.b, st.is_mu, &st.b2));

    PetscCall(VecDuplicate(st.phi, &st.phi_a));
    PetscCall(VecDuplicate(st.phi, &st.phi_0));
    PetscCall(VecDuplicate(st.mu, &st.mu_a));
    PetscCall(VecDuplicate(st.b1, &st.b1Source));
    PetscCall(DMCreateGlobalVector(st.dm, &st.source));

    PetscCall(KSPCreate(PETSC_COMM_WORLD, &st.ksp));
    PetscCall(KSPSetOperators(st.ksp, st.A, st.A));

    PetscCall(DMDestroy(&dmPhi));

    PetscPrintf(PETSC_COMM_WORLD, "Sistema TBGC alocado com sucesso.\n\n");
    PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode createStateTSUV(const SimConfig& cfg,
                              SimState&        st) {
    PetscFunctionBeginUser;

    if (cfg.verbose) {
        PetscPrintf(PETSC_COMM_WORLD, "%s", separator().c_str());
        PetscPrintf(PETSC_COMM_WORLD,
                    "Alocando sistema TSUV  (nx=%" PetscInt_FMT ")\n",
                    cfg.nx);
        PetscPrintf(PETSC_COMM_WORLD, "%s", separator().c_str());
    }

    PetscCall(DMDACreate1d(PETSC_COMM_WORLD,
                           DM_BOUNDARY_NONE,
                           cfg.nx,
                           2,
                           1,
                           nullptr,
                           &st.dm));
    PetscCall(DMSetFromOptions(st.dm));
    PetscCall(DMSetUp(st.dm));
    PetscCall(DMDASetFieldName(st.dm, 0, "U"));
    PetscCall(DMDASetFieldName(st.dm, 1, "V"));

    DM dmPsi = nullptr;
    PetscCall(DMDACreateCompatibleDMDA(st.dm, 1, &dmPsi));

    DMDALocalInfo info;
    PetscCall(DMDAGetLocalInfo(dmPsi, &info));
    const PetscInt localSize = info.xm;
    const PetscInt start     = info.xs;

    PetscInt* idxPsi = nullptr;
    PetscInt* idxV   = nullptr;
    PetscCall(PetscMalloc1(localSize, &idxPsi));
    PetscCall(PetscMalloc1(localSize, &idxV));

    for (PetscInt i = 0; i < localSize; ++i) {
        idxPsi[i] = start + i;
        idxV[i]   = start + i + cfg.nx;
    }

    PetscCall(ISCreateGeneral(PETSC_COMM_WORLD,
                              localSize,
                              idxPsi,
                              PETSC_COPY_VALUES,
                              &st.is_phi));
    PetscCall(ISCreateGeneral(PETSC_COMM_WORLD,
                              localSize,
                              idxV,
                              PETSC_COPY_VALUES,
                              &st.is_mu));
    PetscCall(PetscFree(idxPsi));
    PetscCall(PetscFree(idxV));

    auto createBlock = [&](Mat& M) -> PetscErrorCode {
        PetscCall(MatCreate(PETSC_COMM_WORLD, &M));
        PetscCall(MatSetSizes(M, localSize, localSize, cfg.nx, cfg.nx));
        PetscCall(MatSetType(M, MATMPIAIJ));
        PetscCall(MatSetFromOptions(M));
        return PETSC_SUCCESS;
    };

    PetscCall(createBlock(st.A11));
    PetscCall(createBlock(st.A12));
    PetscCall(createBlock(st.A21));
    PetscCall(createBlock(st.A22));

    PetscInt* dA11 = nullptr;
    PetscInt* dA12 = nullptr;
    PetscInt* dA21 = nullptr;
    PetscInt* dA22 = nullptr;
    PetscCall(PetscCalloc1(localSize, &dA11));
    PetscCall(PetscCalloc1(localSize, &dA12));
    PetscCall(PetscCalloc1(localSize, &dA21));
    PetscCall(PetscCalloc1(localSize, &dA22));

    for (PetscInt il = 0; il < localSize; ++il) {
        const PetscInt g = start + il;
        dA11[il] = 1 + (g > 0 ? 1 : 0) + (g < cfg.nx - 1 ? 1 : 0);
        dA12[il] = (g == 0 || g == cfg.nx - 1) ? 2 : 3;
        dA21[il] = 1 + (g > 0 ? 1 : 0) + (g < cfg.nx - 1 ? 1 : 0);
        dA22[il] = (g == 0 || g == cfg.nx - 1) ? 2 : 3;
    }

    auto prealloc = [&](Mat M, PetscInt* d) -> PetscErrorCode {
        PetscCall(MatMPIAIJSetPreallocation(M, 0, d, 0, nullptr));
        PetscCall(MatSetUp(M));
        PetscCall(MatSetOption(M, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
        return PETSC_SUCCESS;
    };

    PetscCall(prealloc(st.A11, dA11));
    PetscCall(prealloc(st.A12, dA12));
    PetscCall(prealloc(st.A21, dA21));
    PetscCall(prealloc(st.A22, dA22));

    PetscCall(PetscFree(dA11));
    PetscCall(PetscFree(dA12));
    PetscCall(PetscFree(dA21));
    PetscCall(PetscFree(dA22));

    {
        Mat sub[2][2] = {{st.A11, st.A12}, {st.A21, st.A22}};
        IS rows[2]    = {st.is_phi, st.is_mu};
        IS cols[2]    = {st.is_phi, st.is_mu};

        PetscCall(MatCreateNest(PETSC_COMM_WORLD,
                                2,
                                rows,
                                2,
                                cols,
                                &sub[0][0],
                                &st.A));
        PetscCall(MatSetUp(st.A));
    }

    PetscCall(MatCreateVecs(st.A, &st.x, &st.b));
    PetscCall(VecGetSubVector(st.x, st.is_phi, &st.phi));
    PetscCall(VecGetSubVector(st.x, st.is_mu, &st.mu));
    PetscCall(VecGetSubVector(st.b, st.is_phi, &st.b1));
    PetscCall(VecGetSubVector(st.b, st.is_mu, &st.b2));

    PetscCall(VecDuplicate(st.phi, &st.phi_a));
    PetscCall(VecDuplicate(st.phi, &st.phi_0));
    PetscCall(VecDuplicate(st.mu,  &st.mu_a));
    PetscCall(VecDuplicate(st.mu,  &st.mu_0));
    PetscCall(VecDuplicate(st.b1,  &st.b1Source));
    PetscCall(DMCreateGlobalVector(st.dm, &st.source));

    PetscCall(KSPCreate(PETSC_COMM_WORLD, &st.ksp));
    PetscCall(KSPSetOperators(st.ksp, st.A, st.A));

    PetscCall(DMDestroy(&dmPsi));

    PetscPrintf(PETSC_COMM_WORLD, "Sistema TSUV alocado com sucesso.\n\n");
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode createStateTSPV(const SimConfig& cfg,
                               SimState&        st) {
    return createStateTSUV(cfg, st);
}

PetscErrorCode destroyState(SimState& st) {
    PetscFunctionBeginUser;

    if (st.is_phi && st.is_mu) {
        if (st.phi && st.x) {
            PetscCall(VecRestoreSubVector(st.x, st.is_phi, &st.phi));
            st.phi = nullptr;
        }
        if (st.mu && st.x) {
            PetscCall(VecRestoreSubVector(st.x, st.is_mu, &st.mu));
            st.mu = nullptr;
        }
        if (st.b1 && st.b) {
            PetscCall(VecRestoreSubVector(st.b, st.is_phi, &st.b1));
            st.b1 = nullptr;
        }
        if (st.b2 && st.b) {
            PetscCall(VecRestoreSubVector(st.b, st.is_mu, &st.b2));
            st.b2 = nullptr;
        }
    } else {
        if (st.x == st.phi) {
            st.x = nullptr;
        }
        if (st.phi) {
            PetscCall(VecDestroy(&st.phi));
            st.phi = nullptr;
        }
    }

    auto destroyVec = [](Vec& v) -> PetscErrorCode {
        if (v) {
            PetscCall(VecDestroy(&v));
            v = nullptr;
        }
        return PETSC_SUCCESS;
    };

    PetscCall(destroyVec(st.b1Source));
    PetscCall(destroyVec(st.phi_a));
    PetscCall(destroyVec(st.mu_a));
    PetscCall(destroyVec(st.phi_0));
    PetscCall(destroyVec(st.mu_0));
    PetscCall(destroyVec(st.source));
    PetscCall(destroyVec(st.b));
    PetscCall(destroyVec(st.x));

    if (st.ksp) {
        PetscCall(KSPDestroy(&st.ksp));
        st.ksp = nullptr;
    }
    st.pc = nullptr;

    auto destroyMat = [](Mat& m) -> PetscErrorCode {
        if (m) {
            PetscCall(MatDestroy(&m));
            m = nullptr;
        }
        return PETSC_SUCCESS;
    };

    PetscCall(destroyMat(st.A11));
    PetscCall(destroyMat(st.A12));
    PetscCall(destroyMat(st.A21));
    PetscCall(destroyMat(st.A22));
    PetscCall(destroyMat(st.A));

    if (st.is_phi) {
        PetscCall(ISDestroy(&st.is_phi));
        st.is_phi = nullptr;
    }
    if (st.is_mu) {
        PetscCall(ISDestroy(&st.is_mu));
        st.is_mu = nullptr;
    }

    if (st.dm) {
        PetscCall(DMDestroy(&st.dm));
        st.dm = nullptr;
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}



PetscErrorCode createStateTBGCS(const SimConfig& cfg,
                                SimState&        st) {
    PetscFunctionBeginUser;

    if (cfg.verbose) {
        PetscPrintf(PETSC_COMM_WORLD, "%s", separator().c_str());
        PetscPrintf(PETSC_COMM_WORLD,
                    "Alocando sistema TBGCS  (nx=%" PetscInt_FMT ")\n",
                    cfg.nx);
        PetscPrintf(PETSC_COMM_WORLD, "%s", separator().c_str());
    }

    PetscCall(DMDACreate1d(PETSC_COMM_WORLD,
                           DM_BOUNDARY_NONE,
                           cfg.nx,
                           2,
                           1,
                           nullptr,
                           &st.dm));
    PetscCall(DMSetFromOptions(st.dm));
    PetscCall(DMSetUp(st.dm));
    PetscCall(DMDASetFieldName(st.dm, 0, "phi"));
    PetscCall(DMDASetFieldName(st.dm, 1, "mu"));

    DM dmPhi = nullptr;
    PetscCall(DMDACreateCompatibleDMDA(st.dm, 1, &dmPhi));

    DMDALocalInfo info;
    PetscCall(DMDAGetLocalInfo(dmPhi, &info));
    const PetscInt localSize = info.xm;
    const PetscInt start     = info.xs;

    PetscInt* idxPhi = nullptr;
    PetscInt* idxMu  = nullptr;
    PetscCall(PetscMalloc1(localSize, &idxPhi));
    PetscCall(PetscMalloc1(localSize, &idxMu));

    for (PetscInt i = 0; i < localSize; ++i) {
        idxPhi[i] = start + i;
        idxMu[i]  = start + i + cfg.nx;
    }

    PetscCall(ISCreateGeneral(PETSC_COMM_WORLD,
                              localSize,
                              idxPhi,
                              PETSC_COPY_VALUES,
                              &st.is_phi));
    PetscCall(ISCreateGeneral(PETSC_COMM_WORLD,
                              localSize,
                              idxMu,
                              PETSC_COPY_VALUES,
                              &st.is_mu));

    PetscCall(PetscFree(idxPhi));
    PetscCall(PetscFree(idxMu));

    auto createBlock = [&](Mat& M) -> PetscErrorCode {
        PetscCall(MatCreate(PETSC_COMM_WORLD, &M));
        PetscCall(MatSetSizes(M, localSize, localSize, cfg.nx, cfg.nx));
        PetscCall(MatSetType(M, MATMPIAIJ));
        PetscCall(MatSetFromOptions(M));
        return PETSC_SUCCESS;
    };

    PetscCall(createBlock(st.A11));
    PetscCall(createBlock(st.A12));
    PetscCall(createBlock(st.A21));
    PetscCall(createBlock(st.A22));

    PetscInt* dA11 = nullptr;
    PetscInt* dA12 = nullptr;
    PetscInt* dA21 = nullptr;
    PetscInt* dA22 = nullptr;

    PetscCall(PetscCalloc1(localSize, &dA11));
    PetscCall(PetscCalloc1(localSize, &dA12));
    PetscCall(PetscCalloc1(localSize, &dA21));
    PetscCall(PetscCalloc1(localSize, &dA22));

    for (PetscInt il = 0; il < localSize; ++il) {
        const PetscInt g = start + il;

        dA11[il] = (g == 0 || g == cfg.nx - 1) ? 3 : 1;
        dA12[il] = (g == 0 || g == cfg.nx - 1) ? 2 : 3;
        dA21[il] = 3;
        dA22[il] = 1;
    }

    auto prealloc = [&](Mat M, PetscInt* d) -> PetscErrorCode {
        PetscCall(MatMPIAIJSetPreallocation(M, 0, d, 0, nullptr));
        PetscCall(MatSetUp(M));
        PetscCall(MatSetOption(M, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
        return PETSC_SUCCESS;
    };

    PetscCall(prealloc(st.A11, dA11));
    PetscCall(prealloc(st.A12, dA12));
    PetscCall(prealloc(st.A21, dA21));
    PetscCall(prealloc(st.A22, dA22));

    PetscCall(PetscFree(dA11));
    PetscCall(PetscFree(dA12));
    PetscCall(PetscFree(dA21));
    PetscCall(PetscFree(dA22));

    {
        Mat sub[2][2] = {
            {st.A11, st.A12},
            {st.A21, st.A22}
        };

        IS rows[2] = {st.is_phi, st.is_mu};
        IS cols[2] = {st.is_phi, st.is_mu};

        PetscCall(MatCreateNest(PETSC_COMM_WORLD,
                                2,
                                rows,
                                2,
                                cols,
                                &sub[0][0],
                                &st.A));
        PetscCall(MatSetUp(st.A));
    }

    PetscCall(MatCreateVecs(st.A, &st.x, &st.b));

    PetscCall(VecGetSubVector(st.x, st.is_phi, &st.phi));
    PetscCall(VecGetSubVector(st.x, st.is_mu,  &st.mu));

    PetscCall(VecGetSubVector(st.b, st.is_phi, &st.b1));
    PetscCall(VecGetSubVector(st.b, st.is_mu,  &st.b2));

    PetscCall(VecDuplicate(st.phi, &st.phi_a));
    PetscCall(VecDuplicate(st.phi, &st.phi_0));

    PetscCall(VecDuplicate(st.mu,  &st.mu_a));
    PetscCall(VecDuplicate(st.mu,  &st.mu_0));

    PetscCall(VecDuplicate(st.b1,  &st.b1Source));
    PetscCall(DMCreateGlobalVector(st.dm, &st.source));

    PetscCall(KSPCreate(PETSC_COMM_WORLD, &st.ksp));
    PetscCall(KSPSetOperators(st.ksp, st.A, st.A));

    PetscCall(DMDestroy(&dmPhi));

    PetscPrintf(PETSC_COMM_WORLD, "Sistema TBGCS alocado com sucesso.\n\n");

    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace bgc