#include <bgclib/PostProcess.hpp>
#include <bgclib/Analytics.hpp>
#include <bgclib/SimConfig.hpp>
#include <bgclib/SimState.hpp>
#include <bgclib/Misc/Types.hpp>

#include <cmath>
#include <format>

namespace bgc {

// ---------------------------------------------------------------------------
//  computeErrorNorms
// ---------------------------------------------------------------------------

PetscErrorCode computeErrorNorms(const SimConfig& cfg,
                                 SimState&        st,
                                 PetscReal        t,
                                 const FieldFn&   phiFn,
                                 ErrorNorms&      norms) {
    PetscFunctionBeginUser;

    // Atualiza phi_a com a solução analítica no instante t
    PetscCall(computeAnalyticPhi(cfg, st, t, phiFn));

    // Vetor de erro temporário: e = phi_a - phi
    Vec erro = nullptr;
    PetscCall(VecDuplicate(st.phi_a, &erro));
    PetscCall(VecCopy(st.phi_a, erro));
    PetscCall(VecAXPY(erro, -1.0, st.phi));

    // Normas PETSc (sem escalonamento)
    PetscReal L1 = 0.0;
    PetscReal L2 = 0.0;
    PetscReal Linf = 0.0;

    PetscCall(VecNorm(erro, NORM_1,        &L1));
    PetscCall(VecNorm(erro, NORM_2,        &L2));
    PetscCall(VecNorm(erro, NORM_INFINITY, &Linf));

    PetscCall(VecDestroy(&erro));

    // Escalonamento correcto para normas discretas
    const PetscReal h = cfg.h;

    norms.L1   = h * L1;
    norms.L2   = std::sqrt(h) * L2;
    norms.Linf = Linf;

    if (cfg.verbose) {
        PetscPrintf(PETSC_COMM_WORLD,
                    "[erro] tau=%.6e  L1=%.6e  L2=%.6e  Linf=%.6e\n",
                    static_cast<double>(t),
                    static_cast<double>(norms.L1),
                    static_cast<double>(norms.L2),
                    static_cast<double>(norms.Linf));
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------------------------------------------
//  checkThermodynamicConsistency
// ---------------------------------------------------------------------------

PetscErrorCode checkThermodynamicConsistency(const SimConfig& cfg,
                                             SimState&        st,
                                             PetscReal        t,
                                             PetscReal&       F_prev,
                                             PetscBool&       violated) {
    PetscFunctionBeginUser;

    // F_h^n = (h/2) * ||phi||_2^2
    PetscReal norm2 = 0.0;
    PetscCall(VecNorm(st.phi, NORM_2, &norm2));
    const PetscReal F_curr = 0.5 * cfg.h * norm2 * norm2;
    const PetscReal dF     = (F_curr - F_prev) / cfg.dt;

    violated = (F_curr > F_prev + 1.0e-14) ? PETSC_TRUE : PETSC_FALSE;

    PetscPrintf(PETSC_COMM_WORLD,
                "[thermo] tau=%.6e  F=%.10e  dF/dtau=%.4e  %s\n",
                (double)t, (double)F_curr, (double)dF,
                violated ? "*** VIOLACAO ***" : "ok");

    // Verificação local das faces (apenas TBGC, quando mu está disponível)
    if (st.mu) {
        PetscInt xs, xe;
        PetscCall(VecGetOwnershipRange(st.phi, &xs, &xe));
        const PetscInt localSize = xe - xs;

        const PetscScalar* phi_arr = nullptr;
        const PetscScalar* mu_arr  = nullptr;
        PetscCall(VecGetArrayRead(st.phi, &phi_arr));
        PetscCall(VecGetArrayRead(st.mu,  &mu_arr));

        const PetscReal h_inv       = 1.0 / cfg.h;
        PetscInt        nFaces      = 0;
        PetscReal       maxViolacao = 0.0;

        for (PetscInt il = 0; il < localSize - 1; ++il) {
            const PetscReal J_face  = -(mu_arr[il + 1] - mu_arr[il]) * h_inv;
            const PetscReal dphi    =   phi_arr[il + 1] - phi_arr[il];
            const PetscReal produto = J_face * dphi;

            if (produto > 1.0e-14) {
                ++nFaces;
                if (produto > maxViolacao) {
                    maxViolacao = produto;
                }
            }
        }

        PetscCall(VecRestoreArrayRead(st.phi, &phi_arr));
        PetscCall(VecRestoreArrayRead(st.mu,  &mu_arr));

        if (nFaces > 0) {
            violated = PETSC_TRUE;
            PetscPrintf(PETSC_COMM_WORLD,
                        "[thermo] Faces anti-difusivas: %d  max(J*dphi)=%.4e"
                        "  *** FLUXO ANTI-DIFUSIVO ***\n",
                        (int)nFaces, (double)maxViolacao);
        }
    }

    F_prev = F_curr;
    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace bgc