// ---------------------------------------------------------------------------
//  Analytics.cpp
//
//  Avaliação da solução analítica e do termo fonte fabricado.
//
//  Todas as funções operam exclusivamente sobre os volumes locais do
//  processo corrente.  Nenhuma comunicação MPI é necessária: cada processo
//  calcula os seus próprios valores a partir das funções fornecidas pelo
//  chamador.
//
//  computeSourceTerm — escalamento por modelo:
//
//      TBGC:  b1Source[i] = sourceFn * h   +  (h/dt)   * phi_0[i]
//             Consistente com A11 interior diagonal = h/dt.
//
//      BGC:   b1Source[i] = sourceFn * h   +  (h/dt)   * phi_0[i]
//             Consistente com a matriz BGC e com MatShift = h/dt.
// ---------------------------------------------------------------------------

#include <bgclib/Analytics.hpp>
#include <bgclib/SimConfig.hpp>
#include <bgclib/SimState.hpp>
#include <bgclib/Misc/Types.hpp>

namespace bgc {

// ---------------------------------------------------------------------------
//  computeAnalyticPhi
//
//  phi_a[i] = phiFn(cfg, x_i, t)   para todo i local.
// ---------------------------------------------------------------------------

PetscErrorCode computeAnalyticPhi(const SimConfig& cfg,
                                  SimState&        st,
                                  PetscReal        t,
                                  const FieldFn&   phiFn) {
    PetscFunctionBeginUser;

    PetscInt xs, xe;
    PetscCall(VecGetOwnershipRange(st.phi_a, &xs, &xe));

    for (PetscInt i = xs; i < xe; ++i) {
        const PetscReal xi  = cfg.xCenter(i);
        const PetscReal val = phiFn(cfg, xi, t);
        PetscCall(VecSetValue(st.phi_a, i, val, INSERT_VALUES));
    }

    PetscCall(VecAssemblyBegin(st.phi_a));
    PetscCall(VecAssemblyEnd  (st.phi_a));

    PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------------------------------------------
//  computeAnalyticMu
//
//  mu_a[i] = muFn(cfg, x_i, t)   para todo i local.
//  Chamado apenas pelo TBGC.
// ---------------------------------------------------------------------------

PetscErrorCode computeAnalyticMu(const SimConfig& cfg,
                                 SimState&        st,
                                 PetscReal        t,
                                 const FieldFn&   muFn) {
    PetscFunctionBeginUser;

    PetscInt xs, xe;
    PetscCall(VecGetOwnershipRange(st.mu_a, &xs, &xe));

    for (PetscInt i = xs; i < xe; ++i) {
        const PetscReal xi  = cfg.xCenter(i);
        const PetscReal val = muFn(cfg, xi, t);
        PetscCall(VecSetValue(st.mu_a, i, val, INSERT_VALUES));
    }

    PetscCall(VecAssemblyBegin(st.mu_a));
    PetscCall(VecAssemblyEnd  (st.mu_a));

    PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------------------------------------------
//  computeSourceTerm
//
//  Preenche b1Source com a contribuição do termo fonte e do nível
//  temporal anterior.  O escalamento depende do modelo:
//
//      TBGC:  b1Source[i] = sourceFn * h + (h/dt) * phi_0[i]
//      BGC:   b1Source[i] = sourceFn * h + (h/dt) * phi_0[i]
//
//  O acesso a phi_0 usa VecGetArrayRead para evitar cópias desnecessárias.
// ---------------------------------------------------------------------------

PetscErrorCode computeSourceTerm(const SimConfig& cfg,
                                 SimState&        st,
                                 PetscReal        t,
                                 const FieldFn&   sourceFn) {
    PetscFunctionBeginUser;

    PetscInt xs, xe;
    PetscCall(VecGetOwnershipRange(st.phi_0, &xs, &xe));
    const PetscInt localSize = xe - xs;

    const PetscScalar* arr0 = nullptr;
    PetscCall(VecGetArrayRead(st.phi_0, &arr0));

    const PetscReal h    = cfg.h;
    const PetscReal hdt  = h / cfg.dt;

    const bool isTBGC = (cfg.model == Model::TBGC);

    for (PetscInt il = 0; il < localSize; ++il) {
        const PetscInt  i    = xs + il;
        const PetscReal xi   = cfg.xCenter(i);
        const PetscReal src  = sourceFn(cfg, xi, t);
        const PetscReal phi0 = static_cast<PetscReal>(arr0[il]);

        const PetscReal val = isTBGC
            ? src * h + hdt * phi0   // TBGC: consistente com A11 = h/dt
            : src * h + hdt * phi0;  // BGC:  consistente com MatShift = h/dt

        PetscCall(VecSetValue(st.b1Source, i, val, INSERT_VALUES));
    }

    PetscCall(VecRestoreArrayRead(st.phi_0, &arr0));

    PetscCall(VecAssemblyBegin(st.b1Source));
    PetscCall(VecAssemblyEnd  (st.b1Source));

    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace bgc