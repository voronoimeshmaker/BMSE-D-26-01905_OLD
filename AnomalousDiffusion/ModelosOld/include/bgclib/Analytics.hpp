#pragma once

// ---------------------------------------------------------------------------
//  Analytics.hpp
//
//  Avaliação da solução analítica e do termo fonte fabricado em todos
//  os volumes de controle do domínio local do processo.
// ---------------------------------------------------------------------------

#include <petsc.h>
#include <bgclib/Misc/Types.hpp>

namespace bgc {

struct SimConfig;
struct SimState;

/// Preenche phi_a com PhiFn(cfg, x_i, t) em todos os volumes locais.
PetscErrorCode computeAnalyticPhi(const SimConfig& cfg,
                                  SimState&        st,
                                  PetscReal        t,
                                  const FieldFn&   phiFn);

/// Preenche mu_a com MuFn(cfg, x_i, t) em todos os volumes locais (apenas TBGC).
PetscErrorCode computeAnalyticMu(const SimConfig& cfg,
                                 SimState&        st,
                                 PetscReal        t,
                                 const FieldFn&   muFn);

/// Preenche b1Source com:
///   TBGC: sourceFn(x,t)*h + (h/dt)*phi_0[i]
///   BGC:  sourceFn(x,t)*h^4
PetscErrorCode computeSourceTerm(const SimConfig& cfg,
                                 SimState&        st,
                                 PetscReal        t,
                                 const FieldFn&   sourceFn);

} // namespace bgc
