#pragma once

// ---------------------------------------------------------------------------
//  Transient.hpp
//
//  Loop transiente principal  —  ponto de entrada de cada programa.
//
//  Orquestra a sequência completa:
//      1. Alocação do sistema (BGC ou TBGC) a partir de cfg.
//      2. Condição inicial via phiFn(cfg, x, 0).
//      3. Loop temporal com montagem do RHS, solve e verificação
//         termodinâmica.
//      4. Normas de erro no instante final.
//      5. Libertação do estado.
//
//  cfg.model determina o modelo usado.
//  cfg.nTimes e cfg.dt determinam a duração da simulação.
// ---------------------------------------------------------------------------

#include <petsc.h>
#include <bgclib/SimConfig.hpp>
#include <bgclib/Misc/Types.hpp>

namespace bgc {

/// Executa a simulação transiente completa.
///
/// @param cfg       Configuração completa da simulação.
/// @param sourceFn  Termo fonte fabricado S(cfg, x, t).  nullptr → S=0.
/// @param phiFn     Solução analítica phi(cfg, x, t).
/// @param muFn      Solução analítica mu(cfg, x, t) — usado apenas pelo TBGC.
PetscErrorCode runTransient(const SimConfig& cfg,
                             const FieldFn&   sourceFn,
                             const FieldFn&   phiFn,
                             const FieldFn&   muFn = nullptr);

} // namespace bgc