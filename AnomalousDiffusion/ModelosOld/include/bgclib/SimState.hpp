#pragma once

// ---------------------------------------------------------------------------
//  SimState.hpp
//
//  Agrupa todos os objetos PETSc alocados em tempo de execução.
//  Segue DOD: os dados estão em SimConfig; SimState contém apenas
//  handles e aritmética associada ao sistema linear.
//
//  Ciclo de vida:
//      createState(config, state)   — aloca DM, matrizes, vetores, KSP
//      destroyState(state)          — libera todos os objetos em ordem segura
//
//  Os ponteiros são inicializados como nullptr no construtor padrão,
//  portanto destroyState é seguro para chamar em qualquer ponto.
// ---------------------------------------------------------------------------

#include <petsc.h>
  
namespace bgc {

struct SimConfig;

struct SimState {

    // -----------------------------------------------------------------------
    //  Malha distribuída
    // -----------------------------------------------------------------------

    DM dm {nullptr};

    // -----------------------------------------------------------------------
    //  Vetores de campo
    //
    //  BGC:  phi e autonomo; mu nao e usado.
    //  TBGC: phi e mu sao sub-vetores de x via IS.
    //  TSPV: phi armazena Psi e mu armazena V, ambos como sub-vetores de x.
    // -----------------------------------------------------------------------

    Vec phi   {nullptr};    ///< Primeiro campo numerico (phi no BGC/TBGC, Psi no TSPV).
    Vec mu    {nullptr};    ///< Segundo campo numerico (mu no TBGC, V no TSPV).
    Vec phi_0 {nullptr};    ///< Campo phi no passo temporal anterior.
    Vec mu_0  {nullptr};    ///< Segundo campo no passo temporal anterior (TSPV).
    Vec phi_a {nullptr};    ///< Solução analítica de phi.
    Vec mu_a  {nullptr};    ///< Solução analítica de mu (apenas TBGC).

    // -----------------------------------------------------------------------
    //  Vetores do sistema linear
    // -----------------------------------------------------------------------

    Vec b        {nullptr};     ///< Vetor RHS monolítico.
    Vec x        {nullptr};     ///< Vetor solução monolítico.
    Vec b1Source {nullptr};     ///< Contribuição do termo fonte em b1.
    Vec source   {nullptr};     ///< Vetor auxiliar do termo fonte.

    // -----------------------------------------------------------------------
    //  Sub-vetores dos sistemas 2x2 (TBGC e TSPV)
    // -----------------------------------------------------------------------

    Vec b1 {nullptr};   ///< Sub-vetor do RHS  —  equação de phi.
    Vec b2 {nullptr};   ///< Sub-vetor do RHS  —  equação de mu.

    // -----------------------------------------------------------------------
    //  Matrizes
    // -----------------------------------------------------------------------

    Mat A   {nullptr};  ///< Matriz global (MatNest para TBGC/TSPV; MPIAIJ para BGC).
    Mat A11 {nullptr};  ///< Bloco (1,1) do sistema 2x2.
    Mat A12 {nullptr};  ///< Bloco (1,2) do sistema 2x2.
    Mat A21 {nullptr};  ///< Bloco (2,1) do sistema 2x2.
    Mat A22 {nullptr};  ///< Bloco (2,2) do sistema 2x2.

    // -----------------------------------------------------------------------
    //  Solver
    // -----------------------------------------------------------------------

    KSP ksp {nullptr};  ///< Solver linear de Krylov.
    PC  pc  {nullptr};  ///< Pré-condicionador (propriedade interna do KSP).

    // -----------------------------------------------------------------------
    //  Index sets dos sistemas 2x2 (TBGC e TSPV)
    // -----------------------------------------------------------------------

    IS is_phi {nullptr};    ///< Index set de phi no vetor monolítico.
    IS is_mu  {nullptr};    ///< Index set de mu  no vetor monolítico.

    // -----------------------------------------------------------------------
    //  Parâmetro de largura de estêncil
    // -----------------------------------------------------------------------

    static constexpr PetscInt stencilWidth {3};
};

// ---------------------------------------------------------------------------
//  Funções de ciclo de vida
// ---------------------------------------------------------------------------

/// Aloca e configura todos os objetos PETSc para o modelo BGC.
PetscErrorCode createStateBGC(const SimConfig& cfg, SimState& st);

/// Aloca e configura todos os objetos PETSc para o modelo TBGC.
PetscErrorCode createStateTBGC(const SimConfig& cfg, SimState& st);

/// Aloca e configura todos os objetos PETSc para o modelo TBGC.
PetscErrorCode createStateTBGCS(const SimConfig& cfg, SimState& st);


/// Aloca e configura todos os objetos PETSc para o modelo TSUV.
PetscErrorCode createStateTSUV(const SimConfig& cfg, SimState& st);

/// Compatibilidade temporaria com o nome antigo TSPV.
PetscErrorCode createStateTSPV(const SimConfig& cfg, SimState& st);

/// Libera todos os objetos PETSc em ordem segura.  Idempotente.
PetscErrorCode destroyState(SimState& st);

} // namespace bgc
