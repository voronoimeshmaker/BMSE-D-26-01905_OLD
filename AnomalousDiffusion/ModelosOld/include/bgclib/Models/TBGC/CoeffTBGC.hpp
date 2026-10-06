#pragma once

// ---------------------------------------------------------------------------
//  Coefficients/TBGC.hpp
//
//  Cálculo dos coeficientes do estêncil para o modelo TBGC
//  (esquema termodinamicamente consistente, sistema de blocos 2×2).
//
//  Estas funções são puramente matemáticas — não alocam nem acessam
//  nenhum objeto PETSc.  Podem ser chamadas e testadas sem inicializar
//  MPI ou PETSc.
//
//  O sistema TBGC resolve as duas equações acopladas:
//
//      (1)   (h/Δt) phi  -  ∇² mu   =  (h/Δt) phi⁰  +  S(x,t)
//      (2)   mu  -  phi  +  Bv ∇² phi  =  0
//
//  A matriz global é um MatNest 2×2:
//
//          [ A11   A12 ] [ phi ]   [ b1 ]
//          [ A21   A22 ] [ mu  ] = [ b2 ]
//
//  Referência:
//      Tabela 3  —  coeficientes dos blocos A11, A12, A21, A22
//      Tabela 4  —  coeficientes do vetor RHS de fronteira b1, b2
//
//  Estruturas retornadas:
//
//      TBGCCoefficients     —  coeficientes dos três blocos ativos
//                              (A22 é a identidade, sem struct necessária)
//
//      TBGCRHSCoefficients  —  contribuições de fronteira em b1 e b2
//
//  Separação de responsabilidades:
//      computeCoefficientsTBGC  →  depende apenas de h, bv e dt.
//                                  Chamado uma única vez antes do loop.
//
//      computeRHSTBGC           →  depende de gamma(t).
//                                  Chamado a cada passo temporal.
// ---------------------------------------------------------------------------

#include <bgclib/SimConfig.hpp>
#include <bgclib/Core/Coefficients.hpp>

namespace bgc {

// ---------------------------------------------------------------------------
//  TBGCCoefficients
//
//  Coeficientes dos três blocos ativos da matriz TBGC.
//
//      A11  —  M/Δτ  +  contribuição de fronteira nos volumes especiais
//              Diagonal nos volumes interiores; dois vizinhos nos extremos.
//
//      A12  —  operador Laplaciano discreto de mu  (-∇²)
//              Estêncil de três pontos em todos os volumes.
//
//      A21  —  relação constitutiva  mu = phi - Bv * ∇²phi
//              Estêncil de cinco pontos nos volumes interiores;
//              estêncis assimétricos nos volumes especiais.
//
//      A22  —  identidade  (não armazenada aqui — montada diretamente)
// ---------------------------------------------------------------------------

struct TBGCCoefficients {
    StencilCoefficients A11;  ///< Bloco (1,1): M/Δτ.
    StencilCoefficients A12;  ///< Bloco (1,2): Laplaciano discreto de mu.
    StencilCoefficients A21;  ///< Bloco (2,1): relação constitutiva mu-phi.
};

// ---------------------------------------------------------------------------
//  TBGCRHSCoefficients
//
//  Contribuições de fronteira nos vetores RHS b1 e b2 para os quatro
//  volumes especiais.
//
//  Recomputado a cada passo temporal porque depende de gamma(t).
// ---------------------------------------------------------------------------

struct TBGCRHSCoefficients {
    RHSCoefficients b1;  ///< RHS da equação de phi  (equação 1).
    RHSCoefficients b2;  ///< RHS da equação de mu   (equação 2).
};

// ---------------------------------------------------------------------------
//  computeCoefficientsTBGC
//
//  Calcula os coeficientes dos blocos A11, A12 e A21 para a malha e física
//  definidas em cfg.
//
//  Implementa a Tabela 3 do artigo para os cinco grupos de volumes em
//  cada bloco.
//
//  Os coeficientes dependem de:
//      cfg.h          —  espaçamento da malha
//      cfg.bv         —  número de Bevilacqua
//      cfg.dt         —  passo de tempo (entra em A11 via h/dt)
//
//  Nota: A22 é a identidade e é montada diretamente em assembleMatrixTBGC,
//  sem passar por esta função.
//
//  Pré-condição: cfg.nx >= 4,  cfg.h > 0,  cfg.dt > 0.
// ---------------------------------------------------------------------------

[[nodiscard]]
TBGCCoefficients computeCoefficientsTBGC(const SimConfig& cfg);

// ---------------------------------------------------------------------------
//  computeRHSTBGC
//
//  Calcula as contribuições de fronteira nos vetores b1 e b2 para o
//  instante t.
//
//  Implementa a Tabela 4 do artigo para os quatro volumes especiais.
//  Os volumes interiores têm contribuição nula e não são retornados.
//
//  Os coeficientes dependem de:
//      cfg.h              —  espaçamento da malha
//      cfg.bv             —  número de Bevilacqua
//      cfg.bcWest[0,1]    —  alpha, beta e gamma(t) das duas BCs oeste
//      cfg.bcEast[0,1]    —  alpha, beta e gamma(t) das duas BCs leste
//      t                  —  instante de tempo atual
//
//  Pré-condição: cfg.nx >= 4,  cfg.h > 0.
// ---------------------------------------------------------------------------

[[nodiscard]]
TBGCRHSCoefficients computeRHSTBGC(const SimConfig& cfg, PetscReal t);

} // namespace bgc