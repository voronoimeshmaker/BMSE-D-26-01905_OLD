#pragma once

// ---------------------------------------------------------------------------
//  Coefficients/BGC.hpp
//
//  Cálculo dos coeficientes do estêncil para o modelo BGC clássico
//  (Bevilacqua-Galeão-Costa, campo único).
//
//  Estas funções são puramente matemáticas — não alocam nem acessam
//  nenhum objeto PETSc.  Podem ser chamadas e testadas sem inicializar
//  MPI ou PETSc.
//
//  Referência:
//      Tabela 1  —  coeficientes da matriz A
//      Tabela 2  —  coeficientes do vetor RHS de fronteira
//
//  Separação de responsabilidades:
//      computeCoefficientsBGC  →  produz StencilCoefficients a partir de
//                                 SimConfig.  Chamado uma única vez antes
//                                 do loop transiente, pois h e bv são
//                                 constantes.
//
//      computeRHSBGC           →  produz RHSCoefficients a partir de
//                                 SimConfig e do instante t.  Chamado a
//                                 cada passo temporal porque gamma(t) pode
//                                 variar com o tempo.
//
//  A montagem efetiva na matriz e no vetor PETSc é feita pelas funções em
//  Assembly/BGC.hpp, que recebem estes resultados como entrada.
// ---------------------------------------------------------------------------

#include <bgclib/SimConfig.hpp>
#include <bgclib/Core/Coefficients.hpp>

namespace bgc {

// ---------------------------------------------------------------------------
//  computeCoefficientsBGC
//
//  Calcula os coeficientes do estêncil BGC para a malha e física definidas
//  em cfg.
//
//  Implementa a Tabela 1 do artigo para os cinco grupos de volumes:
//      vol0     —  primeiro volume    (fronteira oeste)
//      vol1     —  segundo volume
//      interior —  volumes i ∈ [2, n-3]  (estêncil simétrico de 5 pontos)
//      volNm2   —  penúltimo volume
//      volNm1   —  último volume      (fronteira leste)
//
//  Os coeficientes dependem de:
//      cfg.h              —  espaçamento da malha
//      cfg.bv             —  número de Bevilacqua
//      cfg.bcWest[0]      —  alpha e beta da primeira BC oeste
//      cfg.bcEast[0]      —  alpha e beta da primeira BC leste
//
//  Nota sobre os índices do estêncil interior:
//      StencilRow::col[] contém offsets relativos { -2, -1, 0, +1, +2 }.
//      O chamador de assembleMatrix soma i a cada col[k] antes de inserir
//      na matriz PETSc.
//
//  Pré-condição: cfg.nx >= 4  e  cfg.h > 0.
// ---------------------------------------------------------------------------

[[nodiscard]]
StencilCoefficients computeCoefficientsCBGC(const SimConfig& cfg);

// ---------------------------------------------------------------------------
//  computeRHSBGC
//
//  Calcula as contribuições de fronteira no vetor RHS para o instante t.
//
//  Implementa a Tabela 2 do artigo para os quatro volumes especiais.
//  Os volumes interiores têm contribuição nula e não são retornados.
//
//  Os coeficientes dependem de:
//      cfg.h              —  espaçamento da malha
//      cfg.bv             —  número de Bevilacqua
//      cfg.bcWest[0,1]    —  alpha, beta e gamma(t) das duas BCs oeste
//      cfg.bcEast[0,1]    —  alpha, beta e gamma(t) das duas BCs leste
//      t                  —  instante de tempo atual
//
//  Correção em relação à versão anterior:
//      O volume 0 possui guarda em gamma idêntica ao volume n-1.
//      Com BCs Neumann puras (alpha=0, beta=1, gamma=0) o denominador
//      contém o fator (beta2*alpha1 - alpha2*beta1) = 0.  A guarda
//      verifica gamma antes de calcular, evitando divisão por zero.
//
//  Pré-condição: cfg.nx >= 4  e  cfg.h > 0.
// ---------------------------------------------------------------------------

[[nodiscard]]
RHSCoefficients computeRHSCBGC(const SimConfig& cfg, PetscReal t);

} // namespace bgc