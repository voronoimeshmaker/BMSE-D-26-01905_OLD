#pragma once

// ---------------------------------------------------------------------------
//  Coefficients/Coefficients.hpp
//
//  Estruturas de dados e utilitários comuns a todos os modelos de difusão
//  anômala implementados na bgclib.
//
//  Este arquivo é o equivalente sem herança de uma classe base abstrata:
//  define o contrato que BGC, TBGC e qualquer modelo futuro devem respeitar.
//
//  Não depende de PETSc  —  contém apenas tipos escalares e lógica de
//  classificação de volumes.  Pode ser testado sem inicializar MPI.
//
//  Organização:
//
//      StencilRow              —  uma linha genérica do estêncil
//      StencilCoefficients     —  conjunto completo de linhas da matriz
//      RHSCoefficients         —  contribuições de fronteira no vetor RHS
//      VolumeRegion            —  classificação dos cinco grupos de volumes
//      classifyVolume()        —  retorna a região de um volume dado seu índice
//
//  Invariante de qualquer modelo 1D com equação de quarta ordem:
//
//      A malha possui sempre quatro volumes com equações especiais,
//      independentemente do modelo físico:
//
//          Volume 0    —  primeiro      (fronteira oeste)
//          Volume 1    —  segundo       (adjacente à fronteira oeste)
//          Volume n-2  —  penúltimo     (adjacente à fronteira leste)
//          Volume n-1  —  último        (fronteira leste)
//
//      Todos os demais volumes i ∈ [2, n-3] seguem o estêncil interior.
// ---------------------------------------------------------------------------

#include <array>

#include <petsc.h>

namespace bgc {

// ---------------------------------------------------------------------------
//  StencilRow
//
//  Representa uma única linha da matriz de coeficientes do estêncil.
//
//  Semântica:
//      sum_{k=0}^{ncols-1}  coef[k] * Phi[ col[k] ]
//
//  Para volumes de fronteira:
//      ncols ≤ 4, col[] contém índices globais absolutos (0-based).
//
//  Para volumes interiores:
//      ncols = 5, col[] contém offsets relativos ao volume i:
//          col = { -2, -1, 0, +1, +2 }
//      O chamador de assembleMatrix soma i a cada col[k] antes de inserir.
//
//  Capacidade máxima: 5 entradas (estêncil de cinco pontos).
// ---------------------------------------------------------------------------

struct StencilRow {
    std::array<PetscReal, 5> coef  {};   ///< Coeficientes da linha.
    std::array<PetscInt,  5> col   {};   ///< Índices de coluna (absolutos ou relativos).
    int                      ncols {0};  ///< Número de entradas ativas (1 a 5).
};

// ---------------------------------------------------------------------------
//  StencilCoefficients
//
//  Conjunto completo de linhas da matriz para um modelo e uma malha fixos.
//
//  Calculado uma única vez antes do loop transiente, pois depende apenas
//  de h e bv — ambos constantes ao longo da simulação.
//
//  Nomenclatura:
//      vol0      —  linha do volume 0    (primeiro)
//      vol1      —  linha do volume 1    (segundo)
//      interior  —  linha padrão dos volumes i ∈ [2, n-3]
//      volNm2    —  linha do volume n-2  (penúltimo)
//      volNm1    —  linha do volume n-1  (último)
// ---------------------------------------------------------------------------

struct StencilCoefficients {
    StencilRow vol0;      ///< Volume 0    —  primeiro.
    StencilRow vol1;      ///< Volume 1    —  segundo.
    StencilRow interior;  ///< Volumes interiores i ∈ [2, n-3].
    StencilRow volNm2;    ///< Volume n-2  —  penúltimo.
    StencilRow volNm1;    ///< Volume n-1  —  último.
};

// ---------------------------------------------------------------------------
//  RHSCoefficients
//
//  Contribuições de fronteira no vetor RHS para os quatro volumes especiais.
//
//  Ao contrário de StencilCoefficients, este struct é recomputado a cada
//  passo temporal porque depende de gamma(t), que pode variar com o tempo.
//
//  Os volumes interiores têm contribuição de fronteira nula e não precisam
//  ser armazenados.
// ---------------------------------------------------------------------------

struct RHSCoefficients {
    PetscReal vol0   {0.0};   ///< Contribuição de fronteira no volume 0.
    PetscReal vol1   {0.0};   ///< Contribuição de fronteira no volume 1.
    PetscReal volNm2 {0.0};   ///< Contribuição de fronteira no volume n-2.
    PetscReal volNm1 {0.0};   ///< Contribuição de fronteira no volume n-1.
};

// ---------------------------------------------------------------------------
//  VolumeRegion
//
//  Classificação dos cinco grupos de volumes de controle.
//  Usada em assembleMatrix para selecionar a linha correta do estêncil
//  sem encadeamento de condicionais.
//
//  Exemplo de uso:
//
//      switch (classifyVolume(i, n)) {
//          case VolumeRegion::First:
//              insertRow(i, sc.vol0);
//              break;
//          case VolumeRegion::Second:
//              insertRow(i, sc.vol1);
//              break;
//          case VolumeRegion::Interior:
//              insertRow(i, sc.interior, true);   // true = offsets relativos
//              break;
//          case VolumeRegion::SecondToLast:
//              insertRow(i, sc.volNm2);
//              break;
//          case VolumeRegion::Last:
//              insertRow(i, sc.volNm1);
//              break;
//      }
// ---------------------------------------------------------------------------

enum class VolumeRegion {
    First,         ///< Volume 0    —  fronteira oeste.
    Second,        ///< Volume 1    —  adjacente à fronteira oeste.
    Interior,      ///< Volumes i ∈ [2, n-3]  —  estêncil padrão.
    SecondToLast,  ///< Volume n-2  —  adjacente à fronteira leste.
    Last,          ///< Volume n-1  —  fronteira leste.
};

// ---------------------------------------------------------------------------
//  classifyVolume
//
//  Retorna a VolumeRegion correspondente ao volume de índice global i
//  numa malha com n volumes de controle.
//
//  Pré-condição: 0 <= i < n  e  n >= 4.
// ---------------------------------------------------------------------------

[[nodiscard]]
inline VolumeRegion classifyVolume(PetscInt i, PetscInt n) noexcept {
    if (i == 0)     return VolumeRegion::First;
    if (i == 1)     return VolumeRegion::Second;
    if (i == n - 2) return VolumeRegion::SecondToLast;
    if (i == n - 1) return VolumeRegion::Last;
    return VolumeRegion::Interior;
}

} // namespace bgc