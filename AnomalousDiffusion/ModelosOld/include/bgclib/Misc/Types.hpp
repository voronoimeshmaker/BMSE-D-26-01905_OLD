#pragma once

// ---------------------------------------------------------------------------
//  Types.hpp
//
//  Tipos fundamentais, enumerações, aliases e constantes matemáticas do
//  simulador de difusão anômala BGC / TBGC.
//
//  Este cabeçalho é incluído por todos os módulos da biblioteca.
//  Não deve depender de nenhum outro cabeçalho interno.
// ---------------------------------------------------------------------------

#include <cmath>
#include <format>
#include <functional>
#include <string>

#include <petsc.h>

namespace bgc {

// ---------------------------------------------------------------------------
//  Modelos disponíveis
// ---------------------------------------------------------------------------

enum class Model {
    BGC,    ///< Esquema clássico de Bevilacqua-Galeão-Costa (campo único).
    TBGC,   ///< Esquema termodinamicamente consistente (sistema 2×2).
    TSPV,   ///< Modelo de duas populações nas variáveis Psi e V (sistema 2×2).
};

// ---------------------------------------------------------------------------
//  Tipo da função campo / termo fonte
//
//  Assinatura uniforme para soluções analíticas e termos fonte fabricados:
//      f(config, x, t)  ->  PetscReal
//
//  O SimConfig é passado como referência constante para que a função possa
//  acessar parâmetros físicos (bv, etc.) sem acoplamento adicional.
// ---------------------------------------------------------------------------

// Forward declaration — SimConfig é definido em SimConfig.hpp.
struct SimConfig;

using FieldFn = std::function<PetscReal(const SimConfig&,
                                        PetscReal x,
                                        PetscReal t)>;

// ---------------------------------------------------------------------------
//  Constantes matemáticas
// ---------------------------------------------------------------------------

/// Valor de π calculado em tempo de compilação.
inline constexpr PetscReal PI = std::numbers::pi_v<PetscReal>;

/// π².
inline constexpr PetscReal PI2 = PI * PI;

/// π⁴.
inline constexpr PetscReal PI4 = PI2 * PI2;

/// Tolerância para comparações com zero em coeficientes de estêncil.
inline constexpr PetscReal STENCIL_ZERO = 1.0e-14;

// ---------------------------------------------------------------------------
//  Utilitário de impressão
//
//  Retorna uma linha separadora de 80 hifens seguida de '\n'.
//  Compatível com PetscPrintf e std::print.
// ---------------------------------------------------------------------------

inline std::string separator() {
    return std::format("{:->80}\n", "");
}

} // namespace bgc
