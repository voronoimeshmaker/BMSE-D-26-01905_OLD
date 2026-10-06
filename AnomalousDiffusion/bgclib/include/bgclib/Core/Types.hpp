#pragma once

// -----------------------------------------------------------------------------
// Types.hpp
//
// Fundamental aliases, constants, and lightweight shared types for bgclib.
//
// This header should stay small. It may be included by most library modules, so
// avoid dependencies on other internal bgclib headers.
//
// Model identity is intentionally not represented by a central enum. Models are
// identified by model tags, ModelTraits specializations, and runtime ModelOps
// entries registered by each model module.
// -----------------------------------------------------------------------------

#include <cmath>
#include <format>
#include <functional>
#include <numbers>
#include <string>
#include <string_view>

#include <petsc.h>

namespace bgc {

// -----------------------------------------------------------------------------
// Forward Declarations
// -----------------------------------------------------------------------------

struct SimConfig;

// -----------------------------------------------------------------------------
// Field And Source Function Types
//
// Uniform signature for exact fields and manufactured source terms:
//
//     f(config, x, t) -> PetscReal
//
// SimConfig is passed by const reference so field/source functions can access
// physical parameters without additional coupling.
// -----------------------------------------------------------------------------

using FieldFn = std::function<PetscReal(const SimConfig&,
                                        PetscReal x,
                                        PetscReal t)>;

// -----------------------------------------------------------------------------
// Mathematical Constants
// -----------------------------------------------------------------------------

inline constexpr PetscReal PI = std::numbers::pi_v<PetscReal>;
inline constexpr PetscReal PI2 = PI * PI;
inline constexpr PetscReal PI4 = PI2 * PI2;

// Tolerance used when comparing stencil coefficients with zero.
inline constexpr PetscReal STENCIL_ZERO = 1.0e-14;

// -----------------------------------------------------------------------------
// Model Identity
//
// Models provide their own empty tag type:
//
//     namespace bgc::models::bgc {
//     struct Tag {};
//     }
//
// Each tag is described through ModelTraits<Tag>. This avoids a central model
// enum and allows new models to be added without editing this header.
// -----------------------------------------------------------------------------

template <typename ModelTag>
struct ModelTraits;

// -----------------------------------------------------------------------------
// Formatting Helpers
// -----------------------------------------------------------------------------

inline std::string separator() {
    return std::format("{:->80}\n", "");
}

} // namespace bgc
