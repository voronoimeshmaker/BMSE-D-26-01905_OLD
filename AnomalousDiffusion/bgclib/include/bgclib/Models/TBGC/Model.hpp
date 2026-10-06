#pragma once

// TBGC model tag, traits, and registration metadata.
//
// TBGC is the new-library name for the old TBGCS model. The model has two
// coupled fields per control volume, phi and mu, and keeps its matrix as four
// PETSc-compatible blocks.

#include <array>
#include <string_view>

#include <petsc.h>

#include <bgclib/Core/Types.hpp>

namespace bgc::models::tbgc {

struct Tag {};

struct Constants {
    PetscReal bv {0.0};
};

using Parameters = Constants;

} // namespace bgc::models::tbgc

namespace bgc {

template <>
struct ModelTraits<models::tbgc::Tag> {
    using Constants = models::tbgc::Constants;
    using Parameters = models::tbgc::Parameters;

    static constexpr std::string_view id {"TBGC"};
    static constexpr std::string_view name {"Thermodynamic BGC"};
    static constexpr PetscInt fieldCount {2};
    static constexpr std::array<std::string_view, fieldCount> fieldNames {"phi", "mu"};

    [[nodiscard]] static constexpr bool constantsAreValid(const Constants& constants) noexcept {
        return constants.bv >= 0.0;
    }

    [[nodiscard]] static constexpr bool parametersAreValid(const Parameters& parameters) noexcept {
        return constantsAreValid(parameters);
    }
};

} // namespace bgc
