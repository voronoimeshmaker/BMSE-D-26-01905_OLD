#pragma once

// BGC model tag, constants, and compile-time metadata.
//
// The physics implementation lives in free functions associated with this tag.
// Keeping this header focused on identity and constants makes it cheap to
// include from tests, registries, and future runtime dispatch tables.

#include <array>
#include <string_view>

#include <bgclib/Core/Types.hpp>

namespace bgc::models::bgc {

// Empty type used for compile-time dispatch.
struct Tag {};

// Physical constants for the BGC model.
struct Constants {
    // Bevilacqua number used by the fourth-order anomalous diffusion term.
    PetscReal bv {0.0};
};

// Compatibility alias while older code still says Parameters.
using Parameters = Constants;

} // namespace bgc::models::bgc

namespace bgc {

template <>
struct ModelTraits<models::bgc::Tag> {
    using Constants = models::bgc::Constants;
    using Parameters = Constants;

    // Stable runtime id used by input files and registries.
    static constexpr std::string_view id {"BGC"};

    // Human-readable model name for diagnostics and reports.
    static constexpr std::string_view name {"Burgers-like Generic Concentration"};

    // Number of coupled fields owned by this model.
    static constexpr PetscInt fieldCount {1};

    // Field labels used by output and diagnostics.
    static constexpr std::array<std::string_view, fieldCount> fieldNames {"u"};

    [[nodiscard]] static constexpr bool constantsAreValid(const Constants& constants) noexcept {
        return constants.bv >= 0.0;
    }

    [[nodiscard]] static constexpr bool parametersAreValid(const Parameters& parameters) noexcept {
        return constantsAreValid(parameters);
    }
};

} // namespace bgc
