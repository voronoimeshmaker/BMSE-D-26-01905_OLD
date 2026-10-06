#pragma once

// TSOMPsiV model tag, traits, parameters, and boundary-condition metadata.
//
// TSOMPsiV has two unknowns per control volume, Psi and V. Psi boundary
// conditions remain generic BoundaryCondition objects. The V boundary mode is
// part of the TSOMPsiV model constants because it changes both matrix
// coefficients and boundary source contributions.

#include <array>
#include <stdexcept>
#include <string>
#include <string_view>

#include <petsc.h>

#include <bgclib/Core/Types.hpp>

namespace bgc::models::tsompsiv {

struct Tag {};

enum class VBoundaryCondition {
    // Boundary value V = 0.
    ZeroValue,

    // Boundary gradient dV/dx = cte. The current MMS drivers use the value
    // supplied by the generic boundary condition at each side.
    ConstantGradient,

    // Boundary value V = (1 - sc) * Psi. Consequently U = sc * Psi.
    ScaledPsi,
};

struct VBoundaryConfig {
    VBoundaryCondition type {VBoundaryCondition::ZeroValue};

    // Only used by ScaledPsi. Other modes accept the value in input files but
    // ignore it, so case files can keep a uniform parameter list.
    PetscReal sc {0.0};
};

[[nodiscard]] constexpr std::string_view toString(const VBoundaryCondition type) noexcept {
    switch (type) {
    case VBoundaryCondition::ZeroValue:
        return "zero_value";
    case VBoundaryCondition::ConstantGradient:
        return "constant_gradient";
    case VBoundaryCondition::ScaledPsi:
        return "scaled_psi";
    }
    return "unknown";
}

[[nodiscard]] inline VBoundaryCondition parseVBoundaryCondition(std::string text) {
    for (char& c : text) {
        if (c >= 'A' && c <= 'Z') {
            c = static_cast<char>(c - 'A' + 'a');
        }
        if (c == '-') {
            c = '_';
        }
    }

    if (text == "zero_value" || text == "zero" || text == "v_zero" || text == "v0") {
        return VBoundaryCondition::ZeroValue;
    }
    if (text == "constant_gradient" || text == "dvdx_constant" || text == "dvdx_cte" ||
        text == "zero_gradient" || text == "gradient" || text == "neumann" ||
        text == "dvdx_zero" || text == "dvdx0") {
        return VBoundaryCondition::ConstantGradient;
    }
    if (text == "scaled_psi" || text == "sc_psi" || text == "v_sc_psi") {
        return VBoundaryCondition::ScaledPsi;
    }

    throw std::runtime_error("Invalid TSOMPsiV V boundary condition: " + text);
}

struct Constants {
    PetscReal alpha {0.50};
    PetscReal rho {0.50};
    PetscReal theta {0.5};
    PetscReal lambdaC {1.0};
    PetscReal lambdaR {1.0};
    VBoundaryConfig vBoundary {};
};

using Parameters = Constants;

} // namespace bgc::models::tsompsiv

namespace bgc {

template <>
struct ModelTraits<models::tsompsiv::Tag> {
    using Constants = models::tsompsiv::Constants;
    using Parameters = models::tsompsiv::Parameters;

    static constexpr std::string_view id {"TSOMPsiV"};
    static constexpr std::string_view name {"Two-State One-Dimensional PsiV Model"};
    static constexpr PetscInt fieldCount {2};
    static constexpr std::array<std::string_view, fieldCount> fieldNames {"U", "V"};

    [[nodiscard]] static constexpr bool constantsAreValid(const Constants& constants) noexcept {
        return constants.alpha >= 0.0 && constants.alpha <= 1.0 &&
               constants.rho >= 0.0 && constants.rho <= 1.0 &&
               constants.theta >= 0.0 && constants.theta <= 1.0 &&
               constants.lambdaC >= 0.0 &&
               constants.lambdaR >= 0.0;
    }

    [[nodiscard]] static constexpr bool parametersAreValid(const Parameters& parameters) noexcept {
        return constantsAreValid(parameters);
    }
};

} // namespace bgc
