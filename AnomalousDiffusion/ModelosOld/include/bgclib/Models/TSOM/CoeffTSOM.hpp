#pragma once

#include <bgclib/Core/Coefficients.hpp>
#include <bgclib/SimConfig.hpp>

namespace bgc {

struct TSOMDiffusionCoefficients {
    PetscReal d11 {0.0};  ///< U equation, U_xx coefficient.
    PetscReal d12 {0.0};  ///< U equation, V_xx coefficient.
    PetscReal d21 {0.0};  ///< V equation, U_xx coefficient.
    PetscReal d22 {0.0};  ///< V equation, V_xx coefficient.
};

struct TSOMBlockCoefficients {
    StencilRow first;     ///< Control volume 0.
    StencilRow interior;  ///< Control volumes i in [1, n-2] when n > 2.
    StencilRow last;      ///< Control volume n-1.
};

struct TSOMCoefficients {
    TSOMBlockCoefficients A11;  ///< U equation, U column.
    TSOMBlockCoefficients A12;  ///< U equation, V column.
    TSOMBlockCoefficients A21;  ///< V equation, U column.
    TSOMBlockCoefficients A22;  ///< V equation, V column.
};



struct TSOMBoundaryRHS {
    PetscReal first {0.0};
    PetscReal last  {0.0};
};

struct TSOMRHSCoefficients {
    TSOMBoundaryRHS b1;  ///< Boundary contributions in the U equation.
    TSOMBoundaryRHS b2;  ///< Boundary contributions in the V equation.
};

[[nodiscard]]
TSOMDiffusionCoefficients computeDiffusionCoefficientsTSOM(
    const SimConfig& cfg) noexcept;

[[nodiscard]]
TSOMCoefficients computeCoefficientsTSOM(const SimConfig& cfg);

[[nodiscard]]
TSOMRHSCoefficients computeRHSTSOM(const SimConfig& cfg, PetscReal t);

} // namespace bgc
