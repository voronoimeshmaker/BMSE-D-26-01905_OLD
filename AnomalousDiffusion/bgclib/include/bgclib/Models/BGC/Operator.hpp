#pragma once

// BGC-specific builder for the generic DiscreteOperator.
//
// DiscreteOperator itself is shared by all models. This file only contains the
// rules that convert BGC coefficients into that generic representation. It
// should not call MatSetValue or VecSetValue directly.

#include <bgclib/Core/RunContext.hpp>
#include <bgclib/Models/BGC/Coeff.hpp>
#include <bgclib/Numerics/DiscreteOperator.hpp>

namespace bgc::models::bgc {

[[nodiscard]] DiscreteOperator buildOperator(const Grid1D& grid,
                                             const Coefficients& coefficients);

[[nodiscard]] DiscreteOperator buildOperator(Tag tag,
                                             const RunContext& ctx,
                                             const Constants& constants);

} // namespace bgc::models::bgc
