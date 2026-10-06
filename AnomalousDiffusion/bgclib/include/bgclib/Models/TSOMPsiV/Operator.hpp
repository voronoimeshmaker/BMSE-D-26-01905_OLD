#pragma once

// Converts TSOMPsiV coefficient data into generic matrix blocks.

#include <bgclib/Core/Grid1D.hpp>
#include <bgclib/Models/TSOMPsiV/Coeff.hpp>
#include <bgclib/Numerics/DiscreteOperator.hpp>

namespace bgc::models::tsompsiv {

struct BlockOperator {
    DiscreteOperator A11;
    DiscreteOperator A12;
    DiscreteOperator A21;
    DiscreteOperator A22;
};

[[nodiscard]] BlockOperator buildBlockOperator(const Grid1D& grid,
                                               const Coefficients& coefficients);

[[nodiscard]] DiscreteOperator flattenBlockOperator(const BlockOperator& blocks);

[[nodiscard]] DiscreteOperator buildOperator(const Grid1D& grid,
                                             const Coefficients& coefficients);

} // namespace bgc::models::tsompsiv
