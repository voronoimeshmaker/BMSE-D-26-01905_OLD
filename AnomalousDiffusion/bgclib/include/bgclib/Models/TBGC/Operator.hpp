#pragma once

// Converts TBGC coefficient data into explicit matrix blocks.
//
// The TBGC operator is deliberately represented as four n-by-n blocks. PETSc
// MatNest or another block-aware assembly path can consume these blocks without
// losing the model structure.

#include <bgclib/Core/Grid1D.hpp>
#include <bgclib/Core/RunContext.hpp>
#include <bgclib/Models/TBGC/Coeff.hpp>
#include <bgclib/Numerics/DiscreteOperator.hpp>

namespace bgc::models::tbgc {

struct BlockOperator {
    DiscreteOperator A11;
    DiscreteOperator A12;
    DiscreteOperator A21;
    DiscreteOperator A22;
};

[[nodiscard]] BlockOperator buildBlockOperator(const Grid1D& grid,
                                               const Coefficients& coefficients);

[[nodiscard]] BlockOperator buildBlockOperator(Tag,
                                               const RunContext& ctx,
                                               const Constants& constants);

[[nodiscard]] DiscreteOperator flattenBlockOperator(const BlockOperator& blocks);

[[nodiscard]] DiscreteOperator buildOperator(Tag,
                                             const RunContext& ctx,
                                             const Constants& constants);

} // namespace bgc::models::tbgc
