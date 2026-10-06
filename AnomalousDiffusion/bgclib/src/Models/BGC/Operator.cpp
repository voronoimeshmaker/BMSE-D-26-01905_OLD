#include <bgclib/Models/BGC/Operator.hpp>

namespace bgc::models::bgc {

// Converts BGC stencil rows into the shared sparse operator format.
//
// Boundary rows already store absolute column indices. Interior rows store
// offsets relative to the current control volume and are expanded here.
DiscreteOperator buildOperator(const Grid1D& grid, const Coefficients& coefficients) {
    DiscreteOperator op {
        .layout = {
            .kind = MatrixLayoutKind::Scalar,
            .rows = grid.nx,
            .cols = grid.nx,
            .blockSize = 1,
        },
        .entries = {},
    };

    auto appendRow = [&op](const PetscInt row,
                           const StencilRow& stencil,
                           const bool relativeColumns) {
        for (PetscInt k = 0; k < stencil.ncols; ++k) {
            op.entries.push_back({
                .row = row,
                .col = relativeColumns ? row + stencil.col[static_cast<std::size_t>(k)]
                                       : stencil.col[static_cast<std::size_t>(k)],
                .value = stencil.coef[static_cast<std::size_t>(k)],
            });
        }
    };

    appendRow(0, coefficients.vol0, false);
    appendRow(1, coefficients.vol1, false);

    for (PetscInt i = 2; i < grid.nx - 2; ++i) {
        appendRow(i, coefficients.interior, true);
    }

    appendRow(grid.nx - 2, coefficients.volNm2, false);
    appendRow(grid.nx - 1, coefficients.volNm1, false);

    return op;
}

// Contract entry point used by generic model code.
DiscreteOperator buildOperator(Tag, const RunContext& ctx, const Constants& constants) {
    return buildOperator(ctx.sim.grid, computeCoefficients(ctx.sim.grid, constants));
}

} // namespace bgc::models::bgc
