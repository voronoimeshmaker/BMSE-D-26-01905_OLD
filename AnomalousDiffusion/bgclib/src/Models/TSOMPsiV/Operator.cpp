#include <bgclib/Models/TSOMPsiV/Operator.hpp>

// TSOMPsiV operator implementation.
//
// Coefficient code produces four scalar blocks A11, A12, A21, and A22. This
// file converts each block to a DiscreteOperator and then flattens the blocks
// into the global ordering:
//
//   [ U_0 ... U_{N-1}  V_0 ... V_{N-1} ]
//
// PETSc matrix allocation is deliberately left to the generic assembly layer.

namespace bgc::models::tsompsiv {

namespace {

DiscreteOperator makeBlock(const Grid1D& grid) {
    return {
        .layout = {
            .kind = MatrixLayoutKind::Scalar,
            .rows = grid.nx,
            .cols = grid.nx,
            .blockSize = 1,
        },
        .entries = {},
    };
}

void appendRow(DiscreteOperator& op,
               const PetscInt row,
               const StencilRow& stencil,
               const bool relativeCols) {
    for (PetscInt k = 0; k < stencil.ncols; ++k) {
        op.entries.push_back({
            .row = row,
            .col = relativeCols ? row + stencil.col[static_cast<std::size_t>(k)]
                                : stencil.col[static_cast<std::size_t>(k)],
            .value = stencil.coef[static_cast<std::size_t>(k)],
        });
    }
}

DiscreteOperator buildStencilBlock(const Grid1D& grid, const StencilSet& stencil) {
    DiscreteOperator op = makeBlock(grid);
    const PetscInt n = grid.nx;

    appendRow(op, 0, stencil.first, false);
    for (PetscInt i = 1; i < n - 1; ++i) {
        appendRow(op, i, stencil.interior, true);
    }
    appendRow(op, n - 1, stencil.last, false);
    return op;
}

void appendShiftedBlock(std::vector<OperatorEntry>& entries,
                        const DiscreteOperator& block,
                        const PetscInt rowOffset,
                        const PetscInt colOffset) {
    for (const OperatorEntry& entry : block.entries) {
        entries.push_back({
            .row = rowOffset + entry.row,
            .col = colOffset + entry.col,
            .value = entry.value,
        });
    }
}

} // namespace

BlockOperator buildBlockOperator(const Grid1D& grid, const Coefficients& coefficients) {
    return {
        .A11 = buildStencilBlock(grid, coefficients.A11),
        .A12 = buildStencilBlock(grid, coefficients.A12),
        .A21 = buildStencilBlock(grid, coefficients.A21),
        .A22 = buildStencilBlock(grid, coefficients.A22),
    };
}

DiscreteOperator flattenBlockOperator(const BlockOperator& blocks) {
    const PetscInt n = blocks.A11.layout.rows;

    DiscreteOperator op {
        .layout = {
            .kind = MatrixLayoutKind::Block,
            .rows = 2 * n,
            .cols = 2 * n,
            .blockSize = 2,
        },
        .entries = {},
    };

    appendShiftedBlock(op.entries, blocks.A11, 0, 0);
    appendShiftedBlock(op.entries, blocks.A12, 0, n);
    appendShiftedBlock(op.entries, blocks.A21, n, 0);
    appendShiftedBlock(op.entries, blocks.A22, n, n);
    return op;
}

DiscreteOperator buildOperator(const Grid1D& grid, const Coefficients& coefficients) {
    return flattenBlockOperator(buildBlockOperator(grid, coefficients));
}

} // namespace bgc::models::tsompsiv
