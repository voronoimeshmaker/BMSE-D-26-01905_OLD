#include <bgclib/Numerics/Assembly/GenericMatrixAssembly.hpp>

#include <algorithm>
#include <vector>

// AIJ assembly from global sparse triplets. In parallel, PETSc owns a row range
// on each rank and only the owner inserts those rows, avoiding duplicated matrix
// coefficients when the same model data is visible on every process.

namespace bgc {

PetscErrorCode createSequentialMatrix(const DiscreteOperator& op, Mat& matrix) {
    PetscFunctionBeginUser;

    std::vector<PetscInt> rowNnz(static_cast<std::size_t>(op.layout.rows), 0);
    for (const OperatorEntry& entry : op.entries) {
        if (entry.row >= 0 && entry.row < op.layout.rows) {
            ++rowNnz[static_cast<std::size_t>(entry.row)];
        }
    }
    const PetscInt maxRowNnz =
        rowNnz.empty() ? 1 : *std::max_element(rowNnz.begin(), rowNnz.end());

    PetscCall(MatCreateAIJ(PETSC_COMM_WORLD,
                           PETSC_DECIDE,
                           PETSC_DECIDE,
                           op.layout.rows,
                           op.layout.cols,
                           maxRowNnz,
                           nullptr,
                           maxRowNnz,
                           nullptr,
                           &matrix));

    PetscInt rowStart = 0;
    PetscInt rowEnd = 0;
    PetscCall(MatGetOwnershipRange(matrix, &rowStart, &rowEnd));
    for (const OperatorEntry& entry : op.entries) {
        if (entry.row >= rowStart && entry.row < rowEnd) {
            PetscCall(MatSetValue(matrix, entry.row, entry.col, entry.value, ADD_VALUES));
        }
    }

    PetscCall(MatAssemblyBegin(matrix, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(matrix, MAT_FINAL_ASSEMBLY));

    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace bgc
