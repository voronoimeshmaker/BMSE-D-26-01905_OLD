#include <bgclib/Numerics/Assembly/AssemblyUtils.hpp>

// This implementation intentionally stays model-agnostic: it only knows how to
// move values between std::vector<PetscReal> and PETSc Vec objects.
// Higher-level assembly decisions belong to GenericMatrixAssembly and
// GenericRHSAssembly.

namespace bgc {

PetscErrorCode createSequentialVector(const std::vector<PetscReal>& values, Vec& vector) {
    PetscFunctionBeginUser;

    PetscCall(VecCreateMPI(PETSC_COMM_WORLD,
                           PETSC_DECIDE,
                           static_cast<PetscInt>(values.size()),
                           &vector));

    PetscInt rowStart = 0;
    PetscInt rowEnd = 0;
    PetscCall(VecGetOwnershipRange(vector, &rowStart, &rowEnd));
    for (PetscInt i = rowStart; i < rowEnd; ++i) {
        PetscCall(VecSetValue(vector, i, values[static_cast<std::size_t>(i)], INSERT_VALUES));
    }

    PetscCall(VecAssemblyBegin(vector));
    PetscCall(VecAssemblyEnd(vector));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode copySequentialVector(Vec vector, std::vector<PetscReal>& values) {
    PetscFunctionBeginUser;

    Vec allValues = nullptr;
    VecScatter scatter = nullptr;
    PetscInt size = 0;
    PetscCall(VecGetSize(vector, &size));
    values.assign(static_cast<std::size_t>(size), 0.0);

    PetscCall(VecScatterCreateToAll(vector, &scatter, &allValues));
    PetscCall(VecScatterBegin(scatter, vector, allValues, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(scatter, vector, allValues, INSERT_VALUES, SCATTER_FORWARD));

    const PetscScalar* array = nullptr;
    PetscCall(VecGetArrayRead(allValues, &array));
    for (PetscInt i = 0; i < size; ++i) {
        values[static_cast<std::size_t>(i)] = static_cast<PetscReal>(array[i]);
    }
    PetscCall(VecRestoreArrayRead(allValues, &array));
    PetscCall(VecScatterDestroy(&scatter));
    PetscCall(VecDestroy(&allValues));

    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace bgc
