#include <bgclib/Numerics/Assembly/GenericRHSAssembly.hpp>

// RHS assembly is intentionally additive: different pieces of a model can emit
// contributions to the same row, and PETSc combines them during Vec assembly.
// Dense value insertion is kept as a separate helper for solution/source arrays.
//
// Parallel note: callers may provide a complete global RHS description on every
// rank. This module inserts only rows owned by the current PETSc vector,
// avoiding duplicated values in MPI runs.

namespace bgc {

PetscErrorCode createSequentialRHSVector(const DiscreteRHS& rhs, Vec& vector) {
    PetscFunctionBeginUser;

    PetscCall(VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, rhs.size, &vector));
    PetscCall(addRHSValues(vector, rhs));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode addRHSValues(Vec vector, const DiscreteRHS& rhs) {
    PetscFunctionBeginUser;

    PetscInt rowStart = 0;
    PetscInt rowEnd = 0;
    PetscCall(VecGetOwnershipRange(vector, &rowStart, &rowEnd));
    for (const RHSEntry& entry : rhs.entries) {
        if (entry.row >= rowStart && entry.row < rowEnd) {
            PetscCall(VecSetValue(vector, entry.row, entry.value, ADD_VALUES));
        }
    }

    PetscCall(VecAssemblyBegin(vector));
    PetscCall(VecAssemblyEnd(vector));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode copyValuesToSequentialVector(const std::vector<PetscReal>& values, Vec vector) {
    PetscFunctionBeginUser;

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

} // namespace bgc
