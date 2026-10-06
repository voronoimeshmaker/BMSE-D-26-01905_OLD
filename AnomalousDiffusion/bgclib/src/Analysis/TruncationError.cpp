#include <bgclib/Analysis/TruncationError.hpp>

#include <bgclib/Numerics/Assembly/AssemblyUtils.hpp>
#include <bgclib/Numerics/Assembly/GenericMatrixAssembly.hpp>

// Truncation-error analysis consumes the generic assembly API; it no longer
// owns matrix/vector construction. That keeps this module focused on the
// mathematical residual A*u_exact - source.

namespace bgc {

PetscErrorCode computeLocalTruncationError(const DiscreteOperator& op,
                                           const std::vector<PetscReal>& exactValues,
                                           const std::vector<PetscReal>& sourceValues,
                                           std::vector<PetscReal>& tauValues) {
    PetscFunctionBeginUser;

    Mat matrix = nullptr;
    Vec exact = nullptr;
    Vec source = nullptr;
    Vec tau = nullptr;

    PetscCall(createSequentialMatrix(op, matrix));
    PetscCall(createSequentialVector(exactValues, exact));
    PetscCall(createSequentialVector(sourceValues, source));
    PetscCall(VecDuplicate(source, &tau));
    PetscCall(MatMult(matrix, exact, tau));
    PetscCall(VecAXPY(tau, -1.0, source));
    PetscCall(copySequentialVector(tau, tauValues));

    PetscCall(VecDestroy(&tau));
    PetscCall(VecDestroy(&source));
    PetscCall(VecDestroy(&exact));
    PetscCall(MatDestroy(&matrix));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode writePetscDenseMatrixView(const std::filesystem::path& filePath,
                                         const DiscreteOperator& op) {
    PetscFunctionBeginUser;

    Mat matrix = nullptr;
    PetscViewer viewer = nullptr;

    PetscCall(createSequentialMatrix(op, matrix));
    PetscCall(PetscViewerASCIIOpen(PETSC_COMM_SELF, filePath.string().c_str(), &viewer));
    PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_DENSE));
    PetscCall(MatView(matrix, viewer));
    PetscCall(PetscViewerPopFormat(viewer));
    PetscCall(PetscViewerDestroy(&viewer));
    PetscCall(MatDestroy(&matrix));

    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace bgc
