#include <bgclib/Analysis/ErrorNorms.hpp>

#include <stdexcept>
#include <string>
#include <vector>

namespace bgc {

namespace {

void checkPetsc(const PetscErrorCode ierr, const char* operation) {
    if (ierr != PETSC_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed with PETSc error " +
                                 std::to_string(static_cast<int>(ierr)));
    }
}

} // namespace

ErrorNorms computeVectorNorms(std::span<const PetscReal> values,
                              const PetscReal cellWidth) {
    std::vector<PetscScalar> scalarValues;
    scalarValues.reserve(values.size());
    for (const PetscReal value : values) {
        scalarValues.push_back(static_cast<PetscScalar>(value));
    }

    Vec vec = nullptr;
    checkPetsc(VecCreateSeqWithArray(PETSC_COMM_SELF,
                                     1,
                                     static_cast<PetscInt>(scalarValues.size()),
                                     scalarValues.data(),
                                     &vec),
               "VecCreateSeqWithArray");

    ErrorNorms norms;
    PetscReal norm12[2] = {0.0, 0.0};
    checkPetsc(VecNorm(vec, NORM_1_AND_2, norm12), "VecNorm(NORM_1_AND_2)");
    norms.l1 = norm12[0];
    norms.l2 = norm12[1];
    checkPetsc(VecNorm(vec, NORM_INFINITY, &norms.linf), "VecNorm(NORM_INFINITY)");
    checkPetsc(VecDestroy(&vec), "VecDestroy");

    // Convert the discrete vector norms to finite-volume error norms.
    norms.l1 *= cellWidth;
    norms.l2 *= PetscSqrtReal(cellWidth);
    return norms;
}

ErrorNorms computeErrorNorms(std::span<const PetscReal> numerical,
                             std::span<const PetscReal> exact,
                             const PetscReal cellWidth) {
    if (numerical.size() != exact.size()) {
        throw std::runtime_error("computeErrorNorms: field sizes differ.");
    }

    std::vector<PetscReal> error(numerical.size(), 0.0);
    for (std::size_t i = 0; i < numerical.size(); ++i) {
        error[i] = numerical[i] - exact[i];
    }
    return computeVectorNorms(error, cellWidth);
}

} // namespace bgc
