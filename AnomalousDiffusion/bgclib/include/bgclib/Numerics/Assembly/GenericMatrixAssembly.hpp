#pragma once

// PETSc matrix assembly from generic DiscreteOperator descriptions.
//
// Models emit DiscreteOperator as sparse triplets in global coordinates. This
// module is the single model-independent place that turns those triplets into a
// PETSc Mat. Keeping this here prevents BGC, TBGC, TSOM, MMS programs, and
// analysis routines from each owning their own matrix assembly details.

#include <petsc.h>

#include <bgclib/Numerics/DiscreteOperator.hpp>

namespace bgc {

// Creates and assembles a PETSc AIJ matrix from a DiscreteOperator.
//
// Entries are added with ADD_VALUES so repeated triplets contribute to the same
// matrix position. The returned Mat is owned by the caller and must be destroyed
// with MatDestroy. In parallel, each rank inserts only entries whose row belongs
// to its PETSc ownership range.
PetscErrorCode createSequentialMatrix(const DiscreteOperator& op,
                                      Mat& matrix);

} // namespace bgc
