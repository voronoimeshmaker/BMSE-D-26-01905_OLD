#pragma once

// PETSc vector assembly from generic DiscreteRHS descriptions.
//
// DiscreteRHS stores sparse row/value contributions emitted by model code.
// This module turns those contributions into PETSc Vec objects and provides a
// helper for copying dense C++ values into an existing Vec.

#include <vector>

#include <petsc.h>

#include <bgclib/Numerics/DiscreteRHS.hpp>

namespace bgc {

// Creates a sequential PETSc Vec of rhs.size and adds every sparse RHS entry.
//
// The returned Vec is owned by the caller and must be destroyed with
// VecDestroy. Entries are assembled with ADD_VALUES.
PetscErrorCode createSequentialRHSVector(const DiscreteRHS& rhs,
                                         Vec& vector);

// Adds sparse RHS contributions to an existing PETSc Vec.
//
// This is useful when a caller already owns the vector, for example in a
// transient step that combines mass, source, and boundary contributions.
PetscErrorCode addRHSValues(Vec vector,
                            const DiscreteRHS& rhs);

// Inserts a dense C++ vector into an already-created sequential PETSc Vec.
//
// Values are inserted with INSERT_VALUES, replacing previous entries.
PetscErrorCode copyValuesToSequentialVector(const std::vector<PetscReal>& values,
                                            Vec vector);

} // namespace bgc
