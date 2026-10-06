#pragma once

// Shared PETSc vector helpers used by generic assembly and analysis routines.
//
// This file owns small model-independent conversions between plain C++ arrays
// and PETSc Vec objects. Model code should continue to produce plain
// data structures such as DiscreteOperator and DiscreteRHS; PETSc allocation,
// insertion, assembly, and extraction live here or in the neighbouring generic
// assembly modules.

#include <vector>

#include <petsc.h>

namespace bgc {

// Creates and assembles a PETSc Vec containing the supplied global values.
//
// The returned Vec is owned by the caller and must be destroyed with
// VecDestroy. Values are inserted only by the rank that owns each row, avoiding
// duplicated insertion when running with more than one MPI process.
PetscErrorCode createSequentialVector(const std::vector<PetscReal>& values,
                                      Vec& vector);

// Copies all entries from a PETSc Vec into a std::vector on every rank.
//
// The destination vector is resized to the Vec global size. Internally this
// uses VecScatterCreateToAll, so MMS drivers that still keep plain global
// arrays can run with distributed PETSc objects.
PetscErrorCode copySequentialVector(Vec vector,
                                    std::vector<PetscReal>& values);

} // namespace bgc
