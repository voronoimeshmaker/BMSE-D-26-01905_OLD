#pragma once

// -----------------------------------------------------------------------------
// SolverConfig.hpp
//
// PETSc linear solver configuration.
//
// This struct stores solver choices and tolerances only. It does not own PETSc
// KSP or PC objects.
// -----------------------------------------------------------------------------

#include <petsc.h>

namespace bgc {

enum class LinearSolverKind {
    Direct,
    Iterative,
};

enum class DirectSolverBackend {
    PetscDefault,
    Mumps,
};

enum class IterativeSolverKind {
    GMRES,
    FGMRES,
    CG,
    BiCGStab,
};

enum class PreconditionerKind {
    None,
    Jacobi,
    ILU,
    LU,
    FieldSplit,
};

struct SolverConfig {
    LinearSolverKind    kind           {LinearSolverKind::Direct};
    DirectSolverBackend directBackend  {DirectSolverBackend::Mumps};
    IterativeSolverKind iterativeKind  {IterativeSolverKind::GMRES};
    PreconditionerKind  preconditioner {PreconditionerKind::ILU};

    PetscReal rtol   {1.0e-10};
    PetscReal atol   {PETSC_DEFAULT};
    PetscReal dtol   {PETSC_DEFAULT};
    PetscInt  maxIts {PETSC_DEFAULT};

    bool setFromOptions    {true};
    bool printConvergence  {false};
    bool printLinearSystem {false};
};

} // namespace bgc
