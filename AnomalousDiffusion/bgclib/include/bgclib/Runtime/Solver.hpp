#pragma once

// PETSc solver setup and solve routines driven by SolverConfig.
//
// This module is intended to centralize KSP/PC policy for production drivers:
// direct vs iterative solves, tolerances, preconditioners, and PETSc option
// overrides. MMS helper code currently has a small specialized reusable solver
// because it needs a compact path for fixed linear systems. The long-term goal
// is to move that policy here once the runtime layer is broad enough.
//
// Solver code belongs here or in Numerics/Assembly, not in model coefficient
// files. Models describe matrices; solver modules decide how to solve them.
