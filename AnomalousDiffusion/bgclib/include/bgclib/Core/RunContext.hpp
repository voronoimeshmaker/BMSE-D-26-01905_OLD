#pragma once

// -----------------------------------------------------------------------------
// RunContext.hpp
//
// DOD aggregate for data that defines one executable run.
//
// RunContext does not own PETSc objects. PETSc vectors, matrices, solvers,
// index sets, and field views belong to SimState.
//
// This struct only groups configuration data used by programs and runtime
// helpers. It should not parse files, allocate PETSc objects, assemble matrices,
// solve systems, or write output.
// -----------------------------------------------------------------------------

#include <bgclib/Core/DiagnosticsConfig.hpp>
#include <bgclib/Core/NumericsConfig.hpp>
#include <bgclib/Core/SimConfig.hpp>
#include <bgclib/Core/SolverConfig.hpp>
#include <bgclib/IO/OutputSpec.hpp>

namespace bgc {

struct RunContext {
    SimConfig         sim;
    SolverConfig      solver;
    NumericsConfig    numerics;
    DiagnosticsConfig diagnostics;
    OutputSpec        output;
};

} // namespace bgc
