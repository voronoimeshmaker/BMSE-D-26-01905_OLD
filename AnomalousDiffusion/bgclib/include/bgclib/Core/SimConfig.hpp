#pragma once

// -----------------------------------------------------------------------------
// SimConfig.hpp
//
// High-level physical configuration for a simulation.
//
// SimConfig groups the model choice, mesh, time discretization, and boundary
// condition declarations. It deliberately does not contain PETSc objects,
// assembled coefficients, output paths, or solver tolerances. Those belong to
// SimState, model operators, OutputSpec, and SolverConfig respectively.
// -----------------------------------------------------------------------------

#include <bgclib/Core/BoundarySet.hpp>
#include <bgclib/Core/Grid1D.hpp>
#include <bgclib/Core/ModelId.hpp>
#include <bgclib/Core/TimeConfig.hpp>

namespace bgc {

struct SimConfig {
    // Model family selected by the input file, for example BGC, TBGC, or TSOM.
    ModelId modelId {"BGC"};

    // One-dimensional finite-volume grid shared by the current models.
    Grid1D grid;

    // Time controls. Steady MMS problems can still carry a default TimeConfig.
    TimeConfig time;

    // Boundary declarations for all model fields.
    BoundarySet boundaries;
};

} // namespace bgc
