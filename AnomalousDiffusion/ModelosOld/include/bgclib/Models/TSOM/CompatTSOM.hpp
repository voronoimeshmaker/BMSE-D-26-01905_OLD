#pragma once

#include <bgclib/SimConfig.hpp>
#include <bgclib/SimState.hpp>

namespace bgc { 

// -----------------------------------------------------------------------------
// Current TSOM implementation reuses the same PETSc 2x2 state layout already
// used by TSPV. This compatibility wrapper allows TSOM programs to call a
// TSOM-named factory without requiring immediate changes in the central state
// allocation infrastructure of the library.
// -----------------------------------------------------------------------------
inline PetscErrorCode createStateTSOM(const SimConfig& cfg, SimState& st) {
    return createStateTSPV(cfg, st);
}


} // namespace bgc
