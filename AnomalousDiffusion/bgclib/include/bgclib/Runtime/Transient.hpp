#pragma once

// Optional high-level transient runner for standard time-marching workflows.
//
// A standard transient runner should own the orchestration common to physical
// simulations: allocate state, assemble or reuse operators according to
// NumericsConfig, update RHS values, call the solver, run diagnostics, and write
// requested output. It should not contain model formulas.
//
// Current MMS programs still drive their own time loops because they also build
// manufactured exact fields and truncation-error diagnostics. Shared production
// workflows should migrate here as the runtime API matures.
