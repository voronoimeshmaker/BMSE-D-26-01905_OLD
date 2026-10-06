#pragma once

// Runtime diagnostics configuration.
//
// DiagnosticsConfig controls extra checking and reporting requested by a
// driver. It should not change the mathematical model. Use it for visibility:
// progress messages, invalid-value checks, thermodynamic diagnostics, solver
// statistics, or small-system matrix dumps.

namespace bgc {

struct DiagnosticsConfig {
    // Print high-level progress messages.
    bool verbose             {true};

    // Check vectors/fields for invalid numerical values when supported.
    bool checkInvalidValues  {true};

    // Run energy/free-energy consistency diagnostics when available.
    bool checkThermodynamics {true};

    // Print KSP reason, iteration count, and related solver information.
    bool printSolverStats    {true};

    // Dump assembled linear systems for small debug runs.
    bool printLinearSystem   {false};
};

} // namespace bgc
