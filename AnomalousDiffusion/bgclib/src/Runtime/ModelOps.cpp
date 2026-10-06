// Runtime model operation tables are currently described by public headers.
//
// This translation unit is intentionally kept in the build as the stable home
// for future non-template code that maps a runtime ModelId to the concrete
// functions used to compute coefficients, assemble operators, and dispatch model
// diagnostics. Keeping that responsibility out of programs is important for the
// long-term goal of selecting BGC/TBGC/TSOM entirely from input files.
