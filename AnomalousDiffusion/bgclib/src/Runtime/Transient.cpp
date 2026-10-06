// Transient execution placeholder.
//
// Time-marching logic should eventually be centralized here: advancing SimState,
// requesting model operators/source terms for each step, writing history output,
// and applying diagnostics. The model layer should continue to provide formulas,
// while this runtime layer owns the execution schedule.
