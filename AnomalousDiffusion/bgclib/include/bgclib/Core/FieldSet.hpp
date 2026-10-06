#pragma once

// DOD container for PETSc Vec handles grouped by field, history, and analytic values.
//
// This file marks the intended home for runtime field ownership. A future
// FieldSet should group PETSc Vec handles such as current solution, previous
// solution, exact solution, source, and error views without giving models direct
// ownership of solver policy.
//
// Current MMS drivers use plain std::vector<PetscReal> for exact and numerical
// arrays because that keeps manufactured-solution output simple. Production
// runtime paths should move shared PETSc Vec grouping here.
