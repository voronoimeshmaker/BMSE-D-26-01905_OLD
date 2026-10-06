// -----------------------------------------------------------------------------
//  Includes and forward declarations 
// -----------------------------------------------------------------------------

#include <petsc.h>

#include <bgclib/BGCLib.hpp>


// -----------------------------------------------------------------------------
//  Header 
// -----------------------------------------------------------------------------

PetscErrorCode RunTest();



// -----------------------------------------------------------------------------
// Minimal BGC Smoke Driver
//
// This program is intentionally small. Its purpose is not to run a physically
// meaningful case, but to validate the basic library pipeline:
//
//   Model selection
//      -> PETSc state allocation
//      -> discrete operator construction
//      -> generic matrix assembly
//      -> RHS construction
//      -> generic RHS assembly
//      -> solver configuration
//      -> linear solve
//      -> PETSc cleanup
//
// This should be the first executable built when validating the new architecture.
// It exercises the model/runtime/numerics boundary without adding MMS logic,
// file output, convergence studies, or post-processing.
// -----------------------------------------------------------------------------




int main(int argc, char** argv) {
    PetscCallAbort(PETSC_COMM_WORLD,
                   PetscInitialize(&argc, &argv, nullptr, nullptr));

    PetscCallAbort(PETSC_COMM_WORLD, RunTest());

    PetscCallAbort(PETSC_COMM_WORLD, PetscFinalize());
    return 0;
}


PetscErrorCode RunTest() {
    PetscFunctionBeginUser;

    PetscCall(PetscPrintf(
        PETSC_COMM_WORLD,
        "BGCMinimal smoke test\n"
        "  status : placeholder\n"
        "  purpose: validate PETSc initialization and program wiring\n"
    ));

// RunContext owns the high-level data needed to define a run.
//
// The numerical library remains DOD-oriented: configuration data lives in
// plain structs, while PETSc objects live in SimState.

    bgc::RunContext ctx;

    PetscFunctionReturn(PETSC_SUCCESS);
}


// int main(int argc, char** argv) {

//     std::cout << "This is a placeholder for the BGC minimal smoke test. The actual implementation is in progress." << std::endl;    
//     // PetscCallAbort(PETSC_COMM_WORLD,
//     //                PetscInitialize(&argc, &argv, nullptr, nullptr));

//     // // RunContext owns the high-level data needed to define a run.
//     // //
//     // // The numerical library remains DOD-oriented: configuration data lives in
//     // // plain structs, while PETSc objects live in SimState.
//     // bgc::RunContext ctx;

//     // // Select the model and define a tiny one-dimensional test problem.
//     // //
//     // // BGC is the first smoke test because it uses a scalar matrix. Block models
//     // // such as TBGC and TSOM should be validated after this path works.
//     // ctx.sim.model = bgc::ModelId::BGC;

//     // ctx.sim.grid.nx     = 32;
//     // ctx.sim.grid.length = 1.0;

//     // ctx.sim.time.dt        = 1.0e-3;
//     // ctx.sim.time.finalTime = 1.0e-2;

//     // // SimState owns the PETSc resources allocated during the run:
//     // // vectors, matrices, index sets, KSP objects, and field views.
//     // bgc::SimState st;

//     // // Resolve the runtime model operations.
//     // //
//     // // modelRegistry() returns a table of operations, not a virtual class
//     // // hierarchy. This keeps model dispatch explicit and compatible with the
//     // // no-inheritance DOD design.
//     // const bgc::ModelOps& model = bgc::modelRegistry().get(ctx.sim.model);

//     // // Allocate the PETSc state required by the selected model.
//     // //
//     // // For BGC this should create a scalar matrix and scalar field vectors. For
//     // // block models, the corresponding operation may create block matrices, index
//     // // sets, and field sub-vectors.
//     // PetscCallAbort(PETSC_COMM_WORLD, model.createState(ctx, st));

//     // // Build the model-specific discrete operator.
//     // //
//     // // The model computes mathematical coefficients and returns a generic
//     // // DiscreteOperator description. PETSc insertion logic stays in the generic
//     // // assembly layer.
//     // const bgc::DiscreteOperator op = model.buildOperator(ctx);

//     // // Assemble the PETSc matrix from the generic operator description.
//     // PetscCallAbort(PETSC_COMM_WORLD, bgc::assembleMatrix(ctx, st, op));

//     // // Build and assemble the right-hand side at the initial time.
//     // //
//     // // A real transient driver would repeat this inside a time loop.
//     // const PetscReal t = 0.0;
//     // const bgc::DiscreteRHS rhs = model.buildRHS(ctx, t);

//     // PetscCallAbort(PETSC_COMM_WORLD, bgc::assembleRHS(ctx, st, rhs));

//     // // Configure and solve the linear system.
//     // //
//     // // SolverConfig lives in ctx.solver, keeping PETSc solver choices separate
//     // // from model physics and numerical discretization data.
//     // PetscCallAbort(PETSC_COMM_WORLD, bgc::configureSolver(ctx.solver, st));
//     // PetscCallAbort(PETSC_COMM_WORLD, bgc::solve(st));

//     // // Release PETSc objects owned by SimState before finalizing PETSc.
//     // PetscCallAbort(PETSC_COMM_WORLD, bgc::destroyState(st));

//     // PetscCallAbort(PETSC_COMM_WORLD, PetscFinalize());
//     return 0;
// }
