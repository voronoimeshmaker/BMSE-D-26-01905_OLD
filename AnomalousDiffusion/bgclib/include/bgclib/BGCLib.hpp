#pragma once

// -----------------------------------------------------------------------------
// BGCLib.hpp
//
// Convenience public header for bgclib.
//
// Use this header in small programs, smoke tests, examples, and research drivers
// when a single include is more convenient than fine-grained module includes.
//
// For larger programs, prefer including only the module headers actually needed.
// This reduces rebuild time and makes dependencies more explicit.
//
// External dependencies:
//   PETSc  - required by the numerical core.
//   MPI    - required through PETSc/MPI-enabled execution.
//   YAML   - not part of the numerical core. YAML readers belong to programs/
//            or to optional IO adapters.
//
// Design conventions:
//   - DOD-oriented data structures.
//   - No inheritance-based model hierarchy.
//   - No pure virtual model interface.
//   - Runtime model dispatch uses registries and operation tables.
//   - Model code builds generic discrete operators and RHS descriptions.
//   - Generic assembly code owns PETSc Mat/Vec insertion details.
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// Core
// -----------------------------------------------------------------------------

#include <bgclib/Core/BoundaryCondition.hpp>
#include <bgclib/Core/BoundarySet.hpp>
#include <bgclib/Core/DiagnosticsConfig.hpp>
#include <bgclib/Core/FieldId.hpp>
#include <bgclib/Core/FieldSet.hpp>
#include <bgclib/Core/Grid1D.hpp>
#include <bgclib/Core/MatrixLayout.hpp>
#include <bgclib/Core/ModelId.hpp>
#include <bgclib/Core/NumericsConfig.hpp>
#include <bgclib/Core/RunContext.hpp>
#include <bgclib/Core/SimConfig.hpp>
#include <bgclib/Core/SimState.hpp>
#include <bgclib/Core/SolverConfig.hpp>
#include <bgclib/Core/TimeConfig.hpp>
#include <bgclib/Core/Types.hpp>

// -----------------------------------------------------------------------------
// Numerics
// -----------------------------------------------------------------------------

#include <bgclib/Numerics/Assembly/AssemblyUtils.hpp>
#include <bgclib/Numerics/Assembly/GenericMatrixAssembly.hpp>
#include <bgclib/Numerics/Assembly/GenericRHSAssembly.hpp>
#include <bgclib/Numerics/DiscreteOperator.hpp>
#include <bgclib/Numerics/DiscreteRHS.hpp>
#include <bgclib/Numerics/Stencil.hpp>
#include <bgclib/Numerics/VolumeRegion.hpp>

// -----------------------------------------------------------------------------
// Models
// -----------------------------------------------------------------------------

#include <bgclib/Models/BGC/Coeff.hpp>
#include <bgclib/Models/BGC/MMS.hpp>
#include <bgclib/Models/BGC/Model.hpp>
#include <bgclib/Models/BGC/Operator.hpp>
#include <bgclib/Models/TBGC/Coeff.hpp>
#include <bgclib/Models/TBGC/Model.hpp>
#include <bgclib/Models/TBGC/Operator.hpp>
#include <bgclib/Models/TSOM/Coeff.hpp>
#include <bgclib/Models/TSOM/Model.hpp>
#include <bgclib/Models/TSOM/Operator.hpp>
#include <bgclib/Models/TSOMPsiV/Coeff.hpp>
#include <bgclib/Models/TSOMPsiV/Model.hpp>
#include <bgclib/Models/TSOMPsiV/Operator.hpp>
#include <bgclib/Models/ModelConcepts.hpp>

// -----------------------------------------------------------------------------
// Runtime
// -----------------------------------------------------------------------------

#include <bgclib/Runtime/ModelOps.hpp>
#include <bgclib/Runtime/ModelRegistry.hpp>
#include <bgclib/Runtime/Solver.hpp>
#include <bgclib/Runtime/Transient.hpp>

// -----------------------------------------------------------------------------
// Analysis
// -----------------------------------------------------------------------------

#include <bgclib/Analysis/Averages.hpp>
#include <bgclib/Analysis/ErrorNorms.hpp>
#include <bgclib/Analysis/ExactSolution.hpp>
#include <bgclib/Analysis/FieldComparison.hpp>
#include <bgclib/Analysis/MMS.hpp>
#include <bgclib/Analysis/ManufacturedSource.hpp>
#include <bgclib/Analysis/PostProcess.hpp>
#include <bgclib/Analysis/RunSummary.hpp>
#include <bgclib/Analysis/Thermodynamics.hpp>
#include <bgclib/Analysis/TruncationError.hpp>

// -----------------------------------------------------------------------------
// IO
// -----------------------------------------------------------------------------

#include <bgclib/IO/ConfigReader.hpp>
#include <bgclib/IO/OutputPaths.hpp>
#include <bgclib/IO/OutputSpec.hpp>
#include <bgclib/IO/ResultWriter.hpp>
