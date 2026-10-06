#pragma once

// -----------------------------------------------------------------------------
// BGCLib.hpp
//
// Convenient public header for bgclib.
//
// This header is intended for use in small programs, smoke tests, examples,
// and research drivers where a single include is more convenient than using
// fine-grained module headers.
//
// For larger applications, it is recommended to include only the specific
// module headers that are required. This helps reduce rebuild times and makes
// dependencies more explicit.
//
// External dependencies:
//   PETSc  - required by the numerical core.
//   MPI    - required through PETSc/MPI-enabled execution.
//   YAML   - not part of the numerical core. YAML readers belong in programs/
//            or in optional IO adapters.
//
// Design conventions:
//   - Data-oriented (DOD) data structures.
//   - No inheritance-based model hierarchy.
//   - No pure virtual model interface.
//   - Runtime model dispatch is handled via registries and operation tables.
//   - Model code constructs generic discrete operators and RHS descriptions.
//   - Generic assembly code is responsible for PETSc Mat/Vec insertion details.
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