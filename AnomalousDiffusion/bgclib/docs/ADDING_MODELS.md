# Adding Models

This guide describes how to add a model to `bgclib` without introducing an
inheritance hierarchy or copying PETSc assembly code into the model.

## 1. Create a Model Directory

Use the established layout:

```text
include/bgclib/Models/MYMODEL/Model.hpp
include/bgclib/Models/MYMODEL/Coeff.hpp
include/bgclib/Models/MYMODEL/Operator.hpp
src/Models/MYMODEL/Coeff.cpp
src/Models/MYMODEL/Operator.cpp
```

Add the new files to `bgclib/CMakeLists.txt`.

## 2. Define the Model Tag and Constants

In `Model.hpp`, define:

```cpp
namespace bgc::models::mymodel {

struct Tag {};

struct Constants {
    PetscReal parameter {0.0};
};

} // namespace bgc::models::mymodel
```

Then specialize `ModelTraits`:

```cpp
namespace bgc {

template <>
struct ModelTraits<models::mymodel::Tag> {
    using Constants = models::mymodel::Constants;

    static constexpr std::string_view id {"MYMODEL"};
    static constexpr std::string_view name {"My Model"};
    static constexpr PetscInt fieldCount {1};
    static constexpr std::array<std::string_view, fieldCount> fieldNames {"u"};

    [[nodiscard]] static constexpr bool constantsAreValid(
        const Constants& constants) noexcept {
        return constants.parameter >= 0.0;
    }
};

} // namespace bgc
```

Keep `Model.hpp` focused on identity, constants, and lightweight parsing.

## 3. Define Coefficient Data

In `Coeff.hpp`, define plain data structures for the discretized equations.
Prefer `StencilRow`, `StencilSet`, and small aggregates over PETSc objects.

Example:

```cpp
struct Coefficients {
    StencilRow first;
    StencilRow interior;
    StencilRow last;
};

[[nodiscard]] Coefficients computeCoefficients(const Grid1D& grid,
                                               const TimeConfig& time,
                                               const Constants& constants,
                                               const BoundarySet& boundaries);
```

If the model has boundary source contributions, expose them separately:

```cpp
struct RHSCoefficients {
    PetscReal first {0.0};
    PetscReal last {0.0};
};

[[nodiscard]] RHSCoefficients computeBoundaryRHS(...);
[[nodiscard]] DiscreteRHS buildBoundaryRHS(...);
```

## 4. Separate Interior and Boundary Formulas

In `Coeff.cpp`, split formulas by responsibility:

```text
computeInteriorCoefficients
applyWestBoundaryCoefficients
applyEastBoundaryCoefficients
computeBoundaryRHS
```

If the model has multiple boundary modes, split each mode explicitly:

```text
applyModeACoefficients
applyModeBCoefficients
computeModeARHS
computeModeBRHS
```

This style is important because boundary conditions often change both matrix
rows and source terms. TSOM is the reference example.

## 5. Build Generic Operators

In `Operator.cpp`, convert coefficient data into `DiscreteOperator` entries.
Boundary rows usually use absolute columns. Interior rows often use relative
offsets.

The operator builder should only emit data:

```cpp
DiscreteOperator buildOperator(const Grid1D& grid,
                               const Coefficients& coefficients);
```

Do not call:

```text
MatCreate
MatSetValue
MatAssemblyBegin
VecSetValue
KSPSolve
```

Those belong to the generic numerics layer.

## 6. Document Unknown Ordering

Every multi-field model must document its flattened ordering. The established
convention is block-by-field:

```text
[ field1_0 ... field1_{N-1}  field2_0 ... field2_{N-1} ]
```

For example:

```text
TBGC: [ phi ... mu ]
TSOM: [ U ... V ]
```

Operator builders and RHS builders must use the same ordering.

## 7. Connect to Runtime APIs

If the model should be selectable at runtime, add a `ModelOps` entry and
register it in `ModelRegistry`. Runtime registration should call model public
functions; it should not duplicate coefficient formulas.

## 8. Add MMS or Regression Tests

At minimum, add one MMS or focused regression case that checks:

- matrix/RHS dimensions;
- basic coefficient values for a small grid;
- serial and MPI execution;
- convergence of field errors;
- local truncation error;
- boundary branches when the model has several boundary modes.

For boundary-sensitive models, tests should inspect the first and last volumes
explicitly. Most mistakes in these models appear at boundary rows.

## Checklist

- Model constants validate physically meaningful ranges.
- Coefficient code has no PETSc object allocation.
- Operator code emits `DiscreteOperator` or documented block operators.
- RHS boundary terms are separate from matrix coefficients.
- Unknown ordering is documented.
- MMS drivers use `bgclib` norms, truncation error, IO, and assembly helpers.
- The model compiles and runs with `mpiexec -n 1` and `mpiexec -n 2`.

