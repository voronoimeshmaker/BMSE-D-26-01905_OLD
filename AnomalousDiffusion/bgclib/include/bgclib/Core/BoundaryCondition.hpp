#pragma once

// Boundary condition data structures and helper constructors.
//
// A boundary condition is stored as a value object:
//
//     alpha * phi + beta * dphi/dx = gamma(t)
//
// The model layer reads alpha, beta, and gamma(t) directly. No inheritance or
// virtual dispatch is used; time-dependent gamma values are stored as callables.

#include <functional>
#include <optional>
#include <variant>

#include <petsc.h>

namespace bgc {

class BoundaryCondition {
public:
    enum class Type {
        Dirichlet,
        Neumann,
        Robin,
    };

    using GammaFunction = std::function<PetscReal(PetscReal)>;
    using GammaValue = std::variant<PetscReal, GammaFunction>;

    [[nodiscard]] static BoundaryCondition dirichlet(PetscReal gamma);
    [[nodiscard]] static BoundaryCondition neumann(PetscReal gamma);
    [[nodiscard]] static BoundaryCondition robin(PetscReal alpha,
                                                 PetscReal beta,
                                                 PetscReal gamma);

    [[nodiscard]] static BoundaryCondition dirichlet(GammaFunction gamma);
    [[nodiscard]] static BoundaryCondition neumann(GammaFunction gamma);
    [[nodiscard]] static BoundaryCondition robin(PetscReal alpha,
                                                 PetscReal beta,
                                                 GammaFunction gamma);

    [[nodiscard]] PetscReal alpha() const noexcept;
    [[nodiscard]] PetscReal beta() const noexcept;
    [[nodiscard]] PetscReal gamma(PetscReal t = 0.0) const;

    [[nodiscard]] Type type() const noexcept;
    [[nodiscard]] bool isDirichlet() const noexcept;
    [[nodiscard]] bool isNeumann() const noexcept;
    [[nodiscard]] bool isRobin() const noexcept;
    [[nodiscard]] bool isTimeDependent() const noexcept;

private:
    BoundaryCondition(Type type,
                      PetscReal alpha,
                      PetscReal beta,
                      GammaValue gamma);

    [[nodiscard]] static Type typeFrom(PetscReal alpha, PetscReal beta) noexcept;

    Type type_ {Type::Dirichlet};
    PetscReal alpha_ {1.0};
    PetscReal beta_ {0.0};
    GammaValue gamma_ {0.0};
};

} // namespace bgc
