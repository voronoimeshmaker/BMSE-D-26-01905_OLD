#pragma once

// Boundary condition collection for the west/east sides of a 1D domain.

#include <array>
#include <cstddef>

#include <bgclib/Core/BoundaryCondition.hpp>

namespace bgc {

struct BoundarySide {
    // BGC-like fourth-order models need two independent boundary equations at
    // each side. Defaults match the old library convention.
    std::array<BoundaryCondition, 2> conditions {
        BoundaryCondition::dirichlet(0.0),
        BoundaryCondition::neumann(0.0),
    };

    [[nodiscard]] const BoundaryCondition& operator[](std::size_t index) const noexcept {
        return conditions[index];
    }

    [[nodiscard]] BoundaryCondition& operator[](std::size_t index) noexcept {
        return conditions[index];
    }
};

struct BoundarySet {
    BoundarySide west;
    BoundarySide east;
};

} // namespace bgc
