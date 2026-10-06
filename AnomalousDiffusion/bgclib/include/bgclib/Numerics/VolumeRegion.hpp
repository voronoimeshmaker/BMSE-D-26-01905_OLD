#pragma once

// Volume-region classification for 1D boundary, near-boundary, and interior volumes.
//
// The BGC-family fourth-order stencil has four special boundary-adjacent
// equations and one reusable interior equation.

#include <petsc.h>

namespace bgc {

enum class VolumeRegion {
    First,
    Second,
    Interior,
    SecondToLast,
    Last,
};

[[nodiscard]] constexpr VolumeRegion classifyVolume(const PetscInt i,
                                                    const PetscInt n) noexcept {
    if (i == 0) {
        return VolumeRegion::First;
    }
    if (i == 1) {
        return VolumeRegion::Second;
    }
    if (i == n - 2) {
        return VolumeRegion::SecondToLast;
    }
    if (i == n - 1) {
        return VolumeRegion::Last;
    }
    return VolumeRegion::Interior;
}

} // namespace bgc
