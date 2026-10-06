#include <bgclib/Analysis/PostProcess.hpp>

namespace bgc {

std::vector<PetscInt> makeUniformSampleSteps(const PetscInt finalStep,
                                             const PetscInt sampleCount) {
    if (sampleCount <= 0) {
        return {};
    }

    std::vector<PetscInt> steps(static_cast<std::size_t>(sampleCount), 0);
    if (sampleCount == 1) {
        return steps;
    }

    for (PetscInt k = 1; k < sampleCount; ++k) {
        const PetscReal fraction =
            static_cast<PetscReal>(k) / static_cast<PetscReal>(sampleCount - 1);
        steps[static_cast<std::size_t>(k)] =
            PetscMin(static_cast<PetscInt>(fraction * finalStep + 0.5), finalStep);
    }
    return steps;
}

} // namespace bgc
