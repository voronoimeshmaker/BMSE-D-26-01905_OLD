#include <bgclib/Analysis/Thermodynamics.hpp>

namespace bgc {

PetscReal computeSquaredGradientSum(std::span<const PetscReal> values) {
    PetscReal sum = 0.0;
    for (std::size_t i = 0; i + 1 < values.size(); ++i) {
        const PetscReal diff = values[i + 1] - values[i];
        sum += diff * diff;
    }
    return sum;
}

PetscReal computeQuadraticFreeEnergy(std::span<const PetscReal> values,
                                     const PetscReal cellWidth,
                                     const PetscReal gradientCoefficient) {
    PetscReal bulk = 0.0;
    for (const PetscReal value : values) {
        bulk += value * value;
    }

    const PetscReal gradient = computeSquaredGradientSum(values);
    return 0.5 * (cellWidth * bulk + gradientCoefficient * gradient / cellWidth);
}

} // namespace bgc
