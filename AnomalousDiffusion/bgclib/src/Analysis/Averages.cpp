#include <bgclib/Analysis/Averages.hpp>


namespace bgc {

FieldStatistics computeFieldStatistics(std::span<const PetscReal> values,
                                       const PetscReal cellWidth) {
    FieldStatistics stats;
    if (values.empty()) {
        return stats;
    }

    stats.minimum = values.front();
    stats.maximum = values.front();
    PetscReal sum = 0.0;
    for (const PetscReal value : values) {
        sum += value;
        stats.minimum = PetscMin(stats.minimum, value);
        stats.maximum = PetscMax(stats.maximum, value);
    }

    stats.integral = cellWidth * sum;
    stats.average = sum / static_cast<PetscReal>(values.size());
    return stats;
}

PetscReal computeCenteredRawMoment2(std::span<const PetscReal> values,
                                    const PetscReal cellWidth,
                                    const PetscReal x0,
                                    const PetscReal center) {
    PetscReal moment = 0.0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        const PetscReal x = x0 + (static_cast<PetscReal>(i) + 0.5) * cellWidth;
        const PetscReal dx = x - center;
        moment += dx * dx * values[i];
    }
    return cellWidth * moment;
}

CenteredMomentDiagnostics computeCenteredMomentDiagnostics(
    std::span<const PetscReal> values,
    const PetscReal cellWidth,
    const PetscReal x0,
    const PetscReal center) {
    CenteredMomentDiagnostics diag;
    if (values.empty()) {
        return diag;
    }

    const FieldStatistics stats = computeFieldStatistics(values, cellWidth);
    diag.mass = stats.integral;
    diag.moment2Raw = computeCenteredRawMoment2(values, cellWidth, x0, center);
    diag.moment2 = PetscAbsReal(diag.mass) > 1.0e-30 ? diag.moment2Raw / diag.mass : 0.0;
    diag.minimum = stats.minimum;
    diag.maximum = stats.maximum;
    diag.boundaryWest = values.size() >= 2 ? values[1] : 0.0;
    diag.boundaryEast = values.size() >= 2 ? values[values.size() - 2] : 0.0;
    return diag;
}

} // namespace bgc
