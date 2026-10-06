#include <bgclib/Analysis/FieldComparison.hpp>

#include <stdexcept>

namespace bgc {

std::vector<FieldComparisonPoint> compareFields(std::span<const PetscReal> numerical,
                                                std::span<const PetscReal> exact,
                                                const PetscReal cellWidth,
                                                const PetscReal x0) {
    if (numerical.size() != exact.size()) {
        throw std::runtime_error("compareFields: field sizes differ.");
    }

    std::vector<FieldComparisonPoint> points;
    points.reserve(numerical.size());
    for (std::size_t i = 0; i < numerical.size(); ++i) {
        const PetscReal x = x0 + (static_cast<PetscReal>(i) + 0.5) * cellWidth;
        points.push_back({
            .index = static_cast<PetscInt>(i),
            .x = x,
            .numerical = numerical[i],
            .exact = exact[i],
            .error = numerical[i] - exact[i],
        });
    }
    return points;
}

} // namespace bgc
