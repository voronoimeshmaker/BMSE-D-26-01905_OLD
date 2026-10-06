#pragma once

// Manufactured source wrapper used by MMS tests and convergence studies.
//
// The stored evaluator is a concrete callable, not a virtual interface. It must
// accept the model constants plus the point (x, t):
//
//   evaluator(constants, x, t) -> PetscReal
//
// ExactSolution and ManufacturedSource intentionally share this shape so tests
// can pair an analytical field with the source term produced from the same
// model constants.

#include <concepts>
#include <type_traits>
#include <utility>

#include <petsc.h>

#include <bgclib/Models/ModelConcepts.hpp>

namespace bgc {

template <typename SourceEvaluator, typename Constants>
concept ManufacturedSourceEvaluator =
    requires(const SourceEvaluator& evaluator,
             const Constants& constants,
             PetscReal x,
             PetscReal t) {
        { evaluator(constants, x, t) } -> std::convertible_to<PetscReal>;
    };

template <ModelMetadata ModelTag, typename SourceEvaluator>
requires ManufacturedSourceEvaluator<SourceEvaluator, ModelConstants<ModelTag>>
class ManufacturedSource {
public:
    using Constants = ModelConstants<ModelTag>;

    constexpr ManufacturedSource(Constants constants, SourceEvaluator evaluator)
        : constants_ {std::move(constants)},
          evaluator_ {std::move(evaluator)} {}

    [[nodiscard]] constexpr PetscReal operator()(const PetscReal x,
                                                 const PetscReal t) const {
        return static_cast<PetscReal>(evaluator_(constants_, x, t));
    }

    [[nodiscard]] constexpr const Constants& constants() const noexcept {
        return constants_;
    }

private:
    Constants       constants_;
    SourceEvaluator evaluator_;
};

template <ModelMetadata ModelTag, typename SourceEvaluator>
ManufacturedSource(ModelConstants<ModelTag>, SourceEvaluator)
    -> ManufacturedSource<ModelTag, SourceEvaluator>;

} // namespace bgc
