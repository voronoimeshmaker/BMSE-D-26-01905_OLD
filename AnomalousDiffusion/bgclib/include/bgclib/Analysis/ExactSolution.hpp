#pragma once

// Exact-solution wrapper used by tests, manufactured solutions, and analysis.
//
// The stored evaluator is a concrete callable, not a virtual interface. It must
// accept the model constants plus the point (x, t):
//
//   evaluator(constants, x, t) -> PetscReal
//
// This gives model code flexibility while keeping calls statically dispatched.

#include <concepts>
#include <type_traits>
#include <utility>

#include <petsc.h>

#include <bgclib/Models/ModelConcepts.hpp>

namespace bgc {

template <typename FieldEvaluator, typename Constants>
concept ExactFieldEvaluator =
    requires(const FieldEvaluator& evaluator,
             const Constants& constants,
             PetscReal x,
             PetscReal t) {
        { evaluator(constants, x, t) } -> std::convertible_to<PetscReal>;
    };

template <ModelMetadata ModelTag, typename FieldEvaluator>
requires ExactFieldEvaluator<FieldEvaluator, ModelConstants<ModelTag>>
class ExactSolution {
public:
    using Constants = ModelConstants<ModelTag>;

    constexpr ExactSolution(Constants constants, FieldEvaluator evaluator)
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
    Constants      constants_;
    FieldEvaluator evaluator_;
};

template <ModelMetadata ModelTag, typename FieldEvaluator>
ExactSolution(ModelConstants<ModelTag>, FieldEvaluator)
    -> ExactSolution<ModelTag, FieldEvaluator>;

} // namespace bgc
