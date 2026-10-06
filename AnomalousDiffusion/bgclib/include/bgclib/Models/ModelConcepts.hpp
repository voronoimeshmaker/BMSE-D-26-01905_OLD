#pragma once

// Compile-time model contract.
//
// Models are extended with tags, ModelTraits specializations, model constants,
// and free functions found by ADL:
//
//   createState(tag, ctx, state, params)
//   buildOperator(tag, ctx, params)
//   buildRHS(tag, ctx, params, time)
//
// This keeps the numerical path statically dispatched. Runtime selection can
// still happen at the boundary by mapping input strings to concrete tags.

#include <concepts>
#include <string_view>

#include <petsc.h>

#include <bgclib/Core/RunContext.hpp>
#include <bgclib/Core/SimState.hpp>
#include <bgclib/Core/Types.hpp>
#include <bgclib/Numerics/DiscreteOperator.hpp>
#include <bgclib/Numerics/DiscreteRHS.hpp>

namespace bgc {

template <typename ModelTag>
concept ModelMetadata =
    requires(typename ModelTraits<ModelTag>::Constants constants) {
        typename ModelTraits<ModelTag>::Constants;
        { ModelTraits<ModelTag>::id } -> std::convertible_to<std::string_view>;
        { ModelTraits<ModelTag>::name } -> std::convertible_to<std::string_view>;
        { ModelTraits<ModelTag>::fieldCount } -> std::convertible_to<PetscInt>;
        { ModelTraits<ModelTag>::fieldNames };
        { ModelTraits<ModelTag>::constantsAreValid(constants) } -> std::same_as<bool>;
    };

// State builders allocate or attach PETSc resources required by the model.
// Destruction is intentionally a separate runtime concern so ownership rules
// stay explicit.
template <typename ModelTag>
concept ModelStateBuilder =
    ModelMetadata<ModelTag> &&
    requires(ModelTag tag,
             const RunContext& ctx,
             SimState& state,
             const typename ModelTraits<ModelTag>::Constants& constants) {
        { createState(tag, ctx, state, constants) } -> std::same_as<PetscErrorCode>;
    };

// Operator builders return model-independent sparse data. They should not call
// MatSetValue directly; that belongs to the generic PETSc assembly layer.
template <typename ModelTag>
concept ModelOperatorBuilder =
    ModelMetadata<ModelTag> &&
    requires(ModelTag tag,
             const RunContext& ctx,
             const typename ModelTraits<ModelTag>::Constants& constants) {
        { buildOperator(tag, ctx, constants) } -> std::same_as<DiscreteOperator>;
    };

// RHS builders return model-independent vector contributions for a given time.
template <typename ModelTag>
concept ModelRHSBuilder =
    ModelMetadata<ModelTag> &&
    requires(ModelTag tag,
             const RunContext& ctx,
             const typename ModelTraits<ModelTag>::Constants& constants,
             PetscReal time) {
        { buildRHS(tag, ctx, constants, time) } -> std::same_as<DiscreteRHS>;
    };

// FullModel is the minimal contract for a model that can enter the solve path.
template <typename ModelTag>
concept FullModel =
    ModelStateBuilder<ModelTag> &&
    ModelOperatorBuilder<ModelTag> &&
    ModelRHSBuilder<ModelTag>;

// Backward-compatible name for metadata-only checks.
template <typename ModelTag>
concept ModelTagLike = ModelMetadata<ModelTag>;

template <ModelMetadata ModelTag>
[[nodiscard]] constexpr std::string_view modelId() noexcept {
    return ModelTraits<ModelTag>::id;
}

template <ModelMetadata ModelTag>
[[nodiscard]] constexpr PetscInt modelFieldCount() noexcept {
    return ModelTraits<ModelTag>::fieldCount;
}

template <ModelMetadata ModelTag>
using ModelConstants = typename ModelTraits<ModelTag>::Constants;

} // namespace bgc
