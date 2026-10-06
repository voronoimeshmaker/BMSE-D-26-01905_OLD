#include <bgclib/Core/BoundaryCondition.hpp>

#include <type_traits>
#include <utility>

namespace bgc {

BoundaryCondition::BoundaryCondition(const Type type,
                                     const PetscReal alpha,
                                     const PetscReal beta,
                                     GammaValue gamma)
    : type_(type),
      alpha_(alpha),
      beta_(beta),
      gamma_(std::move(gamma)) {}

BoundaryCondition::Type BoundaryCondition::typeFrom(const PetscReal alpha,
                                                    const PetscReal beta) noexcept {
    if (beta == 0.0) {
        return Type::Dirichlet;
    }
    if (alpha == 0.0) {
        return Type::Neumann;
    }
    return Type::Robin;
}

BoundaryCondition BoundaryCondition::dirichlet(const PetscReal gamma) {
    return {Type::Dirichlet, 1.0, 0.0, gamma};
}

BoundaryCondition BoundaryCondition::neumann(const PetscReal gamma) {
    return {Type::Neumann, 0.0, 1.0, gamma};
}

BoundaryCondition BoundaryCondition::robin(const PetscReal alpha,
                                           const PetscReal beta,
                                           const PetscReal gamma) {
    return {typeFrom(alpha, beta), alpha, beta, gamma};
}

BoundaryCondition BoundaryCondition::dirichlet(GammaFunction gamma) {
    return {Type::Dirichlet, 1.0, 0.0, std::move(gamma)};
}

BoundaryCondition BoundaryCondition::neumann(GammaFunction gamma) {
    return {Type::Neumann, 0.0, 1.0, std::move(gamma)};
}

BoundaryCondition BoundaryCondition::robin(const PetscReal alpha,
                                           const PetscReal beta,
                                           GammaFunction gamma) {
    return {typeFrom(alpha, beta), alpha, beta, std::move(gamma)};
}

PetscReal BoundaryCondition::alpha() const noexcept {
    return alpha_;
}

PetscReal BoundaryCondition::beta() const noexcept {
    return beta_;
}

PetscReal BoundaryCondition::gamma(const PetscReal t) const {
    return std::visit([t](const auto& value) -> PetscReal {
        using ValueType = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<ValueType, PetscReal>) {
            return value;
        } else {
            return value(t);
        }
    }, gamma_);
}

BoundaryCondition::Type BoundaryCondition::type() const noexcept {
    return type_;
}

bool BoundaryCondition::isDirichlet() const noexcept {
    return type_ == Type::Dirichlet;
}

bool BoundaryCondition::isNeumann() const noexcept {
    return type_ == Type::Neumann;
}

bool BoundaryCondition::isRobin() const noexcept {
    return type_ == Type::Robin;
}

bool BoundaryCondition::isTimeDependent() const noexcept {
    return std::holds_alternative<GammaFunction>(gamma_);
}

} // namespace bgc
