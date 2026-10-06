#include <bgclib/Core/BoundaryCondition.hpp>

namespace bgc {

// ---------------------------------------------------------------------------
//  Constructor privado
// ---------------------------------------------------------------------------

BoundaryCondition::BoundaryCondition(Type      t,
                                     PetscReal alpha,
                                     PetscReal beta,
                                     GammaVar  gamma)
    : type_(t), alpha_(alpha), beta_(beta), gamma_(std::move(gamma))
{}

// ---------------------------------------------------------------------------
//  Helper privado
// ---------------------------------------------------------------------------

BoundaryCondition::Type
BoundaryCondition::typeFrom(PetscReal alpha, PetscReal beta) {
    if (beta  == 0.0) return Type::Dirichlet;
    if (alpha == 0.0) return Type::Neumann;
    return Type::Robin;
}

// ---------------------------------------------------------------------------
//  Named constructors  —  gamma constante
// ---------------------------------------------------------------------------

BoundaryCondition BoundaryCondition::dirichlet(PetscReal gamma) {
    return { Type::Dirichlet, 1.0, 0.0, gamma };
}

BoundaryCondition BoundaryCondition::neumann(PetscReal gamma) {
    return { Type::Neumann, 0.0, 1.0, gamma };
}

BoundaryCondition BoundaryCondition::robin(PetscReal alpha,
                                           PetscReal beta,
                                           PetscReal gamma) {
    return { typeFrom(alpha, beta), alpha, beta, gamma };
}

// ---------------------------------------------------------------------------
//  Named constructors  —  gamma função do tempo
// ---------------------------------------------------------------------------

BoundaryCondition BoundaryCondition::dirichlet(GammaFn fn) {
    return { Type::Dirichlet, 1.0, 0.0, std::move(fn) };
}

BoundaryCondition BoundaryCondition::neumann(GammaFn fn) {
    return { Type::Neumann, 0.0, 1.0, std::move(fn) };
}

BoundaryCondition BoundaryCondition::robin(PetscReal alpha,
                                           PetscReal beta,
                                           GammaFn   fn) {
    return { typeFrom(alpha, beta), alpha, beta, std::move(fn) };
}

// ---------------------------------------------------------------------------
//  Construção a partir de YAML
// ---------------------------------------------------------------------------

BoundaryCondition BoundaryCondition::fromYAML(const YAML::Node& node, int k) {
    const std::string s = std::to_string(k);
    const PetscReal a = node["alpha" + s].as<PetscReal>(k == 0 ? 1.0 : 0.0);
    const PetscReal b = node["beta"  + s].as<PetscReal>(k == 0 ? 0.0 : 1.0);
    const PetscReal g = node["gamma" + s].as<PetscReal>(0.0);
    return { typeFrom(a, b), a, b, g };
}

// ---------------------------------------------------------------------------
//  Accessors
// ---------------------------------------------------------------------------

PetscReal BoundaryCondition::alpha() const { return alpha_; }
PetscReal BoundaryCondition::beta()  const { return beta_;  }

PetscReal BoundaryCondition::gamma(std::optional<PetscReal> t) const {
    return std::visit([&](const auto& g) -> PetscReal {

        if constexpr (std::is_same_v<std::decay_t<decltype(g)>, PetscReal>) {
            return g;
        } else {
            PetscCheck(t.has_value(),
                       PETSC_COMM_SELF,
                       PETSC_ERR_ARG_WRONGSTATE,
                       "BoundaryCondition::gamma() chamado sem argumento de "
                       "tempo em uma condição dependente do tempo. "
                       "Use gamma(t) em vez de gamma().");
            return g(*t);
        }

    }, gamma_);
}

BoundaryCondition::Type BoundaryCondition::type()        const { return type_;  }
bool BoundaryCondition::isDirichlet()                     const { return type_ == Type::Dirichlet; }
bool BoundaryCondition::isNeumann()                       const { return type_ == Type::Neumann;   }
bool BoundaryCondition::isRobin()                         const { return type_ == Type::Robin;     }
bool BoundaryCondition::isTimeDependent()                 const {
    return std::holds_alternative<GammaFn>(gamma_);
}

} // namespace bgc
