#pragma once

// ---------------------------------------------------------------------------
//  BoundaryCondition.hpp
//
//  Encapsula uma única condição de contorno linear da forma:
//
//      alpha * Phi  +  beta * dPhi/dxi  =  gamma(t)
//
//  onde alpha e beta são escalares constantes e gamma é uma constante
//  ou uma função genérica do tempo.
//
//  Três tipos canônicos são suportados via Named Constructor Idiom,
//  de modo que o chamador nunca precisa fornecer alpha ou beta diretamente.
//  Não há herança — um único tipo de valor com enum interno.
//
//  Tipo          alpha   beta
//  Dirichlet       1       0     ->  Phi = gamma
//  Neumann         0       1     ->  dPhi/dxi = gamma
//  Robin         != 0    != 0    ->  condição mista
//
//  Uso
//  ---
//      // Gamma constante
//      auto bc = BoundaryCondition::dirichlet(1.0);
//      auto bc = BoundaryCondition::neumann(0.0);
//      auto bc = BoundaryCondition::robin(2.0, 1.0, 0.5);
//
//      // Gamma função do tempo  —  lambda, functor ou ponteiro de função
//      auto bc = BoundaryCondition::dirichlet(
//                    [](PetscReal t){ return std::sin(t); });
//      auto bc = BoundaryCondition::neumann(minha_funcao);
//
//      // Leitura dos coeficientes
//      PetscReal a = bc.alpha();
//      PetscReal b = bc.beta();
//
//      // Avaliação de gamma
//      PetscReal g = bc.gamma();     // condição constante — t irrelevante
//      PetscReal g = bc.gamma(t);    // caso geral; t = 0 é valor legítimo
// ---------------------------------------------------------------------------

#include <functional>
#include <optional>
#include <variant>

#include <petscsys.h>
#include <yaml-cpp/yaml.h>

namespace bgc {

class BoundaryCondition {
public:

    // -----------------------------------------------------------------------
    //  Tipos públicos
    // -----------------------------------------------------------------------

    enum class Type { Dirichlet, Neumann, Robin };

    using GammaFn  = std::function<PetscReal(PetscReal t)>;
    using GammaVar = std::variant<PetscReal, GammaFn>;

    // -----------------------------------------------------------------------
    //  Named constructors  —  gamma constante
    // -----------------------------------------------------------------------

    static BoundaryCondition dirichlet(PetscReal gamma);
    static BoundaryCondition neumann  (PetscReal gamma);
    static BoundaryCondition robin    (PetscReal alpha,
                                       PetscReal beta,
                                       PetscReal gamma);

    // -----------------------------------------------------------------------
    //  Named constructors  —  gamma função do tempo
    //
    //  Aceita lambda, functor ou ponteiro de função com assinatura
    //  compatível com PetscReal(PetscReal).
    // -----------------------------------------------------------------------

    static BoundaryCondition dirichlet(GammaFn fn);
    static BoundaryCondition neumann  (GammaFn fn);
    static BoundaryCondition robin    (PetscReal alpha,
                                       PetscReal beta,
                                       GammaFn   fn);

    // -----------------------------------------------------------------------
    //  Construção a partir de um nó YAML
    //
    //  Lê  alpha<k>,  beta<k>,  gamma<k>  do nó fornecido, onde <k> é o
    //  índice inteiro da condição (0 ou 1).  Gamma é sempre tratado como
    //  constante quando carregado do YAML.
    //
    //  Valores padrão:
    //      k == 0  :  alpha = 1,  beta = 0,  gamma = 0   (Dirichlet)
    //      k == 1  :  alpha = 0,  beta = 1,  gamma = 0   (Neumann)
    // -----------------------------------------------------------------------

    static BoundaryCondition fromYAML(const YAML::Node& node, int k);

    // -----------------------------------------------------------------------
    //  Accessors  —  alpha e beta são sempre constantes
    // -----------------------------------------------------------------------

    PetscReal alpha() const;
    PetscReal beta()  const;

    // -----------------------------------------------------------------------
    //  Accessor  —  gamma
    //
    //  gamma()    ->  válido apenas para condições constantes; lança erro
    //                 via PetscCheck se a condição for dependente do tempo
    //                 e t não for fornecido.
    //
    //  gamma(t)   ->  válido para constante e dependente do tempo;
    //                 t = 0 é um valor legítimo.
    // -----------------------------------------------------------------------

    PetscReal gamma(std::optional<PetscReal> t = std::nullopt) const;

    // -----------------------------------------------------------------------
    //  Consultas de tipo
    // -----------------------------------------------------------------------

    Type type()            const;
    bool isDirichlet()     const;
    bool isNeumann()       const;
    bool isRobin()         const;
    bool isTimeDependent() const;

private:

    BoundaryCondition(Type t,
                      PetscReal alpha,
                      PetscReal beta,
                      GammaVar  gamma);

    static Type typeFrom(PetscReal alpha, PetscReal beta);

    Type      type_;
    PetscReal alpha_;
    PetscReal beta_;
    GammaVar  gamma_;
};

} // namespace bgc
