#pragma once

// ---------------------------------------------------------------------------
//  SimConfig.hpp
//
//  Estrutura plana (DOD) com todos os parâmetros de configuração da
//  simulação.  Contém somente dados escalares e strings — sem objetos
//  PETSc.  É inicializada pela camada de IO e distribuída via MPI antes
//  de qualquer alocação PETSc.
//
//  Separação de responsabilidades:
//      SimConfig   — configuração / parâmetros (este arquivo)
//      SimState    — objetos PETSc alocados em tempo de execução
// ---------------------------------------------------------------------------

#include <string>

#include <petsc.h>

#include <bgclib/Core/BoundaryCondition.hpp>
#include <bgclib/Misc/Types.hpp>

namespace bgc {

enum class DirectSolverBackend {
    PetscDefault,   ///< Backend padrão do PETSc.
    Mumps           ///< Backend MUMPS.
};

struct SimConfig {

    // -----------------------------------------------------------------------
    //  Malha
    // -----------------------------------------------------------------------

    PetscInt  nx  {0};      ///< Número de volumes de controle.
    PetscReal lx  {1.0};    ///< Comprimento do domínio.
    PetscReal x0  {0.0};    ///< Coordenada inicial do domínio.
    PetscReal h   {0.0};    ///< Espaçamento (calculado: lx / nx).

    // -----------------------------------------------------------------------
    //  Tempo
    // -----------------------------------------------------------------------

    PetscReal dt     {0.0};     ///< Passo de tempo.
    PetscReal tf     {1.0};     ///< Tempo final da simulação.
    PetscInt  nTimes {0};       ///< Número de passos temporais.

    // -----------------------------------------------------------------------
    //  Física
    // -----------------------------------------------------------------------

    PetscReal bv    {0.0};          ///< Número de Bevilacqua (Bv).
    Model     model {Model::TBGC};  ///< Modelo ativo: BGC, TBGC ou TSPV.

    // -----------------------------------------------------------------------
    //  Fisica especifica do modelo TSPV
    // -----------------------------------------------------------------------

    PetscReal beta    {0.0};  ///< Deprecated in TSOM-theta; kept for compatibility.
    PetscReal ell     {0.0};  ///< Deprecated in TSOM-theta; kept for compatibility.
    PetscReal lambdaC {0.0};  ///< Dimensionless capture group Lambda_c.
    PetscReal lambdaR {0.0};  ///< Dimensionless release group Lambda_r.

    // -----------------------------------------------------------------------
    //  Fisica especifica do modelo TSOM theta
    // -----------------------------------------------------------------------

    PetscReal alphaTSOM {0.0};  ///< Captura discreta U -> V.
    PetscReal rhoTSOM   {0.0};  ///< Liberacao discreta V -> U.
    PetscReal thetaTSOM {0.0};  ///< Fracao da troca feita antes do transporte.
    PetscReal fTSOM     {0.5};  ///< MMS: U = f Psi, V = (1-f) Psi.

    PetscReal alphaPsiWest {0.0};
    PetscReal betaPsiWest {0.0};
    PetscReal gammaPsiWest {0.0};

    PetscReal alphaPsiEast {0.0};
    PetscReal betaPsiEast {0.0};
    PetscReal gammaPsiEast {0.0};

    PetscReal psiInitial {0.0};
    PetscReal vInitial   {0.0};

    // -----------------------------------------------------------------------
    //  Condições de contorno
    //
    //  Cada fronteira possui duas condições independentes (necessárias para
    //  a equação de quarta ordem).  Os índices 0 e 1 correspondem às
    //  equações k = 0 e k = 1 da notação do artigo.
    // -----------------------------------------------------------------------

    
    BoundaryCondition bcWest[2] = {
        BoundaryCondition::dirichlet(0.0),
        BoundaryCondition::neumann  (0.0)
    };

    BoundaryCondition bcEast[2] = {
        BoundaryCondition::dirichlet(0.0),
        BoundaryCondition::neumann  (0.0)
    };

    // -----------------------------------------------------------------------
    //  Parâmetros da solução analítica fabricada
    // -----------------------------------------------------------------------

    PetscReal alpha  {0.0};     ///< Taxa de decaimento temporal.
    PetscReal delta  {0.0};     ///< Parâmetro de forma.
    PetscReal gamma  {0.0};     ///< Amplitude.
    PetscReal phiMin {0.0};     ///< Valor mínimo de phi.

    // -----------------------------------------------------------------------
    //  Opções do solver
    // -----------------------------------------------------------------------

    PetscReal tole {1.0e-10};   ///< Tolerância do solver linear.

    bool useDirectSolver {true};
    ///< true  = solver direto
    ///< false = solver iterativo

    DirectSolverBackend directSolverBackend {
        DirectSolverBackend::Mumps
    };
    ///< Backend do solver direto.

    bool verbose {true};       ///< Imprime diagnóstico a cada passo.

    PetscReal equationScale {1.0};

    // -----------------------------------------------------------------------
    //  Caminhos de IO
    // -----------------------------------------------------------------------

    std::string inputFilepath;      ///< Caminho completo do arquivo de dados.
    std::string outputDir;          ///< Pasta de saída dos resultados.
    std::string errorsFile;         ///< Arquivo das normas L1, L2, Linf.
    std::string fieldsFile;         ///< Campos phi(x) e mu(x) no instante final.
    std::string transientFile;      ///< phi(x, t) nos instantes selecionados.
    PetscInt    outputFreq {10};    ///< Frequência de gravação do transiente.

    // -----------------------------------------------------------------------
    //  Helpers
    // -----------------------------------------------------------------------

    /// Retorna a coordenada do centro do volume i.
    [[nodiscard]] PetscReal xCenter(PetscInt i) const noexcept {
        return x0 + (static_cast<PetscReal>(i) + 0.5) * h;
    }

    /// Retorna o nome do modelo como string.
    [[nodiscard]] std::string modelName() const {
        switch (model) {
            case Model::BGC:
                return "BGC";
            case Model::TBGC:
                return "TBGC";
            case Model::TSPV:
                return "TSPV";
        }
        return "Unknown";
    }

    /// Retorna o nome do backend direto como string.
    [[nodiscard]] std::string directSolverBackendName() const {
        switch (directSolverBackend) {
            case DirectSolverBackend::PetscDefault:
                return "PETSc default";
            case DirectSolverBackend::Mumps:
                return "MUMPS";
        }
        return "Unknown";
    }
};

} // namespace bgc