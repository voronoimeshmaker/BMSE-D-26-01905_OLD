#include <bgclib/Transient.hpp>
#include <bgclib/Analytics.hpp>
#include <bgclib/Misc/Types.hpp>
#include <bgclib/PostProcess.hpp>
#include <bgclib/SimConfig.hpp>
#include <bgclib/SimState.hpp>
#include <bgclib/Solver.hpp>
#include <bgclib/Models/BGC/AssemblyBGC.hpp>
#include <bgclib/Models/BGC/CoeffBGC.hpp>
#include <bgclib/Models/TBGC/AssemblyTBGC.hpp>
#include <bgclib/Models/TBGC/CoeffTBGC.hpp>

#include <cmath>

namespace bgc {

#define TRACE_MSG(msg) \
    do { \
        if (cfg.verbose) { \
            PetscPrintf(PETSC_COMM_WORLD, "[Transient] %s\n", msg); \
        } \
    } while (0)

#define TRACE_REAL(name, value) \
    do { \
        if (cfg.verbose) { \
            PetscPrintf(PETSC_COMM_WORLD, "[Transient] %s = %.16e\n", name, value); \
        } \
    } while (0)

#define TRACE_INT(name, value) \
    do { \
        if (cfg.verbose) { \
            PetscPrintf(PETSC_COMM_WORLD, "[Transient] %s = %" PetscInt_FMT "\n", \
                        name, static_cast<PetscInt>(value)); \
        } \
    } while (0)

static PetscErrorCode checkVecForInvalidValues(Vec v,
                                               const char* name,
                                               PetscReal   t) {
    PetscFunctionBeginUser;

    const PetscScalar* array = nullptr;
    PetscInt localSize = 0;
    PetscInt firstBad = -1;
    PetscInt badCount = 0;

    PetscCall(VecGetLocalSize(v, &localSize));
    PetscCall(VecGetArrayRead(v, &array));

    for (PetscInt i = 0; i < localSize; ++i) {
        if (PetscIsInfOrNanScalar(array[i])) {
            ++badCount;
            if (firstBad < 0) {
                firstBad = i;
            }
        }
    }

    PetscCall(VecRestoreArrayRead(v, &array));

    PetscCheck(badCount == 0,
               PETSC_COMM_WORLD,
               PETSC_ERR_PLIB,
               "Valores invalidos detectados em %s no tempo t=%.16e. "
               "Quantidade local = %" PetscInt_FMT ", primeiro indice local = %" PetscInt_FMT,
               name,
               t,
               badCount,
               firstBad);

    PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode printVecStats(Vec v,
                                    const char* name,
                                    PetscReal   t,
                                    bool        verbose) {
    PetscFunctionBeginUser;

    if (!verbose) {
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    PetscReal minVal = 0.0;
    PetscReal maxVal = 0.0;
    PetscReal norm2  = 0.0;

    PetscCall(VecMin(v, nullptr, &minVal));
    PetscCall(VecMax(v, nullptr, &maxVal));
    PetscCall(VecNorm(v, NORM_2, &norm2));

    PetscPrintf(PETSC_COMM_WORLD,
                "[Transient] %s em t=%.16e : min=%.16e  max=%.16e  ||.||_2=%.16e\n",
                name,
                t,
                minVal,
                maxVal,
                norm2);

    PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode validateBoundaryPair(const BoundaryCondition (&bc)[2],
                                           const char*                side,
                                           PetscReal                  t,
                                           bool                       verbose) {
    PetscFunctionBeginUser;

    const PetscReal a1 = bc[0].alpha();
    const PetscReal b1 = bc[0].beta();
    const PetscReal a2 = bc[1].alpha();
    const PetscReal b2 = bc[1].beta();
    const PetscReal g1 = bc[0].gamma(t);
    const PetscReal g2 = bc[1].gamma(t);
    const PetscReal det = b2 * a1 - a2 * b1;

    if (verbose) {
        PetscPrintf(PETSC_COMM_WORLD,
                    "[Transient] BC %s: a1=%.16e  b1=%.16e  g1=%.16e  "
                    "a2=%.16e  b2=%.16e  g2=%.16e  det=%.16e\n",
                    side,
                    a1,
                    b1,
                    g1,
                    a2,
                    b2,
                    g2,
                    det);
    }

    PetscCheck(std::abs(det) > 1.0e-30,
               PETSC_COMM_WORLD,
               PETSC_ERR_ARG_WRONG,
               "As duas condicoes de contorno em %s sao linearmente dependentes. "
               "Determinante = %.16e.",
               side,
               det);

    PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode printConfig(const SimConfig& cfg) {
    PetscFunctionBeginUser;

    if (!cfg.verbose) {
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    PetscPrintf(PETSC_COMM_WORLD, "%s", separator().c_str());
    PetscPrintf(PETSC_COMM_WORLD,
                "  Modelo : %s\n"
                "  nx     = %" PetscInt_FMT "\n"
                "  h      = %.16e\n"
                "  Bv     = %.16e\n"
                "  dt     = %.16e\n"
                "  tf     = %.16e\n"
                "  nTimes = %" PetscInt_FMT "\n",
                cfg.modelName().c_str(),
                cfg.nx,
                cfg.h,
                cfg.bv,
                cfg.dt,
                cfg.tf,
                cfg.nTimes);

    PetscPrintf(PETSC_COMM_WORLD,
                "  BC West[0]: type=%d  alpha=%.16e  beta=%.16e  gamma(0)=%.16e\n",
                static_cast<int>(cfg.bcWest[0].type()),
                cfg.bcWest[0].alpha(),
                cfg.bcWest[0].beta(),
                cfg.bcWest[0].gamma(0.0));
    PetscPrintf(PETSC_COMM_WORLD,
                "  BC West[1]: type=%d  alpha=%.16e  beta=%.16e  gamma(0)=%.16e\n",
                static_cast<int>(cfg.bcWest[1].type()),
                cfg.bcWest[1].alpha(),
                cfg.bcWest[1].beta(),
                cfg.bcWest[1].gamma(0.0));
    PetscPrintf(PETSC_COMM_WORLD,
                "  BC East[0]: type=%d  alpha=%.16e  beta=%.16e  gamma(0)=%.16e\n",
                static_cast<int>(cfg.bcEast[0].type()),
                cfg.bcEast[0].alpha(),
                cfg.bcEast[0].beta(),
                cfg.bcEast[0].gamma(0.0));
    PetscPrintf(PETSC_COMM_WORLD,
                "  BC East[1]: type=%d  alpha=%.16e  beta=%.16e  gamma(0)=%.16e\n",
                static_cast<int>(cfg.bcEast[1].type()),
                cfg.bcEast[1].alpha(),
                cfg.bcEast[1].beta(),
                cfg.bcEast[1].gamma(0.0));

    PetscPrintf(PETSC_COMM_WORLD, "%s", separator().c_str());
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode runTransient(const SimConfig& cfg,
                            const FieldFn&   sourceFn,
                            const FieldFn&   phiFn,
                            const FieldFn&   muFn) {
    PetscFunctionBeginUser;

    PetscCheck(cfg.nx >= 4,
               PETSC_COMM_WORLD,
               PETSC_ERR_ARG_OUTOFRANGE,
               "O modelo requer nx >= 4. Valor recebido = %" PetscInt_FMT,
               cfg.nx);
    PetscCheck(cfg.h > 0.0,
               PETSC_COMM_WORLD,
               PETSC_ERR_ARG_OUTOFRANGE,
               "O espacamento h deve ser positivo. Valor recebido = %.16e.",
               cfg.h);
    PetscCheck(cfg.dt > 0.0,
               PETSC_COMM_WORLD,
               PETSC_ERR_ARG_OUTOFRANGE,
               "O passo de tempo dt deve ser positivo. Valor recebido = %.16e.",
               cfg.dt);

    TRACE_MSG("entrou em runTransient");
    PetscCall(printConfig(cfg));
    PetscCall(validateBoundaryPair(cfg.bcWest, "west", 0.0, cfg.verbose));
    PetscCall(validateBoundaryPair(cfg.bcEast, "east", 0.0, cfg.verbose));

    SimState st;

    TRACE_MSG("checkpoint 1: createState");
    if (cfg.model == Model::TBGC) {
        PetscCall(createStateTBGC(cfg, st));
    } else {
        PetscCall(createStateBGC(cfg, st));
        st.x = st.phi;
    }

    TRACE_MSG("checkpoint 2: assemble matrix");
    if (cfg.model == Model::TBGC) {
        const auto coeff = computeCoefficientsTBGC(cfg);
        PetscCall(assembleMatrixTBGC(cfg, st, coeff));
    } else {
        const auto coeff = computeCoefficientsBGC(cfg);
        PetscCall(assembleMatrixBGC(cfg, st, coeff));

        PetscReal normA = 0.0;
        PetscCall(MatNorm(st.A, NORM_FROBENIUS, &normA));
        TRACE_REAL("||A||_F", normA);
        PetscCheck(normA > 0.0,
                   PETSC_COMM_WORLD,
                   PETSC_ERR_PLIB,
                   "A matriz do sistema ficou com norma nula apos a montagem.");
    }

    TRACE_MSG("checkpoint 3: configureSolver");
    PetscCall(configureSolver(cfg, st));

    TRACE_MSG("checkpoint 4: initial condition");
    PetscCall(computeAnalyticPhi(cfg, st, 0.0, phiFn));
    PetscCall(checkVecForInvalidValues(st.phi_a, "phi_a(t=0)", 0.0));
    PetscCall(printVecStats(st.phi_a, "phi_a(t=0)", 0.0, cfg.verbose));
    PetscCall(VecCopy(st.phi_a, st.phi_0));

    if (cfg.model == Model::BGC) {
        PetscCall(VecCopy(st.phi_a, st.phi));
        PetscCall(checkVecForInvalidValues(st.phi, "phi(t=0)", 0.0));
    }

    PetscReal F_prev = 0.0;
    {
        PetscReal norm2 = 0.0;
        PetscCall(VecNorm(st.phi_0, NORM_2, &norm2));
        F_prev = 0.5 * cfg.h * norm2 * norm2;
    }

    if (cfg.verbose) {
        PetscPrintf(PETSC_COMM_WORLD,
                    "[thermo] tau=%.16e  F=%.16e  (inicial)\n",
                    0.0,
                    F_prev);
    }

    PetscReal t = 0.0;

    for (PetscInt n = 0; n < cfg.nTimes; ++n) {
        t = static_cast<PetscReal>(n + 1) * cfg.dt;
        if (cfg.verbose) {
            PetscPrintf(PETSC_COMM_WORLD,
                        "[Transient] ==================== passo %" PetscInt_FMT
                        "  t=%.16e ====================\n",
                        n + 1,
                        t);
        }

        TRACE_MSG("checkpoint 5: assemble RHS");
        if (cfg.model == Model::TBGC) {
            PetscCall(VecZeroEntries(st.b1));
            PetscCall(VecZeroEntries(st.b2));

            const auto rhs = computeRHSTBGC(cfg, t);
            PetscCall(assembleRHSTBGC(cfg, st, rhs));
            PetscCall(computeSourceTerm(cfg, st, t, sourceFn));
            PetscCall(VecAXPY(st.b1, 1.0, st.b1Source));
            PetscCall(checkVecForInvalidValues(st.b1, "b1", t));
            PetscCall(checkVecForInvalidValues(st.b2, "b2", t));
        } else {
            PetscCall(VecZeroEntries(st.b));

            const auto rhs = computeRHSBGC(cfg, t);
            TRACE_REAL("rhs.vol0", rhs.vol0);
            TRACE_REAL("rhs.vol1", rhs.vol1);
            TRACE_REAL("rhs.volNm2", rhs.volNm2);
            TRACE_REAL("rhs.volNm1", rhs.volNm1);

            PetscCall(assembleRHSBGC(cfg, st, rhs));
            PetscCall(checkVecForInvalidValues(st.b, "b apos BC", t));
            PetscCall(printVecStats(st.b, "b apos BC", t, cfg.verbose));

            PetscCall(computeSourceTerm(cfg, st, t, sourceFn));
            PetscCall(checkVecForInvalidValues(st.b1Source, "b1Source", t));
            PetscCall(VecAXPY(st.b, 1.0, st.b1Source));
            PetscCall(checkVecForInvalidValues(st.b, "b final", t));
            PetscCall(printVecStats(st.b, "b final", t, cfg.verbose));
        }

        TRACE_MSG("checkpoint 6: solveLinearSystem");
        PetscCall(solveLinearSystem(cfg, st));
        PetscCall(checkVecForInvalidValues(st.phi, "phi apos solve", t));
        PetscCall(printVecStats(st.phi, "phi apos solve", t, cfg.verbose));

        TRACE_MSG("checkpoint 7: thermodynamics");
        PetscBool violated = PETSC_FALSE;
        PetscCall(checkThermodynamicConsistency(cfg, st, t, F_prev, violated));
        PetscCheck(!violated,
                   PETSC_COMM_WORLD,
                   PETSC_ERR_NOT_CONVERGED,
                   "Violacao termodinamica detectada em t=%.16e.",
                   t);

        TRACE_MSG("checkpoint 8: update history");
        PetscCall(VecCopy(st.phi, st.phi_0));
    }



    ErrorNorms norms;
    PetscCall(computeErrorNorms(cfg, st, t, phiFn, norms));

    TRACE_MSG("checkpoint 9: post-process");
    if (cfg.verbose) {
        PetscPrintf(PETSC_COMM_WORLD,
                "[erro] tau=%.6e  L1=%.6e  L2=%.6e  Linf=%.6e\n",
                static_cast<double>(t),
                static_cast<double>(norms.L1),
                static_cast<double>(norms.L2),
                static_cast<double>(norms.Linf));
    }    

    if (cfg.model == Model::TBGC && muFn) {
        PetscCall(computeAnalyticMu(cfg, st, t, muFn));
    }

    TRACE_MSG("checkpoint 10: destroyState");
    PetscCall(destroyState(st));
    TRACE_MSG("runTransient terminou com sucesso");

    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace bgc
