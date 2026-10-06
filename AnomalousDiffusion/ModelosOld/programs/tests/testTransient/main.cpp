#include <bgclib/Core/BoundaryCondition.hpp>
#include <bgclib/Models/BGC/AssemblyBGC.hpp>
#include <bgclib/Models/BGC/CoeffBGC.hpp>
#include <bgclib/SimConfig.hpp>
#include <bgclib/SimState.hpp>

#include <petsc.h>

#include <array>
#include <string>

namespace {

constexpr PetscReal kTf = 1.0e-3;
constexpr PetscReal kCDt = 1.024e-1;

// -----------------------------------------------------------------------------
// Pretest configuration:
// - West:  Dirichlet = 1, Neumann = 0
// - East:  Dirichlet = 0, Neumann = 0
// - Null field and null source
//
// Current goal: inspect only the algebraic assembly of A and B.
// -----------------------------------------------------------------------------
bgc::SimConfig makeConfig(PetscReal bv,
                          PetscInt  nx) {
    bgc::SimConfig cfg;

    cfg.model = bgc::Model::BGC;
    cfg.nx = nx;
    cfg.lx = 1.0;
    cfg.x0 = 0.0;
    cfg.h = cfg.lx / static_cast<PetscReal>(cfg.nx);
    cfg.bv = bv;
    cfg.tf = kTf;
    cfg.dt = kCDt * cfg.h * cfg.h;
    cfg.nTimes = static_cast<PetscInt>(cfg.tf / cfg.dt + 0.5);
    cfg.dt = cfg.tf / static_cast<PetscReal>(cfg.nTimes);

    cfg.useDirectSolver = true;
    cfg.verbose = true;

// West boundary
    cfg.bcWest[0] = bgc::BoundaryCondition::neumann(0.0);
    cfg.bcWest[1] = bgc::BoundaryCondition::dirichlet(1.0);

// East boundary
    cfg.bcEast[0] = bgc::BoundaryCondition::neumann(0.0);
    cfg.bcEast[1] = bgc::BoundaryCondition::dirichlet(1.0);

    return cfg;
}

PetscErrorCode printVecStats(Vec         v,
                             const char* name) {
    PetscFunctionBeginUser;

    PetscReal minVal = 0.0;
    PetscReal maxVal = 0.0;
    PetscReal norm1 = 0.0;
    PetscReal norm2 = 0.0;
    PetscReal normInf = 0.0;

    PetscCall(VecMin(v, nullptr, &minVal));
    PetscCall(VecMax(v, nullptr, &maxVal));
    PetscCall(VecNorm(v, NORM_1, &norm1));
    PetscCall(VecNorm(v, NORM_2, &norm2));
    PetscCall(VecNorm(v, NORM_INFINITY, &normInf));

    PetscPrintf(PETSC_COMM_WORLD,
                "%s : min=%.16e  max=%.16e  ||.||_1=%.16e  ||.||_2=%.16e  ||.||_inf=%.16e\n",
                name,
                minVal,
                maxVal,
                norm1,
                norm2,
                normInf);

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode printSelectedValues(const char* label,
                                   Vec         v,
                                   PetscInt    n) {
    PetscFunctionBeginUser;

    const std::array<PetscInt, 6> ids = {0, 1, 2, n - 3, n - 2, n - 1};

    PetscPrintf(PETSC_COMM_WORLD, "%s\n", label);
    for (PetscInt id : ids) {
        PetscScalar value = 0.0;
        PetscCall(VecGetValues(v, 1, &id, &value));
        PetscPrintf(PETSC_COMM_WORLD,
                    "  [i=%" PetscInt_FMT "] %.16e\n",
                    id,
                    static_cast<PetscReal>(value));
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode printFullVector(const char* label,
                               Vec         v) {
    PetscFunctionBeginUser;

    PetscInt n = 0;
    PetscCall(VecGetSize(v, &n));

    PetscPrintf(PETSC_COMM_WORLD, "%s\n", label);
    for (PetscInt i = 0; i < n; ++i) {
        PetscScalar value = 0.0;
        PetscCall(VecGetValues(v, 1, &i, &value));
        PetscPrintf(PETSC_COMM_WORLD,
                    "  [%" PetscInt_FMT "] = %.16e\n",
                    i,
                    static_cast<PetscReal>(value));
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode printMatrixRow(Mat      A,
                              PetscInt row) {
    PetscFunctionBeginUser;

    PetscInt ncols = 0;
    const PetscInt* cols = nullptr;
    const PetscScalar* vals = nullptr;

    PetscCall(MatGetRow(A, row, &ncols, &cols, &vals));
    PetscPrintf(PETSC_COMM_WORLD,
                "Matrix row i=%" PetscInt_FMT
                " (ncols=%" PetscInt_FMT ")\n",
                row,
                ncols);

    for (PetscInt k = 0; k < ncols; ++k) {
        PetscPrintf(PETSC_COMM_WORLD,
                    "  col=%" PetscInt_FMT "  val=%.16e\n",
                    cols[k],
                    static_cast<PetscReal>(vals[k]));
    }

    PetscCall(MatRestoreRow(A, row, &ncols, &cols, &vals));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode printFullMatrix(const char* label,
                               Mat         A) {
    PetscFunctionBeginUser;

    PetscInt nrows = 0;
    PetscInt ncols = 0;
    PetscCall(MatGetSize(A, &nrows, &ncols));

    PetscPrintf(PETSC_COMM_WORLD, "%s\n", label);
    for (PetscInt i = 0; i < nrows; ++i) {
        PetscPrintf(PETSC_COMM_WORLD, "  row %" PetscInt_FMT " :", i);
        for (PetscInt j = 0; j < ncols; ++j) {
            PetscScalar value = 0.0;
            PetscCall(MatGetValues(A, 1, &i, 1, &j, &value));
            PetscPrintf(PETSC_COMM_WORLD,
                        " % .16e",
                        static_cast<PetscReal>(value));
        }
        PetscPrintf(PETSC_COMM_WORLD, "\n");
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode runABPretestCampaign(PetscReal          bv,
                                    const std::string& label) {
    PetscFunctionBeginUser;

    constexpr std::array<PetscInt, 1> kNxList = {8};

    PetscPrintf(PETSC_COMM_WORLD,
                "\n======================================================================\n"
                "  Pretest of A and B: %s\n"
                "======================================================================\n",
                label.c_str());

    for (PetscInt nx : kNxList) {
        const bgc::SimConfig cfg = makeConfig(bv, nx);
        const PetscReal t1 = cfg.dt;

        PetscPrintf(PETSC_COMM_WORLD,
                    "\n--- nx=%" PetscInt_FMT
                    "  h=%.16e  dt=%.16e  Bv=%.16e ---\n",
                    cfg.nx,
                    cfg.h,
                    cfg.dt,
                    cfg.bv);

        PetscPrintf(PETSC_COMM_WORLD,
                    "BC west  (stored order):\n");
        PetscPrintf(PETSC_COMM_WORLD,
                    "  bcWest[0] -> alpha=%.16e  beta=%.16e  gamma=%.16e\n",
                    cfg.bcWest[0].alpha(),
                    cfg.bcWest[0].beta(),
                    cfg.bcWest[0].gamma(t1));
        PetscPrintf(PETSC_COMM_WORLD,
                    "  bcWest[1] -> alpha=%.16e  beta=%.16e  gamma=%.16e\n",
                    cfg.bcWest[1].alpha(),
                    cfg.bcWest[1].beta(),
                    cfg.bcWest[1].gamma(t1));

        PetscPrintf(PETSC_COMM_WORLD,
                    "BC east  (stored order):\n");
        PetscPrintf(PETSC_COMM_WORLD,
                    "  bcEast[0] -> alpha=%.16e  beta=%.16e  gamma=%.16e\n",
                    cfg.bcEast[0].alpha(),
                    cfg.bcEast[0].beta(),
                    cfg.bcEast[0].gamma(t1));
        PetscPrintf(PETSC_COMM_WORLD,
                    "  bcEast[1] -> alpha=%.16e  beta=%.16e  gamma=%.16e\n",
                    cfg.bcEast[1].alpha(),
                    cfg.bcEast[1].beta(),
                    cfg.bcEast[1].gamma(t1));

        PetscPrintf(PETSC_COMM_WORLD,
                    "Field    : phi = 0.0\n");
        PetscPrintf(PETSC_COMM_WORLD,
                    "Source   : f = 0.0\n");

        bgc::SimState st;

        PetscCall(bgc::createStateBGC(cfg, st));
        st.x = st.phi;

        // Purely algebraic pretest: all state vectors start at zero.
        PetscCall(VecZeroEntries(st.phi));
        PetscCall(VecZeroEntries(st.phi_0));
        PetscCall(VecZeroEntries(st.phi_a));
        PetscCall(VecZeroEntries(st.b));
        PetscCall(VecZeroEntries(st.b1Source));

        const bgc::StencilCoefficients sc = bgc::computeCoefficientsBGC(cfg);

        PetscPrintf(PETSC_COMM_WORLD,
                    "\nFirst-volume coefficients (before MatShift):\n");
        PetscPrintf(PETSC_COMM_WORLD,
                    "  ap   = %.16e\n",
                    sc.vol0.coef[0]);
        PetscPrintf(PETSC_COMM_WORLD,
                    "  ae   = %.16e\n",
                    sc.vol0.coef[1]);
        PetscPrintf(PETSC_COMM_WORLD,
                    "  aee  = %.16e\n",
                    sc.vol0.coef[2]);
        PetscPrintf(PETSC_COMM_WORLD,
                    "  aeee = %.16e\n",
                    sc.vol0.coef[3]);

        PetscCall(bgc::assembleMatrixBGC(cfg, st, sc));

        const bgc::RHSCoefficients rhs = bgc::computeRHSBGC(cfg, t1);

        PetscPrintf(PETSC_COMM_WORLD,
                    "\nBoundary RHS terms at t = dt:\n");
        PetscPrintf(PETSC_COMM_WORLD,
                    "  rhs.vol0   = %.16e\n",
                    rhs.vol0);
        PetscPrintf(PETSC_COMM_WORLD,
                    "  rhs.vol1   = %.16e\n",
                    rhs.vol1);
        PetscPrintf(PETSC_COMM_WORLD,
                    "  rhs.volNm2 = %.16e\n",
                    rhs.volNm2);
        PetscPrintf(PETSC_COMM_WORLD,
                    "  rhs.volNm1 = %.16e\n",
                    rhs.volNm1);

        PetscCall(bgc::assembleRHSBGC(cfg, st, rhs));

        PetscPrintf(PETSC_COMM_WORLD,
                    "\nSource-term vector (current pretest = zero):\n");
        PetscCall(printVecStats(st.b1Source, "b1Source"));
        PetscCall(printFullVector("Full source vector b1Source:",
                                  st.b1Source));

        PetscPrintf(PETSC_COMM_WORLD,
                    "\nFinal RHS vector B:\n");
        PetscCall(printVecStats(st.b, "B final"));
        PetscCall(printSelectedValues("Selected values of B:",
                                      st.b,
                                      cfg.nx));
        PetscCall(printFullVector("Full vector B:",
                                  st.b));

        PetscPrintf(PETSC_COMM_WORLD,
                    "\nSelected rows of matrix A:\n");
        PetscCall(printMatrixRow(st.A, 0));
        PetscCall(printMatrixRow(st.A, 1));
        PetscCall(printMatrixRow(st.A, 2));
        PetscCall(printMatrixRow(st.A, 3));
        PetscCall(printMatrixRow(st.A, 4));
        PetscCall(printMatrixRow(st.A, 5));
        PetscCall(printMatrixRow(st.A, cfg.nx - 2));
        PetscCall(printMatrixRow(st.A, cfg.nx - 1));

        PetscPrintf(PETSC_COMM_WORLD,
                    "\nFull matrix A:\n");
        PetscCall(printFullMatrix("Dense view of A:",
                                  st.A));

        PetscCall(bgc::destroyState(st));
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace

int main(int argc, char** argv) {
    PetscCallAbort(PETSC_COMM_WORLD,
                   PetscInitialize(&argc,
                                   &argv,
                                   nullptr,
                                   "pretest A/B of the BGC model"));

    PetscPrintf(PETSC_COMM_WORLD,
                "\n######################################################################\n"
                "  Pretest of matrix A and vector B for the BGC model\n"
                "  West: Dirichlet = 1, Neumann = 0\n"
                "  East: Dirichlet = 0, Neumann = 0\n"
                "  Field: phi = 0, source = 0\n"
                "######################################################################\n");

    PetscCallAbort(PETSC_COMM_WORLD,
                   runABPretestCampaign(1.0e-3,
                                        "BGC  Bv=0.001"));
    PetscCallAbort(PETSC_COMM_WORLD,
                   runABPretestCampaign(1.0,
                                        "BGC  Bv=1.0"));

    PetscCallAbort(PETSC_COMM_WORLD, PetscFinalize());
    return 0;
}