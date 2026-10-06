#include "../../Common/MMSCommon.hpp"

#include <petsc.h>

#include <bgclib/Models/TSOM/AssemblyTSOM.hpp>
#include <bgclib/Models/TSOM/CompatTSOM.hpp>
#include <bgclib/Models/TSOM/CoeffTSOM.hpp>
#include <bgclib/Models/TSOM/TruncationErrorTSOM.hpp>
#include <bgclib/SimConfig.hpp>
#include <bgclib/SimState.hpp>
#include <bgclib/Solver.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// ============================================================
//  Parameter set
//  Defaults = P1 from Table 1 of the paper.
//  P2: alpha=0.75, rho=0.25, lambdaC=1.0, lambdaR=1.0/3.0
// ============================================================
struct Parameters {
    PetscReal alpha   {0.50};
    PetscReal rho     {0.50};
    PetscReal theta   {0.5};
    PetscReal lambdaC {1.0};
    PetscReal lambdaR {1.0};

    // Numerics
    PetscReal tf             {1.0};
    PetscReal cdt            {1.024e-1};   // dt = cdt * h^2
    PetscInt  nStepsOverride {0};
    // nVolumes == 0  --> full convergence sweep {8,16,32,64,128,256,512}
    // nVolumes > 0   --> single run with that N
    PetscInt  nVolumes       {0};

    // I/O
    PetscBool   verbose     {PETSC_TRUE};
    PetscBool   printSystem {PETSC_FALSE};
    std::string outputDir   {"Saida/MMS1"};
    std::string inputFile   {"simulation.dat"};
};

// ============================================================
//  Per-run error summary
// ============================================================
struct CaseSummary {
    PetscInt  n        {0};
    PetscReal h        {0.0};
    PetscReal dt       {0.0};
    PetscInt  nSteps   {0};
    PetscReal finalTau {0.0};

    // Global solution errors (L1, L2, Linf) for Psi=U+V, V, U
    PetscReal l1Psi  {0.0};  PetscReal l2Psi  {0.0};  PetscReal linfPsi  {0.0};
    PetscReal l1V    {0.0};  PetscReal l2V    {0.0};  PetscReal linfV    {0.0};
    PetscReal l1U    {0.0};  PetscReal l2U    {0.0};  PetscReal linfU    {0.0};

    // Truncation error at the final time step
    PetscReal lteL2Psi   {0.0};  PetscReal lteLinfPsi   {0.0};
    PetscReal lteL2V     {0.0};  PetscReal lteLinfV     {0.0};
    PetscReal lteL2U     {0.0};  PetscReal lteLinfU     {0.0};
};

// ============================================================
//  String helpers
// ============================================================
[[nodiscard]] static std::string trim(const std::string& s) {
    const auto first = std::find_if_not(s.begin(), s.end(),
        [](unsigned char c){ return std::isspace(c) != 0; });
    if (first == s.end()) return {};
    const auto last = std::find_if_not(s.rbegin(), s.rend(),
        [](unsigned char c){ return std::isspace(c) != 0; }).base();
    return {first, last};
}

[[nodiscard]] static std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
        [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    return s;
}

[[nodiscard]] static PetscBool parseBool(const std::string& v) {
    const std::string lv = toLower(trim(v));
    if (lv == "true"  || lv == "yes" || lv == "on"  || lv == "1") return PETSC_TRUE;
    if (lv == "false" || lv == "no"  || lv == "off" || lv == "0") return PETSC_FALSE;
    throw std::runtime_error("Invalid boolean value: \"" + v + "\"");
}

static void setParameter(const std::string& key,
                         const std::string& val,
                         Parameters&        p) {
    const std::string k = toLower(trim(key));
    const std::string v = trim(val);

    if      (k == "alpha"        || k == "alpha_tsom")       p.alpha          = std::stod(v);
    else if (k == "rho"          || k == "rho_tsom")         p.rho            = std::stod(v);
    else if (k == "theta"        || k == "theta_tsom")       p.theta          = std::stod(v);
    else if (k == "lambda_c"     || k == "lambdac")          p.lambdaC        = std::stod(v);
    else if (k == "lambda_r"     || k == "lambdar")          p.lambdaR        = std::stod(v);
    else if (k == "nvol"         || k == "nvolumes"
                                 || k == "n_volumes")        p.nVolumes       = static_cast<PetscInt>(std::stoll(v));
    else if (k == "tf"           || k == "final_time")       p.tf             = std::stod(v);
    else if (k == "cdt")                                     p.cdt            = std::stod(v);
    else if (k == "n_steps"      || k == "nsteps")           p.nStepsOverride = static_cast<PetscInt>(std::stoll(v));
    else if (k == "verbose")                                 p.verbose        = parseBool(v);
    else if (k == "print_system" || k == "printsystem")      p.printSystem    = parseBool(v);
    else if (k == "output_dir"   || k == "outputdir")        p.outputDir      = v;
    else if (k == "input_file"   || k == "inputfile")        p.inputFile      = v;
    else throw std::runtime_error("Unknown key in input file: \"" + key + "\"");
}

static void readInputFile(const std::filesystem::path& filepath, Parameters& p) {
    std::ifstream in(filepath);
    if (!in.is_open())
        throw std::runtime_error("Cannot open input file: " + filepath.string());

    std::string line;
    std::size_t lineNumber = 0;
    while (std::getline(in, line)) {
        ++lineNumber;
        const auto sharp = line.find('#');
        if (sharp != std::string::npos) line.erase(sharp);
        const std::string cl = trim(line);
        if (cl.empty()) continue;

        std::string key, val;
        const auto eq = cl.find('=');
        if (eq != std::string::npos) {
            key = trim(cl.substr(0, eq));
            val = trim(cl.substr(eq + 1));
        } else {
            std::istringstream iss(cl);
            if (!(iss >> key)) continue;
            if (!(iss >> val))
                throw std::runtime_error(
                    "Bad line " + std::to_string(lineNumber)
                    + ": expected 'key = value' or 'key value'.");
        }
        try { setParameter(key, val, p); }
        catch (const std::exception& e) {
            throw std::runtime_error(
                "Error at line " + std::to_string(lineNumber) + ": " + e.what());
        }
    }
}

// ============================================================
//  MMS-1 manufactured solution
//
//  phi = U(xi, tau) = exp(-tau) * sin(pi * xi)   [mobile]
//  mu  = V(xi, tau) = exp(-tau) * sin(pi * xi)   [retained]
//  Psi = U + V      = 2 * exp(-tau) * sin(pi * xi)
//
//  BCs: Dirichlet homogeneous
//       alpha_w = alpha_e = 1,  beta_w = beta_e = 0,  gamma = 0
//  (Psi(0,tau) = Psi(1,tau) = 0 because sin(0)=sin(pi)=0)
// ============================================================

[[nodiscard]] static inline PetscReal gMMS1(PetscReal x) noexcept {
    return std::sin(std::numbers::pi_v<PetscReal> * x);
}

[[nodiscard]] static inline PetscReal gppMMS1(PetscReal x) noexcept {
    const PetscReal pi = std::numbers::pi_v<PetscReal>;
    return -(pi * pi) * std::sin(pi * x);
}

// phi = U (mobile population)
[[nodiscard]] static inline PetscReal uExact(PetscReal x, PetscReal tau) noexcept {
    return std::exp(-tau) * gMMS1(x);
}

// mu = V (retained population)
[[nodiscard]] static inline PetscReal vExact(PetscReal x, PetscReal tau) noexcept {
    return std::exp(-tau) * gMMS1(x);
}

// Psi = U + V
[[nodiscard]] static inline PetscReal psiExact(PetscReal x, PetscReal tau) noexcept {
    return 2.0 * std::exp(-tau) * gMMS1(x);
}

// ============================================================
//  Source terms
//
//  The assembled system (from CoeffTSOM, interior stencils) implements:
//
//    U_tau + lambdaC*U - lambdaR*V - d11*U_xx - d12*V_xx = S_U   ...(phi eq.)
//    V_tau - lambdaC*U + lambdaR*V - d21*U_xx - d22*V_xx = S_V   ...(mu  eq.)
//
//  Substituting U = V = exp(-tau)*g(xi) and rearranging:
//
//    S_U = exp(-tau) * [ (-1 + lambdaC - lambdaR)*g  -  (d11+d12)*g'' ]
//    S_V = exp(-tau) * [ (-1 - lambdaC + lambdaR)*g  -  (d21+d22)*g'' ]
//
//  For MMS-1: g = sin(pi*xi),  g'' = -pi^2 * sin(pi*xi).
// ============================================================

[[nodiscard]] static PetscReal sourceUFn(const bgc::SimConfig& cfg,
                                         PetscReal             x,
                                         PetscReal             tau) noexcept {
    const auto      d   = bgc::computeDiffusionCoefficientsTSOM(cfg);
    const PetscReal g   = gMMS1(x);
    const PetscReal gpp = gppMMS1(x);
    return std::exp(-tau) * ((-1.0 + cfg.lambdaC - cfg.lambdaR) * g
                             - (d.d11 + d.d12) * gpp);
}

[[nodiscard]] static PetscReal sourceVFn(const bgc::SimConfig& cfg,
                                         PetscReal             x,
                                         PetscReal             tau) noexcept {
    const auto      d   = bgc::computeDiffusionCoefficientsTSOM(cfg);
    const PetscReal g   = gMMS1(x);
    const PetscReal gpp = gppMMS1(x);
    return std::exp(-tau) * ((-1.0 - cfg.lambdaC + cfg.lambdaR) * g
                             - (d.d21 + d.d22) * gpp);
}

// ============================================================
//  SimConfig helpers
// ============================================================

// 1-based: P = 1, ..., n  -->  x_P = (P - 0.5) * h
[[nodiscard]] static inline PetscReal xCentre(PetscInt P, PetscInt n) noexcept {
    const PetscReal h = 1.0 / static_cast<PetscReal>(n);
    return (static_cast<PetscReal>(P) - 0.5) * h;
}

[[nodiscard]] static bgc::SimConfig makeConfig(const Parameters& p,
                                               PetscInt          n,
                                               PetscReal         h,
                                               PetscReal         dt,
                                               PetscInt          nTimes) {
    bgc::SimConfig cfg;
    cfg.model   = bgc::Model::TSPV;
    cfg.nx      = n;
    cfg.lx      = 1.0;
    cfg.x0      = 0.0;
    cfg.h       = h;
    cfg.tf      = dt * static_cast<PetscReal>(nTimes);
    cfg.dt      = dt;
    cfg.nTimes  = nTimes;

    cfg.alphaTSOM = p.alpha;
    cfg.rhoTSOM   = p.rho;
    cfg.thetaTSOM = p.theta;
    cfg.lambdaC   = p.lambdaC;
    cfg.lambdaR   = p.lambdaR;

    // Dirichlet homogeneous: alpha=1, beta=0; gamma is set per step.
    cfg.alphaPsiWest = 1.0;
    cfg.betaPsiWest  = 0.0;
    cfg.alphaPsiEast = 1.0;
    cfg.betaPsiEast  = 0.0;

    cfg.useDirectSolver     = true;
    cfg.verbose             = p.verbose;
    cfg.directSolverBackend = bgc::DirectSolverBackend::PetscDefault;
    return cfg;
}

// gamma = Psi at boundary = 0 for all tau (sin(0)=sin(pi)=0).
[[nodiscard]] static bgc::SimConfig makeStepConfig(const bgc::SimConfig& base,
                                                   PetscReal             /*tau*/) {
    bgc::SimConfig cfg = base;
    cfg.gammaPsiWest = 0.0;
    cfg.gammaPsiEast = 0.0;
    return cfg;
}

// ============================================================
//  PETSc overrides and parameter printing
// ============================================================

static PetscErrorCode applyPetscOverrides(Parameters& p) {
    PetscFunctionBeginUser;

    PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-alpha",    &p.alpha,          nullptr));
    PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-rho",      &p.rho,            nullptr));
    PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-theta",    &p.theta,          nullptr));
    PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-lambda_c", &p.lambdaC,        nullptr));
    PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-lambda_r", &p.lambdaR,        nullptr));
    PetscCall(PetscOptionsGetInt (nullptr, nullptr, "-nvol",     &p.nVolumes,       nullptr));
    PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-tf",       &p.tf,             nullptr));
    PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-cdt",      &p.cdt,            nullptr));
    PetscCall(PetscOptionsGetInt (nullptr, nullptr, "-n_steps",  &p.nStepsOverride, nullptr));
    PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-verbose",       &p.verbose,     nullptr));
    PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-print_system",  &p.printSystem, nullptr));

    char    outbuf[PETSC_MAX_PATH_LEN];
    PetscBool set = PETSC_FALSE;
    std::snprintf(outbuf, sizeof(outbuf), "%s", p.outputDir.c_str());
    PetscCall(PetscOptionsGetString(nullptr, nullptr, "-output_dir",
                                    outbuf, sizeof(outbuf), &set));
    if (set) p.outputDir = outbuf;

    PetscCheck(p.nVolumes >= 0,
               PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
               "nVolumes must be >= 0 (0 = convergence sweep). Got %" PetscInt_FMT ".",
               p.nVolumes);
    PetscCheck(p.nVolumes == 0 || p.nVolumes >= 2,
               PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
               "If nVolumes > 0 it must be >= 2. Got %" PetscInt_FMT ".", p.nVolumes);
    PetscCheck(p.tf > 0.0,
               PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
               "tf must be positive. Got %.16e.", static_cast<double>(p.tf));
    PetscCheck(p.cdt > 0.0,
               PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
               "cdt must be positive. Got %.16e.", static_cast<double>(p.cdt));
    PetscCheck(p.nStepsOverride >= 0,
               PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
               "n_steps override must be >= 0. Got %" PetscInt_FMT ".", p.nStepsOverride);
    PetscCheck(p.alpha >= 0.0 && p.alpha <= 1.0,
               PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
               "alpha must be in [0,1]. Got %.16e.", static_cast<double>(p.alpha));
    PetscCheck(p.rho >= 0.0 && p.rho <= 1.0,
               PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
               "rho must be in [0,1]. Got %.16e.", static_cast<double>(p.rho));
    PetscCheck(p.theta >= 0.0 && p.theta <= 1.0,
               PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
               "theta must be in [0,1]. Got %.16e.", static_cast<double>(p.theta));
    PetscCheck(p.lambdaC >= 0.0,
               PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
               "lambda_c must be >= 0. Got %.16e.", static_cast<double>(p.lambdaC));
    PetscCheck(p.lambdaR >= 0.0,
               PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
               "lambda_r must be >= 0. Got %.16e.", static_cast<double>(p.lambdaR));

    PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode printLoadedParameters(const Parameters& p) {
    PetscFunctionBeginUser;
    PetscCall(PetscPrintf(
        PETSC_COMM_WORLD,
        "\n[MMS-1 TSOM] Parameters\n"
        "---------------------------------------------\n"
        "input_file    = %s\n"
        "alpha         = %.6e\n"
        "rho           = %.6e\n"
        "theta         = %.6e\n"
        "lambda_c      = %.6e\n"
        "lambda_r      = %.6e\n"
        "tf            = %.6e\n"
        "cdt           = %.6e\n"
        "n_steps       = %" PetscInt_FMT "\n"
        "nvol          = %" PetscInt_FMT "  (0 = convergence sweep)\n"
        "verbose       = %s\n"
        "print_system  = %s\n"
        "output_dir    = %s\n"
        "---------------------------------------------\n",
        p.inputFile.c_str(),
        static_cast<double>(p.alpha),
        static_cast<double>(p.rho),
        static_cast<double>(p.theta),
        static_cast<double>(p.lambdaC),
        static_cast<double>(p.lambdaR),
        static_cast<double>(p.tf),
        static_cast<double>(p.cdt),
        p.nStepsOverride,
        p.nVolumes,
        p.verbose     ? "true" : "false",
        p.printSystem ? "true" : "false",
        p.outputDir.c_str()));
    PetscFunctionReturn(PETSC_SUCCESS);
}

// ============================================================
//  Field extraction
// ============================================================

static PetscErrorCode extractFields(const bgc::SimConfig&   cfg,
                                    bgc::SimState&          st,
                                    std::vector<PetscReal>& uVec,
                                    std::vector<PetscReal>& vVec) {
    PetscFunctionBeginUser;
    uVec.assign(static_cast<std::size_t>(cfg.nx), 0.0);
    vVec.assign(static_cast<std::size_t>(cfg.nx), 0.0);

    PetscInt xs = 0, xe = 0;
    PetscCall(VecGetOwnershipRange(st.phi, &xs, &xe));

    const PetscScalar* ua = nullptr;
    const PetscScalar* va = nullptr;
    PetscCall(VecGetArrayRead(st.phi, &ua));
    PetscCall(VecGetArrayRead(st.mu,  &va));

    for (PetscInt i = xs; i < xe; ++i) {
        const std::size_t k = static_cast<std::size_t>(i);
        uVec[k] = static_cast<PetscReal>(ua[i - xs]);
        vVec[k] = static_cast<PetscReal>(va[i - xs]);
    }

    PetscCall(VecRestoreArrayRead(st.phi, &ua));
    PetscCall(VecRestoreArrayRead(st.mu,  &va));
    PetscFunctionReturn(PETSC_SUCCESS);
}

// ============================================================
//  Error norms: L1, L2, Linf for Psi, V, U
// ============================================================

static PetscErrorCode computeSummary(const bgc::SimConfig&   cfg,
                                     bgc::SimState&          st,
                                     PetscReal               tau,
                                     CaseSummary&            s,
                                     std::vector<PetscReal>& uVec,
                                     std::vector<PetscReal>& vVec) {
    PetscFunctionBeginUser;
    PetscCall(extractFields(cfg, st, uVec, vVec));

    PetscReal sL1Psi = 0.0, sL2Psi = 0.0, linfPsi = 0.0;
    PetscReal sL1V   = 0.0, sL2V   = 0.0, linfV   = 0.0;
    PetscReal sL1U   = 0.0, sL2U   = 0.0, linfU   = 0.0;

    for (PetscInt P = 1; P <= cfg.nx; ++P) {
        const PetscReal   x   = xCentre(P, cfg.nx);
        const std::size_t k   = static_cast<std::size_t>(P - 1);

        const PetscReal uNum   = uVec[k];
        const PetscReal vNum   = vVec[k];
        const PetscReal psiNum = uNum + vNum;

        const PetscReal uEx    = uExact(x, tau);
        const PetscReal vEx    = vExact(x, tau);
        const PetscReal psiEx  = psiExact(x, tau);

        const PetscReal ePsi = std::abs(psiNum - psiEx);
        const PetscReal eV   = std::abs(vNum   - vEx);
        const PetscReal eU   = std::abs(uNum   - uEx);

        sL1Psi += ePsi;  sL2Psi += ePsi * ePsi;  linfPsi = std::max(linfPsi, ePsi);
        sL1V   += eV;    sL2V   += eV   * eV;    linfV   = std::max(linfV,   eV);
        sL1U   += eU;    sL2U   += eU   * eU;    linfU   = std::max(linfU,   eU);
    }

    const PetscReal h = cfg.h;
    s.l1Psi   = h * sL1Psi;
    s.l2Psi   = std::sqrt(h * sL2Psi);
    s.linfPsi = linfPsi;
    s.l1V     = h * sL1V;
    s.l2V     = std::sqrt(h * sL2V);
    s.linfV   = linfV;
    s.l1U     = h * sL1U;
    s.l2U     = std::sqrt(h * sL2U);
    s.linfU   = linfU;

    PetscFunctionReturn(PETSC_SUCCESS);
}

// ============================================================
//  Output helpers
// ============================================================

static PetscErrorCode writeSummaryCsv(const std::filesystem::path& path,
                                      const CaseSummary&           r) {
    PetscFunctionBeginUser;
    std::ofstream out(path);
    if (!out.is_open())
        throw std::runtime_error("Cannot open for writing: " + path.string());
    out << std::setprecision(16);
    out << "N,h,dt,n_steps,final_tau,"
           "L1_Psi,L2_Psi,Linf_Psi,"
           "L1_V,L2_V,Linf_V,"
           "L1_U,L2_U,Linf_U,"
           "LTE_L2_Psi,LTE_Linf_Psi,"
           "LTE_L2_V,LTE_Linf_V,"
           "LTE_L2_U,LTE_Linf_U\n";
    out << r.n        << ',' << r.h        << ',' << r.dt       << ','
        << r.nSteps   << ',' << r.finalTau << ','
        << r.l1Psi    << ',' << r.l2Psi    << ',' << r.linfPsi  << ','
        << r.l1V      << ',' << r.l2V      << ',' << r.linfV    << ','
        << r.l1U      << ',' << r.l2U      << ',' << r.linfU    << ','
        << r.lteL2Psi << ',' << r.lteLinfPsi << ','
        << r.lteL2V   << ',' << r.lteLinfV   << ','
        << r.lteL2U   << ',' << r.lteLinfU   << '\n';
    PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode writeFieldFile(const std::filesystem::path&  path,
                                     PetscInt                      n,
                                     const std::vector<PetscReal>& uVec,
                                     const std::vector<PetscReal>& vVec,
                                     PetscReal                     tau) {
    PetscFunctionBeginUser;
    std::ofstream out(path);
    if (!out.is_open())
        throw std::runtime_error("Cannot open for writing: " + path.string());
    out << std::setprecision(16);
    out << "# P x u_num u_exact u_err v_num v_exact v_err"
           " psi_num psi_exact psi_err\n";
    for (PetscInt P = 1; P <= n; ++P) {
        const PetscReal   x   = xCentre(P, n);
        const std::size_t k   = static_cast<std::size_t>(P - 1);
        const PetscReal   uN  = uVec[k];
        const PetscReal   vN  = vVec[k];
        const PetscReal   pN  = uN + vN;
        const PetscReal   uE  = uExact(x, tau);
        const PetscReal   vE  = vExact(x, tau);
        const PetscReal   pE  = psiExact(x, tau);
        out << P       << ' ' << x       << ' '
            << uN      << ' ' << uE      << ' ' << (uN - uE) << ' '
            << vN      << ' ' << vE      << ' ' << (vN - vE) << ' '
            << pN      << ' ' << pE      << ' ' << (pN - pE) << '\n';
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode writeTruncationErrorFile(
        const std::filesystem::path&  path,
        PetscInt                      n,
        const std::vector<PetscReal>& tauU,
        const std::vector<PetscReal>& tauV,
        const std::vector<PetscReal>& tauPsi) {
    PetscFunctionBeginUser;
    std::ofstream out(path);
    if (!out.is_open())
        throw std::runtime_error("Cannot open for writing: " + path.string());
    out << std::setprecision(16);
    out << "# P x_center tau_U tau_V tau_Psi\n";
    for (PetscInt P = 1; P <= n; ++P) {
        const PetscReal   x = xCentre(P, n);
        const std::size_t k = static_cast<std::size_t>(P - 1);
        out << P << ' ' << x << ' '
            << tauU[k] << ' ' << tauV[k] << ' ' << tauPsi[k] << '\n';
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

static void printSummary(const CaseSummary& r) {
    std::ostringstream oss;
    oss << std::scientific << std::setprecision(5);
    oss << "\n  MMS-1 TSOM  N = " << r.n << "\n";
    oss << "  h = " << r.h << "  dt = " << r.dt
        << "  steps = " << r.nSteps << "\n";
    oss << "  Var     L1              L2              Linf\n";
    oss << "  Psi   " << r.l1Psi  << "  " << r.l2Psi  << "  " << r.linfPsi  << "\n";
    oss << "  V     " << r.l1V    << "  " << r.l2V    << "  " << r.linfV    << "\n";
    oss << "  U     " << r.l1U    << "  " << r.l2U    << "  " << r.linfU    << "\n";
    PetscPrintf(PETSC_COMM_WORLD, "%s", oss.str().c_str());
}

// ============================================================
//  Convergence table
// ============================================================

[[nodiscard]] static PetscReal convRate(PetscReal eCoarse,
                                        PetscReal eFine) noexcept {
    if (eCoarse <= 0.0 || eFine <= 0.0) return 0.0;
    return std::log2(eCoarse / eFine);
}

static void writeConvergenceTable(const std::filesystem::path&    path,
                                  const std::vector<CaseSummary>& rows) {
    std::ofstream out(path);
    if (!out.is_open())
        throw std::runtime_error("Cannot open for writing: " + path.string());
    out << std::setprecision(6) << std::scientific;
    out << "N,h,dt,"
           "L1_Psi,rL1_Psi,L2_Psi,rL2_Psi,Linf_Psi,rLinf_Psi,"
           "L1_V,rL1_V,L2_V,rL2_V,Linf_V,rLinf_V,"
           "L1_U,rL1_U,L2_U,rL2_U,Linf_U,rLinf_U\n";

    for (std::size_t i = 0; i < rows.size(); ++i) {
        const auto& r = rows[i];
        auto rStr = [&](PetscReal ec, PetscReal ef) -> std::string {
            if (i == 0) return "---";
            std::ostringstream s;
            s << std::fixed << std::setprecision(3) << convRate(ec, ef);
            return s.str();
        };
        const auto& p = (i > 0) ? rows[i - 1] : r;
        out << r.n        << ',' << r.h        << ',' << r.dt       << ','
            << r.l1Psi    << ',' << rStr(p.l1Psi,    r.l1Psi)    << ','
            << r.l2Psi    << ',' << rStr(p.l2Psi,    r.l2Psi)    << ','
            << r.linfPsi  << ',' << rStr(p.linfPsi,  r.linfPsi)  << ','
            << r.l1V      << ',' << rStr(p.l1V,      r.l1V)      << ','
            << r.l2V      << ',' << rStr(p.l2V,      r.l2V)      << ','
            << r.linfV    << ',' << rStr(p.linfV,    r.linfV)    << ','
            << r.l1U      << ',' << rStr(p.l1U,      r.l1U)      << ','
            << r.l2U      << ',' << rStr(p.l2U,      r.l2U)      << ','
            << r.linfU    << ',' << rStr(p.linfU,    r.linfU)    << '\n';
    }
}

static void printConvergenceTable(const std::vector<CaseSummary>& rows) {
    // Helper: format one table section for a given variable
    auto printSection = [&](const char*    varName,
                             std::function<std::array<PetscReal,3>(const CaseSummary&)> getErr)
    {
        std::ostringstream oss;
        oss << "\n  " << varName << "\n";
        const char* hdr = "  N     h           L1          rate  "
                          "L2          rate  Linf        rate\n";
        oss << hdr;
        oss << "  " << std::string(74, '-') << "\n";
        for (std::size_t i = 0; i < rows.size(); ++i) {
            const auto errs = getErr(rows[i]);
            oss << std::scientific << std::setprecision(3);
            oss << "  " << std::left << std::setw(5) << rows[i].n
                << rows[i].h;
            for (int j = 0; j < 3; ++j) {
                oss << "  " << errs[j];
                if (i > 0) {
                    const auto prev = getErr(rows[i - 1]);
                    oss << std::fixed << std::setprecision(2)
                        << "  " << convRate(prev[j], errs[j]);
                } else {
                    oss << "   ---";
                }
            }
            oss << "\n";
        }
        PetscPrintf(PETSC_COMM_WORLD, "%s", oss.str().c_str());
    };

    PetscPrintf(PETSC_COMM_WORLD,
        "\n%s\n  MMS-1 TSOM -- Convergence Table\n%s\n",
        std::string(80, '=').c_str(), std::string(80, '=').c_str());

    printSection("Psi = U + V", [](const CaseSummary& r) -> std::array<PetscReal,3> {
        return {r.l1Psi, r.l2Psi, r.linfPsi}; });
    printSection("V (retained)", [](const CaseSummary& r) -> std::array<PetscReal,3> {
        return {r.l1V, r.l2V, r.linfV}; });
    printSection("U = Psi - V (mobile)", [](const CaseSummary& r) -> std::array<PetscReal,3> {
        return {r.l1U, r.l2U, r.linfU}; });

    PetscPrintf(PETSC_COMM_WORLD, "  %s\n", std::string(78, '=').c_str());
}

// ============================================================
//  Time-marching loop
// ============================================================

template <typename StepConfigFn>
static PetscErrorCode runTransientLoop(const bgc::SimConfig& cfg,
                                       bgc::SimState&        st,
                                       const bgc::FieldFn&   srcU,
                                       const bgc::FieldFn&   srcV,
                                       StepConfigFn&&        stepCfgFn) {
    PetscFunctionBeginUser;

    // Condições iniciais: phi = U(x,0),  mu = V(x,0)
    {
        PetscInt xs = 0, xe = 0;
        PetscCall(VecGetOwnershipRange(st.phi, &xs, &xe));

        const PetscInt             localN = xe - xs;
        std::vector<PetscInt>    idx(static_cast<std::size_t>(localN));
        std::vector<PetscScalar> valU(static_cast<std::size_t>(localN));
        std::vector<PetscScalar> valV(static_cast<std::size_t>(localN));

        for (PetscInt i = xs; i < xe; ++i) {
            const std::size_t k = static_cast<std::size_t>(i - xs);
            const PetscReal   x = xCentre(i + 1, cfg.nx);  // 1-based
            idx[k]  = i;
            valU[k] = uExact(x, 0.0);
            valV[k] = vExact(x, 0.0);
        }

        PetscCall(VecSetValues(st.phi, localN, idx.data(), valU.data(), INSERT_VALUES));
        PetscCall(VecSetValues(st.mu,  localN, idx.data(), valV.data(), INSERT_VALUES));
        PetscCall(VecAssemblyBegin(st.phi));  PetscCall(VecAssemblyEnd(st.phi));
        PetscCall(VecAssemblyBegin(st.mu));   PetscCall(VecAssemblyEnd(st.mu));
    }
    PetscCall(VecCopy(st.phi, st.phi_0));
    PetscCall(VecCopy(st.mu,  st.mu_0));

    for (PetscInt step = 1; step <= cfg.nTimes; ++step) {
        const PetscReal      tauNp1  = static_cast<PetscReal>(step) * cfg.dt;
        const bgc::SimConfig stepCfg = stepCfgFn(cfg, tauNp1);
        const auto           rhs     = bgc::computeRHSTSOM(stepCfg, tauNp1);

        PetscCall(bgc::assembleFullRHSTSOM(stepCfg, st, rhs, tauNp1, srcU, srcV));
        PetscCall(bgc::solveLinearSystem(stepCfg, st));

        if (step < cfg.nTimes) {
            PetscCall(VecCopy(st.phi, st.phi_0));
            PetscCall(VecCopy(st.mu,  st.mu_0));
        }
    }

    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "[time-march] model=TSOM  nx=%" PetscInt_FMT
        "  tf=%.6e  dt=%.6e  nTimes=%" PetscInt_FMT "\n",
        cfg.nx,
        static_cast<double>(cfg.tf),
        static_cast<double>(cfg.dt),
        cfg.nTimes));

    PetscFunctionReturn(PETSC_SUCCESS);
}
// ============================================================
//  Single-case driver
// ============================================================

static PetscErrorCode runCase(const Parameters&            p,
                              PetscInt                     n,
                              const std::filesystem::path& outDir,
                              CaseSummary&                 summary) {
    PetscFunctionBeginUser;

    const PetscReal h          = 1.0 / static_cast<PetscReal>(n);
    const PetscReal dtEstimate = p.cdt * h * h;
    PetscInt nSteps = 0;
    if (p.nStepsOverride > 0) {
        nSteps = p.nStepsOverride;
    } else {
        nSteps = static_cast<PetscInt>(std::ceil(p.tf / dtEstimate));
        nSteps = std::max<PetscInt>(nSteps, 1);
    }
    const PetscReal dt       = p.tf / static_cast<PetscReal>(nSteps);
    const PetscReal finalTau = dt   * static_cast<PetscReal>(nSteps);

    bgc::SimConfig cfg = makeConfig(p, n, h, dt, nSteps);

    bgc::SimState st;
    PetscCall(bgc::createStateTSUV(cfg, st));

    const auto coeff = bgc::computeCoefficientsTSOM(cfg);
    PetscCall(bgc::assembleMatrixTSOM(cfg, st, coeff));
    PetscCall(bgc::configureSolver(cfg, st));

    if (p.printSystem) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n[DEBUG] Global matrix A\n"));
        Mat flat = nullptr;
        PetscCall(MatConvert(st.A, MATAIJ, MAT_INITIAL_MATRIX, &flat));
        PetscCall(MatView(flat, PETSC_VIEWER_STDOUT_WORLD));
        PetscCall(MatDestroy(&flat));
    }

    const bgc::FieldFn srcU = [](const bgc::SimConfig& c, PetscReal x, PetscReal tau) {
        return sourceUFn(c, x, tau);
    };
    const bgc::FieldFn srcV = [](const bgc::SimConfig& c, PetscReal x, PetscReal tau) {
        return sourceVFn(c, x, tau);
    };

    PetscCall(runTransientLoop(
        cfg, st, srcU, srcV,
        [](const bgc::SimConfig& c, PetscReal tau) {
            return makeStepConfig(c, tau);
        }));

    summary.n        = n;
    summary.h        = h;
    summary.dt       = dt;
    summary.nSteps   = nSteps;
    summary.finalTau = finalTau;

    std::vector<PetscReal> uVec, vVec;
    PetscCall(computeSummary(cfg, st, finalTau, summary, uVec, vVec));

    // ------- Truncation error at the final step -------
    {
        const bgc::SimConfig cfgFinal = makeStepConfig(cfg, finalTau);
        bgc::TSOMTruncationError te;
        std::vector<PetscReal> tauU, tauV;

        PetscCall(bgc::computeTruncationErrorTSOM(
            cfgFinal,
            [](const bgc::SimConfig&, PetscReal x, PetscReal tau) { return uExact(x, tau); },
            [](const bgc::SimConfig&, PetscReal x, PetscReal tau) { return vExact(x, tau); },
            srcU,
            srcV,
            finalTau - dt,
            finalTau,
            te,
            &tauU,
            &tauV));

        std::vector<PetscReal> tauPsi(static_cast<std::size_t>(n), 0.0);
        PetscReal sL2U = 0.0, sL2V = 0.0, sL2Psi = 0.0;
        PetscReal linfU = 0.0, linfV = 0.0, linfPsi = 0.0;

        for (PetscInt i = 0; i < n; ++i) {
            const std::size_t k    = static_cast<std::size_t>(i);
            const PetscReal   tU   = tauU[k];
            const PetscReal   tV   = tauV[k];
            const PetscReal   tPsi = tU + tV;
            tauPsi[k] = tPsi;
            sL2U   += tU   * tU;
            sL2V   += tV   * tV;
            sL2Psi += tPsi * tPsi;
            linfU    = std::max(linfU,   std::abs(tU));
            linfV    = std::max(linfV,   std::abs(tV));
            linfPsi  = std::max(linfPsi, std::abs(tPsi));
        }

        summary.lteL2U     = std::sqrt(h * sL2U);
        summary.lteLinfU   = linfU;
        summary.lteL2V     = std::sqrt(h * sL2V);
        summary.lteLinfV   = linfV;
        summary.lteL2Psi   = std::sqrt(h * sL2Psi);
        summary.lteLinfPsi = linfPsi;

        const std::filesystem::path ltePath =
            outDir / ("mms1_lte_N" + std::to_string(n) + ".dat");
        PetscCall(writeTruncationErrorFile(ltePath, n, tauU, tauV, tauPsi));
    }

    // ------- Field file -------
    {
        const std::filesystem::path fieldPath =
            outDir / ("mms1_fields_N" + std::to_string(n) + ".dat");
        PetscCall(writeFieldFile(fieldPath, n, uVec, vVec, finalTau));
    }

    // ------- Per-case CSV -------
    {
        const std::filesystem::path csvPath =
            outDir / ("mms1_summary_N" + std::to_string(n) + ".csv");
        PetscCall(writeSummaryCsv(csvPath, summary));
    }

    PetscCall(bgc::destroyState(st));
    PetscFunctionReturn(PETSC_SUCCESS);
}

// ============================================================
//  Convergence sweep
// ============================================================

static const std::array<PetscInt, 7> kSweepMeshes = {8, 16, 32, 64, 128, 256, 512};

static PetscErrorCode runConvergenceSweep(const Parameters&            p,
                                          const std::filesystem::path& outDir) {
    PetscFunctionBeginUser;

    std::vector<CaseSummary> rows;
    rows.reserve(kSweepMeshes.size());

    for (const PetscInt n : kSweepMeshes) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "\n[MMS-1 TSOM] ---- N = %" PetscInt_FMT " ----\n", n));
        CaseSummary s;
        PetscCall(runCase(p, n, outDir, s));
        if (p.verbose) printSummary(s);
        rows.push_back(s);
    }

    printConvergenceTable(rows);

    const std::filesystem::path convPath = outDir / "mms1_convergence.csv";
    writeConvergenceTable(convPath, rows);
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "\n[MMS-1 TSOM] Convergence table written to: %s\n",
        convPath.string().c_str()));

    PetscFunctionReturn(PETSC_SUCCESS);
}

} // anonymous namespace

// ============================================================
//  main
// ============================================================
int main(int argc, char** argv) {
    PetscCallAbort(PETSC_COMM_WORLD,
        PetscInitialize(&argc, &argv, nullptr,
            "MMS1TSOM -- MMS-1 convergence study for the TSOM model\n"
            "  Manufactured solution:  U = V = exp(-tau)*sin(pi*xi)\n"
            "  Boundary conditions:    Dirichlet homogeneous\n"
            "  Default param set P1:   alpha=0.5, rho=0.5, lambdaC=lambdaR=1\n"
            "  Default behaviour:      convergence sweep N in {8,16,32,64,128,256,512}\n"
            "  Single run:             pass -nvol N\n"));

    PetscInt exitCode = 0;
    try {
        PetscMPIInt size = 1;
        PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
        if (size != 1)
            throw std::runtime_error(
                "This MMS driver is sequential. Run with a single MPI process.");

        Parameters params;

        char inputBuf[PETSC_MAX_PATH_LEN] = "simulation.dat";
        PetscBool inputSet = PETSC_FALSE;
        PetscCall(PetscOptionsGetString(nullptr, nullptr, "-input_file",
                                        inputBuf, sizeof(inputBuf), &inputSet));
        if (inputSet) params.inputFile = inputBuf;

        readInputFile(params.inputFile, params);
        PetscCallAbort(PETSC_COMM_WORLD, applyPetscOverrides(params));
        PetscCallAbort(PETSC_COMM_WORLD, printLoadedParameters(params));

        const std::filesystem::path outDir = params.outputDir;
        std::filesystem::create_directories(outDir);

        if (params.nVolumes > 0) {
            // Single-case mode: useful for debugging a specific N.
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                "[MMS-1 TSOM] Single run  N = %" PetscInt_FMT "\n",
                params.nVolumes));
            CaseSummary summary;
            PetscCallAbort(PETSC_COMM_WORLD,
                runCase(params, params.nVolumes, outDir, summary));
            printSummary(summary);
            PetscCallAbort(PETSC_COMM_WORLD,
                writeSummaryCsv(outDir / "mms1_summary.csv", summary));
        } else {
            // Convergence sweep (default).
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                "[MMS-1 TSOM] Convergence sweep"
                "  N in {8, 16, 32, 64, 128, 256, 512}\n"));
            PetscCallAbort(PETSC_COMM_WORLD,
                runConvergenceSweep(params, outDir));
        }

        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "\nOutput written to: %s\n", outDir.string().c_str()));

    } catch (const std::exception& ex) {
        PetscPrintf(PETSC_COMM_WORLD, "\nERROR: %s\n", ex.what());
        exitCode = 1;
    }

    PetscCallAbort(PETSC_COMM_WORLD, PetscFinalize());
    return exitCode;
}