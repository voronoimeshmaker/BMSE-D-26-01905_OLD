#pragma once

#include <bgclib/Analytics.hpp>
#include <bgclib/Core/BoundaryCondition.hpp>
#include <bgclib/Models/BGC/AssemblyBGC.hpp>
#include <bgclib/Models/BGC/CoeffBGC.hpp>
#include <bgclib/Models/TBGC/AssemblyTBGC.hpp>
#include <bgclib/Models/TBGC/CoeffTBGC.hpp>
#include <bgclib/Models/TBGC/AssemblyTBGC.hpp>
#include <bgclib/Models/TBGC/CoeffTBGC.hpp>
#include <bgclib/Models/TBGCS/AssemblyTBGCS.hpp>
#include <bgclib/Models/TBGCS/CoeffTBGCS.hpp>   
#include <bgclib/PostProcess.hpp>
#include <bgclib/SimConfig.hpp>
#include <bgclib/SimState.hpp>
#include <bgclib/Solver.hpp>

#include <petsc.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace paper_mms {

inline PetscBool parsePetscBool(const std::string& text) {
    std::string cleaned = text;

    cleaned.erase(cleaned.begin(),
                  std::find_if(cleaned.begin(),
                               cleaned.end(),
                               [](unsigned char ch) {
                                   return !std::isspace(ch);
                               }));

    cleaned.erase(std::find_if(cleaned.rbegin(),
                               cleaned.rend(),
                               [](unsigned char ch) {
                                   return !std::isspace(ch);
                               }).base(),
                  cleaned.end());

    std::transform(cleaned.begin(),
                   cleaned.end(),
                   cleaned.begin(),
                   [](unsigned char ch) {
                       return static_cast<char>(std::tolower(ch));
                   });

    if (cleaned == "true"  ||
        cleaned == "yes"   ||
        cleaned == "on"    ||
        cleaned == "1"     ||
        cleaned == "petsc_true") {
        return PETSC_TRUE;
    }

    if (cleaned == "false" ||
        cleaned == "no"    ||
        cleaned == "off"   ||
        cleaned == "0"     ||
        cleaned == "petsc_false") {
        return PETSC_FALSE;
    }

    throw std::runtime_error("Invalid PetscBool value: " + cleaned);
}

struct InputData {
    PetscReal tf = 1.0e-3;
    PetscReal cdt = 1.024e-1;
    PetscBool verbose = PETSC_FALSE;
    std::vector<PetscInt> nxList = {8, 16, 32, 64};
    std::vector<PetscReal> bvList = {
        static_cast<PetscReal>(1.0e-3),
        static_cast<PetscReal>(1.0)
    };
};

struct ErrorRecord {
    PetscInt nx = 0;
    PetscReal h = 0.0;
    PetscReal dt = 0.0;
    PetscInt nTimes = 0;
    PetscReal err1 = 0.0;
    PetscReal err2 = 0.0;
    PetscReal errInf = 0.0;
};

enum class ThermoCriterion {
    BGC_LocalFluxAgainstPhiGradient,
    TBGC_MixedFluxAgainstMuGradient,
};

struct ThermoScan {
    ThermoCriterion criterion =
        ThermoCriterion::BGC_LocalFluxAgainstPhiGradient;
    PetscBool violated = PETSC_FALSE;
    PetscBool hasReferenceValue = PETSC_FALSE;
    PetscInt positiveCount = 0;
    PetscReal minIndicator = 0.0;
    PetscReal maxIndicator = 0.0;
    PetscReal referenceValue = 0.0;
};

inline std::string trim(const std::string& text) {
    const auto begin = std::find_if_not(text.begin(), text.end(),
                                        [](unsigned char ch) {
                                            return std::isspace(ch) != 0;
                                        });
    if (begin == text.end()) {
        return "";
    }

    const auto end = std::find_if_not(text.rbegin(), text.rend(),
                                      [](unsigned char ch) {
                                          return std::isspace(ch) != 0;
                                      }).base();
    return std::string(begin, end);
}

inline std::string toLower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char ch) {
                       return static_cast<char>(std::tolower(ch));
                   });
    return text;
}

inline std::vector<std::string> splitByComma(const std::string& text) {
    std::vector<std::string> parts;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        parts.push_back(trim(item));
    }
    return parts;
}

inline PetscReal parsePetscReal(const std::string& text) {
    const std::string cleaned = trim(text);
    const char* begin = cleaned.c_str();
    char* end = nullptr;
    const PetscReal value = static_cast<PetscReal>(std::strtod(begin, &end));
    if (end == begin) {
        throw std::runtime_error("Invalid PetscReal value: " + cleaned);
    }
    return value;
}

inline PetscInt parsePetscInt(const std::string& text) {
    const std::string cleaned = trim(text);
    const char* begin = cleaned.c_str();
    char* end = nullptr;
    const long value = std::strtol(begin, &end, 10);
    if (end == begin) {
        throw std::runtime_error("Invalid PetscInt value: " + cleaned);
    }
    return static_cast<PetscInt>(value);
}

inline std::vector<PetscInt> parsePetscIntList(const std::string& text) {
    const auto parts = splitByComma(text);
    std::vector<PetscInt> values;
    values.reserve(parts.size());
    for (const auto& part : parts) {
        if (!part.empty()) {
            values.push_back(parsePetscInt(part));
        }
    }
    if (values.empty()) {
        throw std::runtime_error("Empty PetscInt list.");
    }
    return values;
}

inline std::vector<PetscReal> parsePetscRealList(const std::string& text) {
    const auto parts = splitByComma(text);
    std::vector<PetscReal> values;
    values.reserve(parts.size());
    for (const auto& part : parts) {
        if (!part.empty()) {
            values.push_back(parsePetscReal(part));
        }
    }
    if (values.empty()) {
        throw std::runtime_error("Empty PetscReal list.");
    }
    return values;
}

inline std::unordered_map<std::string, std::string>
readKeyValueFile(const std::filesystem::path& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open input file: " +
                                 filePath.string());
    }

    std::unordered_map<std::string, std::string> data;
    std::string line;
    PetscInt lineNumber = 0;
    while (std::getline(file, line)) {
        ++lineNumber;

        const std::size_t commentPosHash = line.find('#');
        const std::size_t commentPosSlash = line.find("//");
        std::size_t commentPos = std::string::npos;
        if (commentPosHash != std::string::npos) {
            commentPos = commentPosHash;
        }
        if (commentPosSlash != std::string::npos &&
            (commentPos == std::string::npos || commentPosSlash < commentPos)) {
            commentPos = commentPosSlash;
        }
        if (commentPos != std::string::npos) {
            line = line.substr(0, commentPos);
        }

        line = trim(line);
        if (line.empty()) {
            continue;
        }

        const std::size_t eqPos = line.find('=');
        if (eqPos == std::string::npos) {
            throw std::runtime_error(
                "Invalid line in file " + filePath.string() +
                " at line " + std::to_string(lineNumber) +
                ". Expected 'key = value'.");
        }

        const std::string key = toLower(trim(line.substr(0, eqPos)));
        const std::string value = trim(line.substr(eqPos + 1));
        if (key.empty()) {
            throw std::runtime_error("Empty key in file " + filePath.string() +
                                     " at line " +
                                     std::to_string(lineNumber) + ".");
        }
        data[key] = value;
    }
    return data;
}

inline std::filesystem::path findDataDirectory() {
    const std::filesystem::path dataDir =
        std::filesystem::current_path() / "Dados";
    if (std::filesystem::exists(dataDir) &&
        std::filesystem::is_directory(dataDir)) {
        return std::filesystem::canonical(dataDir);
    }
    throw std::runtime_error("Could not locate the Dados directory at: " +
                             dataDir.string());
}

inline InputData readInputData(const std::filesystem::path& dataDir) {
    InputData input;
    const auto simData = readKeyValueFile(dataDir / "simulation.dat");
    if (simData.contains("tf")) {
        input.tf = parsePetscReal(simData.at("tf"));
    }
    if (simData.contains("cdt")) {
        input.cdt = parsePetscReal(simData.at("cdt"));
    }
    if (simData.contains("nx_list")) {
        input.nxList = parsePetscIntList(simData.at("nx_list"));
    }
    if (simData.contains("bv_list")) {
        input.bvList = parsePetscRealList(simData.at("bv_list"));
    }

    if (simData.contains("verbose")) {
        input.verbose = parsePetscBool(simData.at("verbose"));
    }

    return input;
}

inline void configureStandardSimulation(const InputData& input,
                                        bgc::SimConfig& cfg,
                                        const bgc::Model model,
                                        const PetscReal bv,
                                        const PetscInt nx) {
    cfg.model = model;
    cfg.nx = nx;
    cfg.lx = 1.0;
    cfg.x0 = 0.0;
    cfg.h = cfg.lx / static_cast<PetscReal>(cfg.nx);
    cfg.bv = bv;
    cfg.tf = input.tf;
    cfg.dt = input.cdt * cfg.h * cfg.h;
    cfg.nTimes = static_cast<PetscInt>(cfg.tf / cfg.dt + 0.5);
    if (cfg.nTimes < 1) {
        cfg.nTimes = 1;
    }
    cfg.dt = cfg.tf / static_cast<PetscReal>(cfg.nTimes);
    cfg.useDirectSolver = true;
    cfg.verbose = input.verbose;
    cfg.directSolverBackend = bgc::DirectSolverBackend::PetscDefault;
}

template <typename FieldFn>
inline PetscErrorCode fillVectorFromField(const bgc::SimConfig& cfg,
                                          Vec v,
                                          const PetscReal t,
                                          FieldFn&& fieldFn) {
    PetscFunctionBeginUser;
    PetscInt iStart = 0;
    PetscInt iEnd = 0;
    PetscCall(VecGetOwnershipRange(v, &iStart, &iEnd));
    for (PetscInt i = iStart; i < iEnd; ++i) {
        const PetscReal x = cfg.xCenter(i);
        const PetscScalar value = static_cast<PetscScalar>(fieldFn(cfg, x, t));
        PetscCall(VecSetValue(v, i, value, INSERT_VALUES));
    }
    PetscCall(VecAssemblyBegin(v));
    PetscCall(VecAssemblyEnd(v));
    PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename PhiFn, typename SourceFn>
inline PetscErrorCode runBGCTransientLoop(const bgc::SimConfig& cfg,
                                          bgc::SimState& st,
                                          PhiFn&& phiFn,
                                          SourceFn&& sourceFn) {
    PetscFunctionBeginUser;
    PetscCall(bgc::computeAnalyticPhi(cfg, st, 0.0, phiFn));
    PetscCall(VecCopy(st.phi_a, st.phi_0));
    PetscCall(VecCopy(st.phi_a, st.phi));

    for (PetscInt step = 1; step <= cfg.nTimes; ++step) {
        const PetscReal timeNp1 = static_cast<PetscReal>(step) * cfg.dt;
        const auto rhs = bgc::computeRHSBGC(cfg, timeNp1);
        PetscCall(bgc::assembleRHSBGC(cfg, st, rhs));
        PetscCall(bgc::computeSourceTerm(cfg, st, timeNp1, sourceFn));
        PetscCall(VecAXPY(st.b, 1.0, st.b1Source));
        PetscCall(bgc::solveLinearSystem(cfg, st));
        if (step < cfg.nTimes) {
            PetscCall(VecCopy(st.phi, st.phi_0));
        }
    }

    PetscPrintf(PETSC_COMM_WORLD,
                "[time-march] model=%s  nx=%" PetscInt_FMT
                "  Bv=%.16e  tf=%.16e  dt=%.16e  nTimes=%" PetscInt_FMT "\n",
                cfg.modelName().c_str(),
                cfg.nx,
                cfg.bv,
                cfg.tf,
                cfg.dt,
                cfg.nTimes);
    PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename PhiFn, typename MuFn, typename SourceFn>
inline PetscErrorCode runTBGCSTransientLoop(const bgc::SimConfig& cfg,
                                            bgc::SimState& st,
                                            PhiFn&& phiFn,
                                            MuFn&& muFn,
                                            SourceFn&& sourceFn) {
    PetscFunctionBeginUser;
    PetscCall(bgc::computeAnalyticPhi(cfg, st, 0.0, phiFn));
    PetscCall(fillVectorFromField(cfg, st.mu, 0.0, muFn));
    PetscCall(VecCopy(st.phi_a, st.phi_0));
    PetscCall(VecCopy(st.phi_a, st.phi));
    PetscCall(VecCopy(st.mu,    st.mu_0));  // linha em falta

    for (PetscInt step = 1; step <= cfg.nTimes; ++step) {
        const PetscReal timeNp1 = static_cast<PetscReal>(step) * cfg.dt;
        const auto rhs = bgc::computeRHSTBGCS(cfg, timeNp1);
        PetscCall(bgc::assembleRHSTBGCS(cfg, st, rhs));
        PetscCall(bgc::computeSourceTerm(cfg, st, timeNp1, sourceFn));
        PetscCall(VecAXPY(st.b1, 1.0, st.b1Source));
        PetscCall(bgc::solveLinearSystem(cfg, st));
        if (step < cfg.nTimes) {
            PetscCall(VecCopy(st.phi, st.phi_0));
            PetscCall(VecCopy(st.mu,  st.mu_0));  // linha em falta
        }
    }

    PetscPrintf(PETSC_COMM_WORLD,
                "[time-march] model=%s  nx=%" PetscInt_FMT
                "  Bv=%.16e  tf=%.16e  dt=%.16e  nTimes=%" PetscInt_FMT "\n",
                cfg.modelName().c_str(),
                cfg.nx,
                cfg.bv,
                cfg.tf,
                cfg.dt,
                cfg.nTimes);
    PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename PhiFn, typename MuFn, typename SourceFn>
inline PetscErrorCode runTBGCTransientLoop(const bgc::SimConfig& cfg,
                                           bgc::SimState& st,
                                           PhiFn&& phiFn,
                                           MuFn&& muFn,
                                           SourceFn&& sourceFn) {
    PetscFunctionBeginUser;
    PetscCall(bgc::computeAnalyticPhi(cfg, st, 0.0, phiFn));
    PetscCall(fillVectorFromField(cfg, st.mu, 0.0, muFn));
    PetscCall(VecCopy(st.phi_a, st.phi_0));
    PetscCall(VecCopy(st.phi_a, st.phi));

    for (PetscInt step = 1; step <= cfg.nTimes; ++step) {
        const PetscReal timeNp1 = static_cast<PetscReal>(step) * cfg.dt;
        const auto rhs = bgc::computeRHSTBGC(cfg, timeNp1);
        PetscCall(bgc::assembleRHSTBGC(cfg, st, rhs));
        PetscCall(bgc::computeSourceTerm(cfg, st, timeNp1, sourceFn));
        PetscCall(VecAXPY(st.b1, 1.0, st.b1Source));
        PetscCall(bgc::solveLinearSystem(cfg, st));
        if (step < cfg.nTimes) {
            PetscCall(VecCopy(st.phi, st.phi_0));
        }
    }

    PetscPrintf(PETSC_COMM_WORLD,
                "[time-march] model=%s  nx=%" PetscInt_FMT
                "  Bv=%.16e  tf=%.16e  dt=%.16e  nTimes=%" PetscInt_FMT "\n",
                cfg.modelName().c_str(),
                cfg.nx,
                cfg.bv,
                cfg.tf,
                cfg.dt,
                cfg.nTimes);
    PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------------------------------------------
//  runTSPVTransientLoop
//
//  Marcha temporal para o modelo TSPV (sistema 2x2: Psi e V).
//
//  Parametros:
//    cfg          -- configuracao base (nx, dt, nTimes, parametros fisicos).
//    st           -- estado PETSc alocado por createStateTSPV.
//    psiFn        -- solucao exacta (ou inicial) de Psi: f(cfg, x, t).
//    vFn          -- solucao exacta (ou inicial) de V:   f(cfg, x, t).
//    sourcePsiFn  -- termo fonte da equacao de Psi:      f(cfg, x, t).
//    sourceVFn    -- termo fonte da equacao de V:         f(cfg, x, t).
//    stepConfigFn -- devolve o SimConfig actualizado para o passo t^{n+1}.
//                    Usado para actualizar BCs de Robin dependentes do tempo.
//                    Para BCs fixas, passe: [](const bgc::SimConfig& c,
//                                             PetscReal) { return c; }
//
//  Apos o retorno, st.phi e st.mu contem a solucao em t = tf.
// ---------------------------------------------------------------------------
template <typename StepConfigFn>
inline PetscErrorCode runTSPVTransientLoop(
        const bgc::SimConfig& cfg,
        bgc::SimState&        st,
        const bgc::FieldFn&   psiFn,
        const bgc::FieldFn&   vFn,
        const bgc::FieldFn&   sourcePsiFn,
        const bgc::FieldFn&   sourceVFn,
        StepConfigFn&&        stepConfigFn)
{
    PetscFunctionBeginUser;

//     // --- Condicao inicial: Psi^0 e V^0 ---
//     PetscCall(fillVectorFromField(cfg, st.phi, 0.0, psiFn));
//     PetscCall(fillVectorFromField(cfg, st.mu,  0.0, vFn));
//     PetscCall(VecCopy(st.phi, st.phi_0));
//     PetscCall(VecCopy(st.mu,  st.mu_0));

//     // --- Marcha temporal ---
//     for (PetscInt step = 1; step <= cfg.nTimes; ++step) {
//         const PetscReal tauNp1          = static_cast<PetscReal>(step) * cfg.dt;
//         const bgc::SimConfig stepCfg    = stepConfigFn(cfg, tauNp1);
//         const auto rhs                  = bgc::computeRHS(stepCfg, tauNp1);

//         PetscCall(bgc::assembleFullRHSTSPV(stepCfg,
//                                            st,
//                                            rhs,
//                                            tauNp1,
//                                            sourcePsiFn,
//                                            sourceVFn));
//         PetscCall(bgc::solveLinearSystem(stepCfg, st));

//         if (step < cfg.nTimes) {
//             PetscCall(VecCopy(st.phi, st.phi_0));
//             PetscCall(VecCopy(st.mu,  st.mu_0));
//         }
//     }

//     PetscPrintf(PETSC_COMM_WORLD,
//                 "[time-march] model=%s  nx=%" PetscInt_FMT
//                 "  tf=%.16e  dt=%.16e  nTimes=%" PetscInt_FMT "\n",
//                 cfg.modelName().c_str(),
//                 cfg.nx,
//                 cfg.tf,
//                 cfg.dt,
//                 cfg.nTimes);

    PetscFunctionReturn(!PETSC_SUCCESS);
}

template <typename PhiXFn, typename PhiXXXFn>
inline ThermoScan analyseBGCConstitutiveCriterion(const bgc::SimConfig& cfg,
                                                  PhiXFn&& phiXFn,
                                                  PhiXXXFn&& phiXXXFn,
                                                  PetscInt sampleCount = 4097,
                                                  PetscReal tol = 1.0e-14) {
    ThermoScan scan;
    scan.criterion = ThermoCriterion::BGC_LocalFluxAgainstPhiGradient;
    scan.minIndicator = std::numeric_limits<PetscReal>::max();
    scan.maxIndicator = -std::numeric_limits<PetscReal>::max();

    const PetscInt effectiveSamples = std::max<PetscInt>(sampleCount, 64);
    for (PetscInt i = 0; i < effectiveSamples; ++i) {
        const PetscReal x = cfg.x0 + cfg.lx * static_cast<PetscReal>(i) /
                                         static_cast<PetscReal>(effectiveSamples - 1);
        const PetscReal d1 = phiXFn(x);
        const PetscReal d3 = phiXXXFn(x);
        if ((d1 * d3 > tol) && (std::abs(d3) > tol)) {
            const PetscReal localCrit = std::abs(d1) / std::abs(d3);
            if (!scan.hasReferenceValue || localCrit > scan.referenceValue) {
                scan.referenceValue = localCrit;
                scan.hasReferenceValue = PETSC_TRUE;
            }
        }
    }

    for (PetscInt i = 0; i < cfg.nx; ++i) {
        const PetscReal x = cfg.xCenter(i);
        const PetscReal d1 = phiXFn(x);
        const PetscReal d3 = phiXXXFn(x);
        const PetscReal indicator = (-d1 + cfg.bv * d3) * d1;
        scan.minIndicator = std::min(scan.minIndicator, indicator);
        scan.maxIndicator = std::max(scan.maxIndicator, indicator);
        if (indicator > tol) {
            ++scan.positiveCount;
        }
    }

    if (cfg.nx == 0) {
        scan.minIndicator = 0.0;
        scan.maxIndicator = 0.0;
    }
    scan.violated = (scan.positiveCount > 0) ? PETSC_TRUE : PETSC_FALSE;
    return scan;
}

template <typename PhiXFn, typename PhiXXXFn>
inline ThermoScan analyseTBGCMixedCriterion(const bgc::SimConfig& cfg,
                                            PhiXFn&& phiXFn,
                                            PhiXXXFn&& phiXXXFn,
                                            PetscReal tol = 1.0e-14) {
    ThermoScan scan;
    scan.criterion = ThermoCriterion::TBGC_MixedFluxAgainstMuGradient;
    scan.minIndicator = std::numeric_limits<PetscReal>::max();
    scan.maxIndicator = -std::numeric_limits<PetscReal>::max();

    for (PetscInt i = 0; i < cfg.nx; ++i) {
        const PetscReal x = cfg.xCenter(i);
        const PetscReal d1 = phiXFn(x);
        const PetscReal d3 = phiXXXFn(x);
        const PetscReal muX = d1 - cfg.bv * d3;
        const PetscReal indicator = -(muX * muX);
        scan.minIndicator = std::min(scan.minIndicator, indicator);
        scan.maxIndicator = std::max(scan.maxIndicator, indicator);
        if (indicator > tol) {
            ++scan.positiveCount;
        }
    }

    if (cfg.nx == 0) {
        scan.minIndicator = 0.0;
        scan.maxIndicator = 0.0;
    }
    scan.violated = (scan.positiveCount > 0) ? PETSC_TRUE : PETSC_FALSE;
    return scan;
}

inline void printThermoSummary(const char* caseName,
                               const bgc::SimConfig& cfg,
                               const ThermoScan& scan) {
    if (scan.criterion ==
        ThermoCriterion::BGC_LocalFluxAgainstPhiGradient) {
        if (scan.violated) {
            PetscPrintf(PETSC_COMM_WORLD,
                        "[criterion-check] case=%s  model=%s  nx=%" PetscInt_FMT
                        "  Bv=%.16e",
                        caseName,
                        cfg.modelName().c_str(),
                        cfg.nx,
                        cfg.bv);
            if (scan.hasReferenceValue) {
                PetscPrintf(PETSC_COMM_WORLD,
                            "  sampled_Bvcrit=%.16e",
                            scan.referenceValue);
            } else {
                PetscPrintf(PETSC_COMM_WORLD,
                            "  sampled_Bvcrit=not-applicable");
            }
            PetscPrintf(PETSC_COMM_WORLD,
                        "  positive_points=%" PetscInt_FMT
                        "  min(J*phi_x)=%.16e  max(J*phi_x)=%.16e"
                        "  *** WARNING: local BGC constitutive inversion detected ***\n",
                        scan.positiveCount,
                        scan.minIndicator,
                        scan.maxIndicator);
        } else {
            PetscPrintf(PETSC_COMM_WORLD,
                        "[criterion-check] case=%s  model=%s  nx=%" PetscInt_FMT
                        "  Bv=%.16e",
                        caseName,
                        cfg.modelName().c_str(),
                        cfg.nx,
                        cfg.bv);
            if (scan.hasReferenceValue) {
                PetscPrintf(PETSC_COMM_WORLD,
                            "  sampled_Bvcrit=%.16e",
                            scan.referenceValue);
            } else {
                PetscPrintf(PETSC_COMM_WORLD,
                            "  sampled_Bvcrit=not-applicable");
            }
            PetscPrintf(PETSC_COMM_WORLD,
                        "  positive_points=%" PetscInt_FMT
                        "  min(J*phi_x)=%.16e  max(J*phi_x)=%.16e  ok\n",
                        scan.positiveCount,
                        scan.minIndicator,
                        scan.maxIndicator);
        }
        return;
    }

    if (scan.violated) {
        PetscPrintf(PETSC_COMM_WORLD,
                    "[criterion-check] case=%s  model=%s  nx=%" PetscInt_FMT
                    "  Bv=%.16e  positive_points=%" PetscInt_FMT
                    "  min(J*mu_x)=%.16e  max(J*mu_x)=%.16e"
                    "  *** WARNING: mixed TBGC constitutive check failed ***\n",
                    caseName,
                    cfg.modelName().c_str(),
                    cfg.nx,
                    cfg.bv,
                    scan.positiveCount,
                    scan.minIndicator,
                    scan.maxIndicator);
    } else {
        PetscPrintf(PETSC_COMM_WORLD,
                    "[criterion-check] case=%s  model=%s  nx=%" PetscInt_FMT
                    "  Bv=%.16e  positive_points=%" PetscInt_FMT
                    "  min(J*mu_x)=%.16e  max(J*mu_x)=%.16e  ok\n",
                    caseName,
                    cfg.modelName().c_str(),
                    cfg.nx,
                    cfg.bv,
                    scan.positiveCount,
                    scan.minIndicator,
                    scan.maxIndicator);
    }
}

inline void printSimulationSummary(const bgc::SimConfig& cfg,
                                   const bgc::ErrorNorms& norms) {
    PetscPrintf(PETSC_COMM_WORLD,
                "nx=%" PetscInt_FMT
                "  h=%.16e"
                "  dt=%.16e"
                "  nTimes=%" PetscInt_FMT
                "  ||e||_1=%.16e"
                "  ||e||_2=%.16e"
                "  ||e||_inf=%.16e\n",
                cfg.nx,
                cfg.h,
                cfg.dt,
                cfg.nTimes,
                norms.L1,
                norms.L2,
                norms.Linf);
}

inline void appendRecord(const bgc::SimConfig& cfg,
                         const bgc::ErrorNorms& norms,
                         std::vector<ErrorRecord>& records) {
    records.push_back(ErrorRecord{cfg.nx,
                                  cfg.h,
                                  cfg.dt,
                                  cfg.nTimes,
                                  norms.L1,
                                  norms.L2,
                                  norms.Linf});
}

inline void printConvergenceTable(const std::string& modelLabel,
                                  const PetscReal bv,
                                  const std::vector<ErrorRecord>& records) {
    std::ostringstream oss;
    oss << "\n======================================================================\n";
    oss << "  Convergence summary for " << modelLabel << " and Bv = "
        << std::scientific << std::setprecision(6) << bv << "\n";
    oss << "======================================================================\n";
    oss << std::setw(8)  << "nx"
        << std::setw(18) << "h"
        << std::setw(18) << "dt"
        << std::setw(10) << "nTimes"
        << std::setw(18) << "||e||_1"
        << std::setw(18) << "||e||_2"
        << std::setw(18) << "||e||_inf"
        << std::setw(14) << "p_1"
        << std::setw(14) << "p_2"
        << std::setw(14) << "p_inf"
        << "\n";

    for (std::size_t i = 0; i < records.size(); ++i) {
        const auto& rec = records[i];
        PetscReal p1 = 0.0;
        PetscReal p2 = 0.0;
        PetscReal pInf = 0.0;
        if (i > 0) {
            const auto& prev = records[i - 1];
            p1 = std::log(prev.err1 / rec.err1) / std::log(prev.h / rec.h);
            p2 = std::log(prev.err2 / rec.err2) / std::log(prev.h / rec.h);
            pInf = std::log(prev.errInf / rec.errInf) / std::log(prev.h / rec.h);
        }

        oss << std::setw(8) << rec.nx
            << std::setw(18) << std::scientific << std::setprecision(6)
            << rec.h
            << std::setw(18) << rec.dt
            << std::setw(10) << rec.nTimes
            << std::setw(18) << rec.err1
            << std::setw(18) << rec.err2
            << std::setw(18) << rec.errInf;

        if (i == 0) {
            oss << std::setw(14) << "-"
                << std::setw(14) << "-"
                << std::setw(14) << "-";
        } else {
            oss << std::setw(14) << std::fixed << std::setprecision(6)
                << p1
                << std::setw(14) << p2
                << std::setw(14) << pInf;
        }
        oss << "\n";
    }
    PetscPrintf(PETSC_COMM_WORLD, "%s", oss.str().c_str());
}

}  // namespace paper_mms