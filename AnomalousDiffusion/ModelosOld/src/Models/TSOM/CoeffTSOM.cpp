#include <bgclib/Models/TSOM/CoeffTSOM.hpp>
#include <iostream>

namespace bgc {

TSOMDiffusionCoefficients computeDiffusionCoefficientsTSOM(
    const SimConfig& cfg) noexcept {
    const PetscReal alpha = cfg.alphaTSOM;
    const PetscReal rho   = cfg.rhoTSOM;
    const PetscReal theta = cfg.thetaTSOM;


    TSOMDiffusionCoefficients d;

    d.d11 =(1.0 - alpha * theta) * (1.0 - alpha * (1.0 - theta));
    d.d12 = 0.0 * rho * theta * (1.0 - alpha * (1.0 - theta));
    d.d21 = alpha * (1.0 - theta) * (1.0 - alpha * theta);
    d.d22 = 0.0 * alpha * rho * theta * (1.0 - theta);

    std::cout << "D11 = " << d.d11 << "\n";
    std::cout << "D12 = " << d.d12 << "\n";
    std::cout << "D21 = " << d.d21 << "\n";    
    std::cout << "D22 = " << d.d22 << "\n";

    return d;
}

TSOMCoefficients computeCoefficientsTSOM(const SimConfig& cfg) {
    const PetscInt  n       = cfg.nx;
    const PetscReal h       = cfg.h;
    const PetscReal dt      = cfg.dt;
    const PetscReal lambdaC = cfg.lambdaC;
    const PetscReal lambdaR = cfg.lambdaR;

    const PetscReal hinv = 1.0 / h;
    const PetscReal dhdt = h / dt;

    const auto d = computeDiffusionCoefficientsTSOM(cfg);

    const PetscReal  alphaw = cfg.alphaPsiWest;
    const PetscReal  betaw  = cfg.betaPsiWest;

    const PetscReal  alphae = cfg.alphaPsiEast;
    const PetscReal  betae  = cfg.betaPsiEast;  

    if (alphaw == 0.0 && betaw == 0.0) {
        throw std::runtime_error(
            "computeCoefficientsTSOM: Problemas com as condições de contorno em west.");
    }

    if (alphae == 0.0 && betae == 0.0) {
        throw std::runtime_error(
            "computeCoefficientsTSOM: Problemas com as condições de contorno em east.");
    }

    const PetscReal denow = 1.0 / (3.0 * alphaw * h - 8.0 * betaw);
    const PetscReal denoe = 1.0 / (3.0 * alphae * h + 8.0 * betae); 

    TSOMCoefficients tc{};

    // =========================================================================
    // A11 -- U equation, U column
    // PDE term: -lambdaC * U + d11 * U_xx
    // =========================================================================
    {
        auto& A = tc.A11;

        A.first.ncols   = 2;
        A.first.col[0]  = 0;
        A.first.coef[0] = dhdt + lambdaC * h + 4.0 * d.d11 * (3.0 * alphaw * h - 2.0 * betaw) * denow * hinv;
        A.first.col[1]  = 1;
        A.first.coef[1] = - d.d11 * (hinv + alphaw * denow);

        A.interior.ncols   = 3;
        A.interior.col[0]  = -1;
        A.interior.coef[0] = - d.d11 * hinv;
        A.interior.col[1]  = 0;
        A.interior.coef[1] = dhdt + lambdaC * h + 2.0 * d.d11 * hinv;
        A.interior.col[2]  = 1;
        A.interior.coef[2] = - d.d11 * hinv;

        A.last.ncols   = 2;
        A.last.col[0]  = n - 2;
        A.last.coef[0] = - d.d11 * (hinv + alphae * denoe);
        A.last.col[1]  = n - 1;
        A.last.coef[1] = dhdt + lambdaC * h + 4.0 * d.d11 * (3.0 * alphae * h + 2 * betae) * denoe * hinv;

    }

    // =========================================================================
    // A12 -- U equation, V column
    // PDE term: +lambdaR * V + d12 * V_xx
    // =========================================================================
    {
        auto& A = tc.A12;

        A.first.ncols   = 2;
        A.first.col[0]  = 0;
        // A.first.coef[0] = - lambdaR * h + 4.0 * d.d12 * hinv  - d.d11 * (3.0 * hinv - 9.0 * alphaw * denow); 
        A.first.coef[0] = - lambdaR * h + d.d11 * (hinv - 9.0 * alphaw * denow); 
        A.first.col[1]  = 1;
        // A.first.coef[1] = - (4.0 / 3.0) * d.d12 * hinv + (hinv / 3.0 - alphaw * denow) * d.d11;
        A.first.coef[1] =  - (hinv + alphaw * denow) * d.d11;

        A.interior.ncols   = 3;
        A.interior.col[0]  = -1;
        A.interior.coef[0] = - d.d12 * hinv;
        A.interior.col[1]  = 0;
        A.interior.coef[1] = - lambdaR * h + 2.0 * d.d12 * hinv;
        A.interior.col[2]  = 1;
        A.interior.coef[2] = -d.d12 * hinv;

        A.last.ncols   = 2;
        A.last.col[0]  = n - 2;
        A.last.coef[0] = - (4.0 / 3.0) * d.d12 * hinv + (hinv / 3.0 - alphae * denoe) * d.d11;
        A.last.col[1]  = n - 1;
        A.last.coef[1] = - lambdaR * h + 4.0 * d.d12 * hinv  - d.d11 * (3.0 * hinv - 9.0 * alphae * denoe); 

    }

    // =========================================================================
    // A21 -- V equation, U column
    // PDE term: +lambdaC * U + d21 * U_xx
    // =========================================================================
    {
        auto& A = tc.A21;

        A.first.ncols   = 2;
        A.first.col[0]  = 0;
        A.first.coef[0] = - lambdaC * h + 4.0 * d.d21 * hinv * (3.0 * alphaw * h - 2.0 * betaw) * denow;
        A.first.col[1]  = 1;
        A.first.coef[1] = - 4.0  * d.d21 * hinv * (alphaw * h - 2.0 * betaw) * denow;

        A.interior.ncols   = 3;
        A.interior.col[0]  = -1;
        A.interior.coef[0] = -d.d21 * hinv;
        A.interior.col[1]  = 0;
        A.interior.coef[1] = -lambdaC * h + 2.0 * d.d21 * hinv;
        A.interior.col[2]  = 1;
        A.interior.coef[2] = -d.d21 * hinv;


        
        A.last.ncols   = 2;
        A.last.col[0]  = n - 2;
        A.last.coef[0] = - 4.0  * d.d21 * hinv * (alphae * h + 2 * betae) * denoe;
        A.last.col[1]  = n - 1;
        A.last.coef[1] = - lambdaC * h + 4.0 * d.d21 * hinv * (3.0 * alphae * h + 2 * betae) * denoe;
    }

    // =========================================================================
    // A22 -- V equation, V column
    // PDE term: -lambdaR * V + d22 * V_xx
    // =========================================================================
    {
        auto& A = tc.A22;

        A.first.ncols   = 2;
        A.first.col[0]  = 0;
        // A.first.coef[0] = dhdt + lambdaR * h + 4.0 * d.d22 * hinv + 24.0 * betaw * d.d21 * denow * hinv;
        A.first.coef[0] = dhdt + lambdaR * h + 4.0 * d.d22 * hinv * (3.0 * alphaw * h - 2.0 * betaw) * denow;
        A.first.col[1]  = 1;
        // A.first.coef[1] = - (4.0 / 3.0) * d.d22 * hinv - (8.0 / 3.0)  * betaw * d.d21 * denow * hinv;
        A.first.coef[1] =  - 4.0  * (alphaw * h - 2.0 * betaw) * d.d21 * denow * hinv;

        A.interior.ncols   = 3;
        A.interior.col[0]  = -1;
        A.interior.coef[0] = - d.d22 * hinv;
        A.interior.col[1]  = 0;
        A.interior.coef[1] = dhdt + lambdaR * h + 2.0 * d.d22 * hinv;
        A.interior.col[2]  = 1;
        A.interior.coef[2] = - d.d22 * hinv;

        A.last.ncols   = 2;
        A.last.col[0]  = n - 2;
        A.last.coef[0] = - (4.0 / 3.0) * d.d22 * hinv + (8.0 / 3.0)  * betae * d.d21 * denoe * hinv;
        A.last.col[1]  = n - 1;
        A.last.coef[1] = dhdt + lambdaR * h + 4.0 * d.d22 * hinv - 24.0 * betae * d.d21 * denoe * hinv;

    }

    return tc;
}

TSOMRHSCoefficients computeRHSTSOM(const SimConfig& cfg, PetscReal /*t*/) {
    const PetscReal h = cfg.h;
    const auto d      = computeDiffusionCoefficientsTSOM(cfg);
    const PetscReal  alphaw = cfg.alphaPsiWest;
    const PetscReal  alphae = cfg.alphaPsiEast;
    const PetscReal  betaw  = cfg.betaPsiWest;
    const PetscReal  betae  = cfg.betaPsiEast;  
    const PetscReal  gammaw = cfg.gammaPsiWest;
    const PetscReal  gammae = cfg.gammaPsiEast;

    const PetscReal denow = 8.0 * gammaw / (3.0 * alphaw * h - 8.0 * betaw);
    const PetscReal denoe = 8.0 * gammae / (3.0 * alphae * h + 8.0 * betae);     

    TSOMRHSCoefficients rhs{};


    rhs.b1.first = d.d11 * denow;
    rhs.b1.last  = d.d11 * denoe;

    rhs.b2.first = d.d21 * denow;
    rhs.b2.last  = d.d21 * denoe;

    return rhs;
}

} // namespace bgc
