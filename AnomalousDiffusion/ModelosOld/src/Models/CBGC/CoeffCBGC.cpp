#include <bgclib/Models/BGC/CoeffBGC.hpp>

#include <cmath>
#include <limits>

namespace bgc {

StencilCoefficients computeCoefficientsCBGC(const SimConfig& cfg) {
    const PetscInt n = cfg.nx;
    const PetscReal h = cfg.h;
    const PetscReal h2 = h * h;
    const PetscReal h3 = h2 * h;
    const PetscReal bv = cfg.bv;

    StencilCoefficients sc;

// ---------------------------------------------------------------------------------------------------------------
//      Equação para o primeiro ponto interno (vol0)
// ---------------------------------------------------------------------------------------------------------------

    {
        const PetscReal deno = 1.0 / (225.0 * h3);
        const PetscReal ap =    225.0 * h2 + 10800.0 * bv;
        const PetscReal ae =  -  (250.0 * h2 + 2400.0 * bv) ;
        const PetscReal aee =    9.0 * h2 + 432.0 * bv;

        sc.vol0.ncols = 3;
        sc.vol0.col[0] = 0; sc.vol0.coef[0] = ap * deno;
        sc.vol0.col[1] = 1; sc.vol0.coef[1] = ae * deno;
        sc.vol0.col[2] = 2; sc.vol0.coef[2] = aee * deno;
    }

// ---------------------------------------------------------------------------------------------------------------
//      Equação para o segundo ponto interno (vol1)
// ---------------------------------------------------------------------------------------------------------------

    {
        const PetscReal deno =  1.0 / (3600. * h3);
        const PetscReal aw =   - ( 3750.0 * h2 +  3600.0 * bv);
        const PetscReal ap =     ( 8050.0 * h2 + 20400.0 * bv);
        const PetscReal ae =   - ( 4194.0 * h2 + 14256.0 * bv);
        const PetscReal aee =    (  150.0 * h2 +  3600.0 * bv);

        sc.vol1.ncols = 4;
        sc.vol1.col[0] = 0; sc.vol1.coef[0] = aw * deno;
        sc.vol1.col[1] = 1; sc.vol1.coef[1] = ap * deno;
        sc.vol1.col[2] = 2; sc.vol1.coef[2] = ae * deno;
        sc.vol1.col[3] = 3; sc.vol1.coef[3] = aee * deno;
    }

// ---------------------------------------------------------------------------------------------------------------
//      Equação para os volumes internos     
// ---------------------------------------------------------------------------------------------------------------

    {
        const PetscReal aww = (1.0 / (24.0 * h) + bv / h3) ;
        const PetscReal aw = (-7.0 / (6.0 * h) - 4.0 * bv / h3);
        const PetscReal ap = (9.0 / (4.0 * h) + 6.0 * bv / h3) ;

        sc.interior.ncols = 5;
        sc.interior.col[0] = -2; sc.interior.coef[0] = aww;
        sc.interior.col[1] = -1; sc.interior.coef[1] = aw;
        sc.interior.col[2] = 0;  sc.interior.coef[2] = ap;
        sc.interior.col[3] = 1;  sc.interior.coef[3] = aw;
        sc.interior.col[4] = 2;  sc.interior.coef[4] = aww;
    }


// ---------------------------------------------------------------------------------------------------------------
//      Equação para o penultimo volume (volNm1)
// ---------------------------------------------------------------------------------------------------------------

    {
        const PetscReal deno =  1.0 / (3600.0 * h3);
        const PetscReal ae =   - ( 3750.0 * h2 +  3600.0 * bv);
        const PetscReal ap =     ( 8050.0 * h2 + 20400.0 * bv);
        const PetscReal aw =   - ( 4194.0 * h2 + 14256.0 * bv);
        const PetscReal aww =    (  150.0 * h2 +  3600.0 * bv);

        sc.volNm2.ncols = 4;
        sc.volNm2.col[0] = n - 4;  sc.volNm2.coef[0] = aww * deno;
        sc.volNm2.col[1] = n - 3;  sc.volNm2.coef[1] = aw * deno;
        sc.volNm2.col[2] = n - 2;  sc.volNm2.coef[2] = ap * deno;
        sc.volNm2.col[3] = n - 1;  sc.volNm2.coef[3] = ae * deno;

    }




// ---------------------------------------------------------------------------------------------------------------
//      Equação para o primeiro ponto interno (vol0)
// ---------------------------------------------------------------------------------------------------------------

    {

        const PetscReal deno = 1.0 / (225.0 * h3);
        const PetscReal ap =    225.0 * h2 + 10800.0 * bv;
        const PetscReal aw =  -  (250.0 * h2 + 2400.0 * bv) ;
        const PetscReal aww =    9.0 * h2 + 432.0 * bv;
        sc.volNm1.ncols = 3;    
        sc.volNm1.col[0] = n - 3;       sc.volNm1.coef[0] = aww * deno;
        sc.volNm1.col[1] = n - 2;       sc.volNm1.coef[1] = aw * deno;
        sc.volNm1.col[2] = n - 1;       sc.volNm1.coef[2] = ap * deno;    
    }
    


    return sc;
}

RHSCoefficients computeRHSCBGC(const SimConfig& cfg,
                              PetscReal        t) {
    const PetscReal h = cfg.h;
    const PetscReal h2 = h * h;
    const PetscReal h3 = h2 * h;
    const PetscReal bv = cfg.bv;
    const PetscReal nan = std::numeric_limits<PetscReal>::quiet_NaN();
    const PetscReal eps = 1.0e-30;

    const PetscReal a1w = cfg.bcWest[0].alpha();
    const PetscReal b1w = cfg.bcWest[0].beta();
    const PetscReal a2w = cfg.bcWest[1].alpha();
    const PetscReal b2w = cfg.bcWest[1].beta();
    const PetscReal g1w = cfg.bcWest[0].gamma(t);
    const PetscReal g2w = cfg.bcWest[1].gamma(t);

    const PetscReal a1e = cfg.bcEast[0].alpha();
    const PetscReal b1e = cfg.bcEast[0].beta();
    const PetscReal a2e = cfg.bcEast[1].alpha();
    const PetscReal b2e = cfg.bcEast[1].beta();
    const PetscReal g1e = cfg.bcEast[0].gamma(t);
    const PetscReal g2e = cfg.bcEast[1].gamma(t);

    RHSCoefficients rhs{};

    const PetscReal detW =  b2w * a1w - a2w * b1w;
    const PetscReal detE =  b2e * a1e - a2e * b1e;

    if (std::abs(detW) <= eps) {
        rhs.vol0 = nan;
        rhs.vol1 = nan;
        return rhs;
    }

    if (std::abs(detE) <= eps) {
        rhs.volNm2 = nan;
        rhs.volNm1 = nan;
        return rhs;
    }
// ====================================================================================================================
//      Termo fonte para o primeiro ponto interno (vol0)
// ====================================================================================================================
    {
        const PetscReal DENO = 225.0 * detW * h3;
        const PetscReal sp1 =    - (2880 * h * bv + 240.0 * h3) * a2w + (8832 * bv - 16 * h2) * b2w;
        const PetscReal sp2 =      (2880 * h * bv - 240.0 * h3) * a1w - (8832 * bv - 16 * h2) * b1w;
        rhs.vol0 = (sp1 * g1w + sp2 * g2w) / DENO;
    }

// ====================================================================================================================
//      Termo fonte para o segundo ponto interno (vol1)
// ====================================================================================================================

{
    const PetscReal DENO = 225.0 * detW * h3 / (h2 + 24.0 * bv);
    const PetscReal sp1 = 16.0 * b2w - 15.0 * h * a2w;
    const PetscReal sp2 = -(16.0 * b1w - 15.0 * h * a1w);
    const PetscReal numer = sp1 * g1w + sp2 * g2w;

    rhs.vol1 = numer / DENO;

    if (cfg.verbose) {  
        PetscPrintf(PETSC_COMM_WORLD,
                "\n[DEBUG vol1] ----------------------------------------------------\n");
        PetscPrintf(PETSC_COMM_WORLD,
                "[DEBUG vol1] a1w=%.16e  b1w=%.16e  g1w=%.16e\n",
                a1w, b1w, g1w);
        PetscPrintf(PETSC_COMM_WORLD,
                "[DEBUG vol1] a2w=%.16e  b2w=%.16e  g2w=%.16e\n",
                a2w, b2w, g2w);
        PetscPrintf(PETSC_COMM_WORLD,
                "[DEBUG vol1] detW=%.16e\n",
                detW);
        PetscPrintf(PETSC_COMM_WORLD,
                "[DEBUG vol1] h=%.16e  h2=%.16e  h3=%.16e  bv=%.16e\n",
                h, h2, h3, bv);
        PetscPrintf(PETSC_COMM_WORLD,
                "[DEBUG vol1] sp1=%.16e  sp2=%.16e\n",
                sp1, sp2);
        PetscPrintf(PETSC_COMM_WORLD,
                "[DEBUG vol1] numer=%.16e  DENO=%.16e\n",
                numer, DENO);
        PetscPrintf(PETSC_COMM_WORLD,
                "[DEBUG vol1] rhs.vol1=%.16e\n",
                rhs.vol1);
        PetscPrintf(PETSC_COMM_WORLD,
                "[DEBUG vol1] ----------------------------------------------------\n");
    }
}
// ====================================================================================================================
//      Termo fonte para o penultimo ponto interno (volNm2)
// ====================================================================================================================

    
    {
        const PetscReal DENO = 225.0 * detE * h3 / (h2 + 24.0 * bv);
        const PetscReal sp1 = 16.0 * b2e + 15.0 * h * a2e;
        const PetscReal sp2 = -(16.0 * b1e + 15.0 * h * a1e);
        const PetscReal numer = sp1 * g1e + sp2 * g2e;

        rhs.volNm2 = numer / DENO;

        if (cfg.verbose) {      

            PetscPrintf(PETSC_COMM_WORLD,
                "\n[DEBUG volNm2] --------------------------------------------------\n");
            PetscPrintf(PETSC_COMM_WORLD,
                "[DEBUG volNm2] a1e=%.16e  b1e=%.16e  g1e=%.16e\n",
                a1e, b1e, g1e);
            PetscPrintf(PETSC_COMM_WORLD,
                "[DEBUG volNm2] a2e=%.16e  b2e=%.16e  g2e=%.16e\n",
                a2e, b2e, g2e);
            PetscPrintf(PETSC_COMM_WORLD,
                "[DEBUG volNm2] detE=%.16e\n",
                detE);
            PetscPrintf(PETSC_COMM_WORLD,
                "[DEBUG volNm2] h=%.16e  h2=%.16e  h3=%.16e  bv=%.16e\n",
                h, h2, h3, bv);
            PetscPrintf(PETSC_COMM_WORLD,
                "[DEBUG volNm2] sp1=%.16e  sp2=%.16e\n",
                sp1, sp2);
            PetscPrintf(PETSC_COMM_WORLD,
                "[DEBUG volNm2] numer=%.16e  DENO=%.16e\n",
                numer, DENO);
            PetscPrintf(PETSC_COMM_WORLD,
                "[DEBUG volNm2] rhs.volNm2=%.16e\n",
                rhs.volNm2);
            PetscPrintf(PETSC_COMM_WORLD,
                "[DEBUG volNm2] --------------------------------------------------\n");
    
        }
    }
    
// ====================================================================================================================
//      Termo fonte para o ultimo volume (volNm1) 
// ====================================================================================================================

    {
        const PetscReal DENO = 225.0 * detE * h3;
        const PetscReal sp1 =      (2880 * h * bv - 240.0 * h3) * a2e + (8832 * bv - 16 * h2) * b2e;
        const PetscReal sp2 =    - (2880 * h * bv - 240.0 * h3) * a1e - (8832 * bv - 16 * h2) * b1e;

        rhs.volNm1 = (sp1 * g1e + sp2 * g2e) / DENO;
        
    }
    

    if (cfg.verbose) {
        PetscPrintf(
            PETSC_COMM_WORLD,
            "[CoeffCBGC::RHS] ------------------------------------------------------------\n");
        PetscPrintf(PETSC_COMM_WORLD,
                    "[CoeffBGC::RHS] t=%.16e  h=%.16e  bv=%.16e\n",
                    t,
                    h,
                    bv);
        PetscPrintf(PETSC_COMM_WORLD,
                    "[CoeffCBGC::RHS] west: a1=%.16e  b1=%.16e  g1=%.16e  "
                    "a2=%.16e  b2=%.16e  g2=%.16e  det=%.16e\n",
                    a1w,
                    b1w,
                    g1w,
                    a2w,
                    b2w,
                    g2w,
                    detW);
        PetscPrintf(PETSC_COMM_WORLD,
                    "[CoeffCBGC::RHS] east: a1=%.16e  b1=%.16e  g1=%.16e  "
                    "a2=%.16e  b2=%.16e  g2=%.16e  det=%.16e\n",
                    a1e,
                    b1e,
                    g1e,
                    a2e,
                    b2e,
                    g2e,
                    detE);
        PetscPrintf(PETSC_COMM_WORLD,
                    "[CoeffCBGC::RHS] vol0   = %.16e\n",
                    rhs.vol0);
        PetscPrintf(PETSC_COMM_WORLD,
                    "[CoeffCBGC::RHS] vol1   = %.16e\n",
                    rhs.vol1);
        PetscPrintf(PETSC_COMM_WORLD,
                    "[CoeffCBGC::RHS] volNm2 = %.16e\n",
                    rhs.volNm2);
        PetscPrintf(PETSC_COMM_WORLD,
                    "[CoeffCBGC::RHS] volNm1 = %.16e\n",
                    rhs.volNm1);
        PetscPrintf(
            PETSC_COMM_WORLD,
            "[CoeffCBGC::RHS] ------------------------------------------------------------\n");
    }

    return rhs;
}

}  // namespace bgc
