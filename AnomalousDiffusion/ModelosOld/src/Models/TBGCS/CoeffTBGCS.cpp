#include <bgclib/Models/TBGCS/CoeffTBGCS.hpp>

#include <cmath>
#include <iostream>

namespace bgc {

// ---------------------------------------------------------------------------
//  computeCoefficientsTBGCS
//
//  Implementa a Tabela 3 do artigo para os blocos A11, A12 e A21.
//
//  Convenção de índices em StencilRow — idêntica ao BGC:
//      Volumes de fronteira : col[] contém índices globais absolutos.
//      Volume interior      : col[] contém offsets relativos { -2,-1,0,+1,+2 }.
// ---------------------------------------------------------------------------

TBGCSCoefficients computeCoefficientsTBGCS(const SimConfig& cfg) {

    const PetscInt  n     = cfg.nx;
    const PetscReal hinv  = 1.0 / cfg.h;
    const PetscReal hinv2 = hinv  * hinv;
    const PetscReal hinv3 = hinv2 * hinv;
    const PetscReal bv    = cfg.bv;
    const PetscReal bvh2  = bv * hinv2;
    const PetscReal bvh3  = bv * hinv3;
    const PetscReal dhdt  = cfg.h / cfg.dt;

    TBGCSCoefficients tc;

    // =======================================================================
    //  Bloco A11  —  M/Δτ + contribuição de fronteira
    //
    //  Volumes interiores: apenas diagonal = dhdt.
    //  Volumes de fronteira: diagonal acrescida de termos em bvh3 que
    //  incorporam a condição de contorno de quarta ordem.
    // =======================================================================
    {
        auto& A = tc.A11;
            const PetscReal c3 = - 80.0 * bvh3 / 27.0; 
            const PetscReal c4 =    8.0 * bvh3 / 25.0;

        // -------------------------------------------------------------------
        //  Volume 0
        // -------------------------------------------------------------------
        {

            A.vol0.ncols  = 3;
            A.vol0.col[0] = 0;  A.vol0.coef[0] = dhdt + 40.0 * bvh3;
            A.vol0.col[1] = 1;  A.vol0.coef[1] = c3;
            A.vol0.col[2] = 2;  A.vol0.coef[2] = c4 ;
        }

        // -------------------------------------------------------------------
        //  Volumes interiores  —  apenas diagonal (offset relativo 0)
        // -------------------------------------------------------------------

        {
            A.interior.ncols  = 1;
            A.interior.col[0] = 0;
            A.interior.coef[0] = dhdt;
        }

        // -------------------------------------------------------------------
        //  Volume n-1
        // -------------------------------------------------------------------
        {
            A.volNm1.ncols  = 3;
            A.volNm1.col[0] = n-3;  A.volNm1.coef[0] = c4;
            A.volNm1.col[1] = n-2;  A.volNm1.coef[1] = c3;
            A.volNm1.col[2] = n-1;  A.volNm1.coef[2] = dhdt + 40.0 * bvh3;
            
        }
    }

    // =======================================================================
    //  Bloco A12  —  Laplaciano discreto de mu  (-∇²)
    //
    //  Estêncil de três pontos escalado por 1/h.
    //  Volumes de fronteira usam aproximações assimétricas de segunda ordem.
    // =======================================================================
    {
        auto& A = tc.A12;

        // -------------------------------------------------------------------
        //  Volume 0
        // -------------------------------------------------------------------
        {
            A.vol0.ncols  = 2;
            A.vol0.col[0] = 0;  A.vol0.coef[0] =  4.0 * hinv;
            A.vol0.col[1] = 1;  A.vol0.coef[1] = -4.0 * hinv / 3.0;
        }

        // -------------------------------------------------------------------
        //  Volumes interiores  —  estêncil centrado de três pontos
        //  col[] contém offsets relativos { -1, 0, +1 }
        // -------------------------------------------------------------------
        {
            A.interior.ncols  = 3;
            A.interior.col[0] = -1;  A.interior.coef[0] = -hinv;
            A.interior.col[1] =  0;  A.interior.coef[1] =  2.0 * hinv;
            A.interior.col[2] =  1;  A.interior.coef[2] = -hinv;
        }


        // -------------------------------------------------------------------
        //  Volume n-1
        // -------------------------------------------------------------------
        {
            A.volNm1.ncols  = 2;
            A.volNm1.col[0] = n-2;  A.volNm1.coef[0] = -4.0 * hinv / 3.0;
            A.volNm1.col[1] = n-1;  A.volNm1.coef[1] =  4.0 * hinv;
        }
    }

    // =======================================================================
    //  Bloco A21  —  relação constitutiva  mu = phi - Bv * ∇²phi
    //
    //  Volumes interiores: estêncil simétrico de cinco pontos de quarta ordem.
    //  Volumes especiais:  estêncis assimétricos de segunda ordem.
    // =======================================================================
    {
        auto& A = tc.A21;
        const PetscReal c1 =   14.0 * bvh2 / 9.;
        const PetscReal c2 = -  3.0 * bvh2 / 25.;

        // -------------------------------------------------------------------
        //  Volume 0
        // -------------------------------------------------------------------
        {

            A.vol0.ncols  = 3;
            A.vol0.col[0] = 0;  A.vol0.coef[0] = - (1.0 + 3.0 * bvh2);
            A.vol0.col[1] = 1;  A.vol0.coef[1] = c1;
            A.vol0.col[2] = 2;  A.vol0.coef[2] = c2;
        }


        // -------------------------------------------------------------------
        //  Volumes interiores  —  estêncil simétrico de cinco pontos
        //  col[] contém offsets relativos { -2, -1, 0, +1, +2 }
        // -------------------------------------------------------------------
        {
            A.interior.ncols     = 3;
            A.interior.col[0]    = -1;  A.interior.coef[0] =  bvh2;
            A.interior.col[1]    =  0;  A.interior.coef[1] = - (1.0 + 2.0  * bvh2);
            A.interior.col[2]    =  1;  A.interior.coef[2] =  bvh2;
        }


        // -------------------------------------------------------------------
        //  Volume n-1
        // -------------------------------------------------------------------
        {
            A.volNm1.ncols  = 3;
            A.volNm1.col[0] = n-3;  A.volNm1.coef[0] = c2;
            A.volNm1.col[1] = n-2;  A.volNm1.coef[1] = c1;
            A.volNm1.col[2] = n-1;  A.volNm1.coef[2] = -(1.0 + 3.0 * bvh2);
        }
    }

    return tc;
}

// ---------------------------------------------------------------------------
//  computeRHSTBGCS
//
//  Implementa a Tabela 4 do artigo para os quatro volumes especiais.
//  Os volumes interiores têm contribuição nula — não são calculados.
// ---------------------------------------------------------------------------

TBGCSRHSCoefficients computeRHSTBGCS(const SimConfig& cfg, PetscReal t) {

    // const PetscInt  n     = cfg.nx;

    const PetscReal h     = cfg.h;
    const PetscReal h2    = h * h;
    const PetscReal hinv  = 1.0 / h;
    const PetscReal hinv2 = hinv  * hinv;
    const PetscReal hinv3 = hinv2 * hinv;
    const PetscReal bv    = cfg.bv;

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

    if (cfg.verbose) {
        std::cout << "a1w = " << a1w << ", b1w = " << b1w << ", g1w = " << g1w << std::endl;
        std::cout << "a2w = " << a2w << ", b2w = " << b2w << ", g2w = " << g2w << std::endl;
        std::cout << "a1e = " << a1e << ", b1e = " << b1e << ", g1e = " << g1e      << std::endl;
        std::cout << "a2e = " << a2e << ", b2e = " << b2e << ", g2e = " << g2e << std::endl;  
    }    


    TBGCSRHSCoefficients rhs{};

    // -----------------------------------------------------------------------
    //  Volume 0  —  primeiro volume
    //  Tabela 4, linha "First Volume"
    // -----------------------------------------------------------------------

    const PetscReal denow = b2w * a1w - b1w * a2w;

    {
        const PetscReal auxiDenow = 8.0 * hinv2 * bv / (225.0 * denow);
        const PetscReal sp1 =  - (15.0 * h * a2w + 44.0 * b2w) * g1w;
        const PetscReal sp2 =    (15.0 * h * a1w + 44.0 * b1w) * g2w;

        rhs.b2.vol0 = (sp1 + sp2) * auxiDenow;

    }

        // b2 — equação de mu
    {
        const PetscReal auxiDenow = 8.0 * hinv3 / (675.0 * denow);
        const PetscReal sp1 =   (225.0 * h2 * b2w - (1380.0 * a2w * h - 3152.0 * b2w) * bv) * g1w;
        const PetscReal sp2 = - (225.0 * h2 * b1w - (1380.0 * a1w * h - 3152.0 * b1w) * bv) * g2w;

        rhs.b1.vol0 = (sp1 + sp2) * auxiDenow;
    
    }


    // -----------------------------------------------------------------------
    //  Volume n-1  —  último volume
    //  Tabela 4, linha "Last Volume"
    // -----------------------------------------------------------------------

    const PetscReal denoe = b2e * a1e - b1e * a2e;
        // b1 — equação de phi
    {
        const PetscReal auxiDenoe = 8.0 * hinv2 * bv / (225.0 * denoe);
        const PetscReal sp1 =    (15.0 * h * a2e - 44.0 * b2e) * g1e;
        const PetscReal sp2 =  - (15.0 * h * a1e - 44.0 * b1e) * g2e;

        rhs.b2.volNm1 = (sp1 + sp2) * auxiDenoe;
    }

        // b2 — equação de mu
    {
        const PetscReal auxiDenoe = 8.0 * hinv3 / (675.0 * denoe);
        const PetscReal sp1 =   (225.0 * h2 * b2e + (1380.0 * a2e * h + 3152.0 * b2e) * bv) * g1e;
        const PetscReal sp2 = - (225.0 * h2 * b1e + (1380.0 * a1e * h + 3152.0 * b1e) * bv) * g2e;

        rhs.b1.volNm1 = (sp1 + sp2) * auxiDenoe;    
    }

    return rhs;

    }

} // namespace bgc