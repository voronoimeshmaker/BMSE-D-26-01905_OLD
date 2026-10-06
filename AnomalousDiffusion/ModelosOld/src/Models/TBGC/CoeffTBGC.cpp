#include <bgclib/Models/TBGC/CoeffTBGC.hpp>

#include <cmath>

namespace bgc {

// ---------------------------------------------------------------------------
//  computeCoefficientsTBGC
//
//  Implementa a Tabela 3 do artigo para os blocos A11, A12 e A21.
//
//  Convenção de índices em StencilRow — idêntica ao BGC:
//      Volumes de fronteira : col[] contém índices globais absolutos.
//      Volume interior      : col[] contém offsets relativos { -2,-1,0,+1,+2 }.
// ---------------------------------------------------------------------------

TBGCCoefficients computeCoefficientsTBGC(const SimConfig& cfg) {

    const PetscInt  n     = cfg.nx;
    const PetscReal hinv  = 1.0 / cfg.h;
    const PetscReal hinv2 = hinv  * hinv;
    const PetscReal hinv3 = hinv2 * hinv;
    const PetscReal bv    = cfg.bv;
    const PetscReal bvh2  = bv * hinv2;
    const PetscReal bvh3  = bv * hinv3;
    const PetscReal dhdt  = cfg.h / cfg.dt;

    TBGCCoefficients tc;

    // =======================================================================
    //  Bloco A11  —  M/Δτ + contribuição de fronteira
    //
    //  Volumes interiores: apenas diagonal = dhdt.
    //  Volumes de fronteira: diagonal acrescida de termos em bvh3 que
    //  incorporam a condição de contorno de quarta ordem.
    // =======================================================================
    {
        auto& A = tc.A11;

        // -------------------------------------------------------------------
        //  Volume 0
        // -------------------------------------------------------------------
        {
            constexpr PetscReal c3 = -PetscReal(80) / PetscReal(27);
            constexpr PetscReal c4 =  PetscReal(8)  / PetscReal(25);

            A.vol0.ncols  = 3;
            A.vol0.col[0] = 0;  A.vol0.coef[0] = dhdt + 40.0*bvh3;
            A.vol0.col[1] = 1;  A.vol0.coef[1] = c3 * bvh3;
            A.vol0.col[2] = 2;  A.vol0.coef[2] = c4 * bvh3;
        }

        // -------------------------------------------------------------------
        //  Volume 1  —  apenas diagonal (interior do ponto de vista de A11)
        // -------------------------------------------------------------------
        {
            A.vol1.ncols  = 1;
            A.vol1.col[0] = 1;
            A.vol1.coef[0] = dhdt;
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
        //  Volume n-2  —  apenas diagonal
        // -------------------------------------------------------------------
        {
            A.volNm2.ncols  = 1;
            A.volNm2.col[0] = n-2;
            A.volNm2.coef[0] = dhdt;
        }

        // -------------------------------------------------------------------
        //  Volume n-1
        // -------------------------------------------------------------------
        {
            constexpr PetscReal c3 = -PetscReal(80) / PetscReal(27);
            constexpr PetscReal c4 =  PetscReal(8)  / PetscReal(25);

            A.volNm1.ncols  = 3;
            A.volNm1.col[0] = n-3;  A.volNm1.coef[0] = c4 * bvh3;
            A.volNm1.col[1] = n-2;  A.volNm1.coef[1] = c3 * bvh3;
            A.volNm1.col[2] = n-1;  A.volNm1.coef[2] = dhdt + 40.0*bvh3;
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
        //  Volume 1  —  estêncil interior padrão (centrado)
        // -------------------------------------------------------------------
        {
            A.vol1.ncols  = 3;
            A.vol1.col[0] = 0;  A.vol1.coef[0] = -hinv;
            A.vol1.col[1] = 1;  A.vol1.coef[1] =  2.0 * hinv;
            A.vol1.col[2] = 2;  A.vol1.coef[2] = -hinv;
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
        //  Volume n-2  —  estêncil interior padrão (centrado)
        // -------------------------------------------------------------------
        {
            A.volNm2.ncols  = 3;
            A.volNm2.col[0] = n-3;  A.volNm2.coef[0] = -hinv;
            A.volNm2.col[1] = n-2;  A.volNm2.coef[1] =  2.0 * hinv;
            A.volNm2.col[2] = n-1;  A.volNm2.coef[2] = -hinv;
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

        // -------------------------------------------------------------------
        //  Volume 0
        // -------------------------------------------------------------------
        {
            constexpr PetscReal c1 =  PetscReal(14) / PetscReal(9);
            constexpr PetscReal c2 = -PetscReal(3)  / PetscReal(25);

            A.vol0.ncols  = 3;
            A.vol0.col[0] = 0;  A.vol0.coef[0] = -(1.0 + 3.0*bvh2);
            A.vol0.col[1] = 1;  A.vol0.coef[1] = c1 * bvh2;
            A.vol0.col[2] = 2;  A.vol0.coef[2] = c2 * bvh2;
        }

        // -------------------------------------------------------------------
        //  Volume 1
        // -------------------------------------------------------------------
        {
            constexpr PetscReal c1 =  PetscReal(10) / PetscReal(9);
            constexpr PetscReal c2 =  PetscReal(21) / PetscReal(25);

            A.vol1.ncols  = 3;
            A.vol1.col[0] = 0;  A.vol1.coef[0] = -3.0 * bvh2;
            A.vol1.col[1] = 1;  A.vol1.coef[1] = -(1.0 + c1*bvh2);
            A.vol1.col[2] = 2;  A.vol1.coef[2] =  c2  * bvh2;
        }

        // -------------------------------------------------------------------
        //  Volumes interiores  —  estêncil simétrico de cinco pontos
        //  col[] contém offsets relativos { -2, -1, 0, +1, +2 }
        // -------------------------------------------------------------------
        {
            constexpr PetscReal c2 =  PetscReal(5)  / PetscReal(2);
            constexpr PetscReal c3 =  PetscReal(4)  / PetscReal(3);
            constexpr PetscReal c4 = -PetscReal(1)  / PetscReal(12);

            A.interior.ncols  = 5;
            A.interior.col[0] = -2;  A.interior.coef[0] = c4 * bvh2;
            A.interior.col[1] = -1;  A.interior.coef[1] = c3 * bvh2;
            A.interior.col[2] =  0;  A.interior.coef[2] = -(1.0 + c2*bvh2);
            A.interior.col[3] =  1;  A.interior.coef[3] = c3 * bvh2;  // ae = aw
            A.interior.col[4] =  2;  A.interior.coef[4] = c4 * bvh2;  // aee = aww
        }

        // -------------------------------------------------------------------
        //  Volume n-2
        // -------------------------------------------------------------------
        {
            constexpr PetscReal c1 =  PetscReal(10) / PetscReal(9);
            constexpr PetscReal c2 =  PetscReal(21) / PetscReal(25);

            A.volNm2.ncols  = 3;
            A.volNm2.col[0] = n-3;  A.volNm2.coef[0] =  c2  * bvh2;
            A.volNm2.col[1] = n-2;  A.volNm2.coef[1] = -(1.0 + c1*bvh2);
            A.volNm2.col[2] = n-1;  A.volNm2.coef[2] = -3.0 * bvh2;
        }

        // -------------------------------------------------------------------
        //  Volume n-1
        // -------------------------------------------------------------------
        {
            constexpr PetscReal c1 =  PetscReal(14) / PetscReal(9);
            constexpr PetscReal c2 = -PetscReal(3)  / PetscReal(25);

            A.volNm1.ncols  = 3;
            A.volNm1.col[0] = n-3;  A.volNm1.coef[0] = c2 * bvh2;
            A.volNm1.col[1] = n-2;  A.volNm1.coef[1] = c1 * bvh2;
            A.volNm1.col[2] = n-1;  A.volNm1.coef[2] = -(1.0 + 3.0*bvh2);
        }
    }

    return tc;
}

// ---------------------------------------------------------------------------
//  computeRHSTBGC
//
//  Implementa a Tabela 4 do artigo para os quatro volumes especiais.
//  Os volumes interiores têm contribuição nula — não são calculados.
// ---------------------------------------------------------------------------

TBGCRHSCoefficients computeRHSTBGC(const SimConfig& cfg, PetscReal t) {

    // const PetscInt  n     = cfg.nx;
    const PetscReal hinv  = 1.0 / cfg.h;
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

    TBGCRHSCoefficients rhs{};

    // -----------------------------------------------------------------------
    //  Volume 0  —  primeiro volume
    //  Tabela 4, linha "First Volume"
    // -----------------------------------------------------------------------
    if (std::abs(g1w) > 1.0e-30 || std::abs(g2w) > 1.0e-30) {

        const PetscReal auxi = b2w*a1w - b1w*a2w;

        // b1 — equação de phi
        {
            const PetscReal deno = 675.0 * auxi;
            const PetscReal sp1  =
                ((25216.0*hinv3*b2w - 11040.0*hinv2*a2w)*bv
                 + 1800.0*b2w*hinv) * g1w;
            const PetscReal sp2  =
               -((25216.0*hinv3*b1w - 11040.0*hinv2*a1w)*bv
                 + 1800.0*b1w*hinv) * g2w;
            rhs.b1.vol0 = (sp1 + sp2) / deno;
        }

        // b2 — equação de mu
        {
            const PetscReal deno = 225.0 * auxi;
            const PetscReal sp1  = -(352.0*hinv2*b2w + 120.0*hinv*a2w)*bv * g1w;
            const PetscReal sp2  =  (352.0*hinv2*b1w + 120.0*hinv*a1w)*bv * g2w;
            rhs.b2.vol0 = (sp1 + sp2) / deno;
        }
    }

    // -----------------------------------------------------------------------
    //  Volume 1  —  segundo volume
    //  Tabela 4, linha "Second Volume"
    // -----------------------------------------------------------------------
    if (std::abs(g1w) > 1.0e-30 || std::abs(g2w) > 1.0e-30) {

        const PetscReal auxi = b2w*a1w - b1w*a2w;

        // b1 — sem contribuição de fronteira no segundo volume
        rhs.b1.vol1 = 0.0;

        // b2 — equação de mu
        {
            const PetscReal deno = 225.0 * auxi;
            const PetscReal sp1  = -(736.0*hinv2*b2w - 240.0*hinv*a2w)*bv * g1w;
            const PetscReal sp2  =  (736.0*hinv2*b1w - 240.0*hinv*a1w)*bv * g2w;
            rhs.b2.vol1 = (sp1 + sp2) / deno;
        }
    }

    // -----------------------------------------------------------------------
    //  Volume n-2  —  penúltimo volume
    //  Tabela 4, linha "Second-to-last Volume"
    // -----------------------------------------------------------------------
    if (std::abs(g1e) > 1.0e-30 || std::abs(g2e) > 1.0e-30) {

        const PetscReal auxi = b2e*a1e - b1e*a2e;

        // b1 — sem contribuição de fronteira no penúltimo volume
        rhs.b1.volNm2 = 0.0;

        // b2 — equação de mu
        {
            const PetscReal deno = 225.0 * auxi;
            const PetscReal sp1  = -(736.0*hinv2*b2e + 240.0*hinv*a2e)*bv * g1e;
            const PetscReal sp2  =  (736.0*hinv2*b1e + 240.0*hinv*a1e)*bv * g2e;
            rhs.b2.volNm2 = (sp1 + sp2) / deno;
        }
    }

    // -----------------------------------------------------------------------
    //  Volume n-1  —  último volume
    //  Tabela 4, linha "Last Volume"
    // -----------------------------------------------------------------------
    if (std::abs(g1e) > 1.0e-30 || std::abs(g2e) > 1.0e-30) {

        const PetscReal auxi = b2e*a1e - b1e*a2e;

        // b1 — equação de phi
        {
            const PetscReal deno = 675.0 * auxi;
            const PetscReal sp1  =
                ((25216.0*hinv3*b2e + 11040.0*hinv2*a2e)*bv
                 + 1800.0*b2e*hinv) * g1e;
            const PetscReal sp2  =
               -((25216.0*hinv3*b1e + 11040.0*hinv2*a1e)*bv
                 + 1800.0*b1e*hinv) * g2e;
            rhs.b1.volNm1 = (sp1 + sp2) / deno;
        }

        // b2 — equação de mu
        {
            const PetscReal deno = 225.0 * auxi;
            const PetscReal sp1  = -(352.0*hinv2*b2e - 120.0*hinv*a2e)*bv * g1e;
            const PetscReal sp2  =  (352.0*hinv2*b1e - 120.0*hinv*a1e)*bv * g2e;
            rhs.b2.volNm1 = (sp1 + sp2) / deno;
        }
    }

    return rhs;
}

} // namespace bgc