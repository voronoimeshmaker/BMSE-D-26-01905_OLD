"""Gaussian-pulse benchmark (Section 9): Table 10 ratios and the width measures of Table 11 (FWHM, int|phi|, second moment of |phi|)."""
import numpy as np, sys
from fvsolver import *
N=512; h=1/N; xi=(np.arange(N)+0.5)*h; s0=0.04; m0=0.5; tf=5e-3
phi0=np.exp(-(xi-m0)**2/(2*s0**2))/(s0*np.sqrt(2*np.pi))
def metrics(p):
    M=h*p.sum(); s2=h*np.sum((xi-m0)**2*p)/M
    A=np.abs(p); s2abs=h*np.sum((xi-m0)**2*A)/(h*A.sum())
    pmax=p.max(); half=pmax/2; idx=np.where(p>=half)[0]
    # FWHM with linear interpolation
    i0,i1=idx[0],idx[-1]
    xl=xi[i0-1]+(half-p[i0-1])*(h)/(p[i0]-p[i0-1]); xr=xi[i1]+(half-p[i1])*h/(p[i1+1]-p[i1])
    pos=p>0; neg=~pos
    # width of central positive region
    c=np.argmax(p); l=c
    while l>0 and p[l-1]>0: l-=1
    r=c
    while r<N-1 and p[r+1]>0: r+=1
    return dict(M=M,s2=s2,s2abs=s2abs,L1=h*A.sum(),pmax=pmax,pmin=p.min(),fwhm=xr-xl,
                Apos=h*p[pos].sum(),Aneg=h*p[neg].sum(),wpos=(r-l+1)*h)
M0=h*phi0.sum()
print('initial mass',M0)
for Bv in [1e-4,5e-4,1e-3]:
    for sch in ['bgc','tbgc']:
        F0=Fh(phi0,Bv,h)
        p,_,dt=run(sch,N,Bv,phi0,tf)
        m=metrics(p); sF=s0**2+2*tf
        print(f"Bv={Bv:g} {sch:5} s2/sF={m['s2']/sF:.6f} F/F0={Fh(p,Bv,h)/F0:.4e} pmin={m['pmin']:.3e} pmax={m['pmax']:.4f} "
              f"dM/M0={(m['M']-M0)/M0:.3e} Apos={m['Apos']:.6f} Aneg={m['Aneg']:.3e} s2abs/sF={m['s2abs']/sF:.5f} "
              f"L1={m['L1']:.5f} FWHM={m['fwhm']:.5f} wpos={m['wpos']:.4f}")
        np.save(f'gauss_{sch}_{Bv:g}.npy',p)
# Fickian whole-line reference FWHM
print('Fick whole-line FWHM', 2*np.sqrt(2*np.log(2))*np.sqrt(s0**2+2*tf), 'pmax', 1/np.sqrt(2*np.pi*(s0**2+2*tf)))
