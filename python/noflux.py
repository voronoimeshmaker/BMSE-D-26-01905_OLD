"""Closed-system test of Proposition 1 and the fully discrete Remark (Section 6.3)."""
import numpy as np
from fvsolver import *
for Bv in [1e-3,10]:
  for N in [64,512]:
    h=1/N; xi=(np.arange(N)+0.5)*h
    p0=np.exp(-(xi-0.3)**2/(2*0.04**2))/(0.04*np.sqrt(2*np.pi))   # off-centre pulse, not symmetric
    L=neumann_Lh(N); A=sps.identity(N)-Bv*L
    rec=[]
    def mon(t,p): return (Fh(p,Bv,h), h*p.sum(), p.copy())
    p,out,dt=run('tbgc_noflux',N,Bv,p0,5e-3,monitor=mon)
    F=np.array([o[0] for o in out]); M=np.array([o[1] for o in out])
    # energy identity residual for each step
    res=[]
    for k in range(len(out)-1):
        a=out[k+1][2]; b=out[k][2]; mu=A@a
        lhs=F[k+1]-F[k]+0.5*h*(a-b)@(A@(a-b)); rhs=-dt*h*np.sum((np.diff(mu)/h)**2)
        res.append(abs(lhs-rhs)/max(abs(rhs),1e-300))
    print(f"Bv={Bv:g} N={N}: steps={len(F)-1} monotone={np.all(np.diff(F)<=0)} max rel mass change={np.abs(M-M[0]).max()/M[0]:.1e} "
          f"min phi={p.min():.2e} max rel residual of energy identity={max(res):.1e} F_end/F0={F[-1]/F[0]:.3e}")
