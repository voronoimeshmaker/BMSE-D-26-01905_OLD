"""Free-energy refinement study, B_v = 10 (Section 8.2, Table 9), including N = 512."""
import numpy as np, sys
from fvsolver import *
Bv=10; tf=1e-2
print('Sec 8 benchmark, Bv=10, sin^2 IC')
prev=None
for N in [16,32,64,128,256,512]:
    h=1/N; xi=(np.arange(N)+0.5)*h; p0=np.sin(np.pi*xi)**2
    res={}
    for sch in ['bgc','tbgc']:
        mon=lambda t,p: Fh(p,Bv,h)
        p,F,dt=run(sch,N,Bv,p0,tf,monitor=mon)
        F=np.array(F); res[sch]=(F[-1], np.all(np.diff(F)<=0), F)
    eps=(res['bgc'][0]/res['tbgc'][0]-1)*100
    rate='' if prev is None else f"{np.log(abs(prev)/abs(eps))/np.log(2):.2f}"
    # late-time decay rate from last steps
    Ft=res['tbgc'][2]; lam=-np.log(Ft[-1]/Ft[-2])/(2*dt)
    print(f"N={N:4d} F_BGC={res['bgc'][0]:.6e} F_TBGC={res['tbgc'][0]:.6e} eps%={eps:.4f} p={rate} mono={res['bgc'][1]},{res['tbgc'][1]} lambda_h(late)={lam:.1f}")
    prev=eps
# continuous clamped-clamped eigenvalue of -(d2) + Bv d4 on [0,1]: solve numerically by fine FD on TBGC N=4096? use eigs of BGC matrix
import scipy.sparse.linalg as spla
for N in [256,1024]:
    A=bgc_matrix(N,Bv)*N   # divide by h
    w=spla.eigs(A,k=3,sigma=0,return_eigenvectors=False)
    print('N',N,'smallest eigenvalues of BGC operator',np.sort(w.real))
