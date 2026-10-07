"""Limit B_v -> 0 with two conditions per wall versus the heat equation with Dirichlet data (Section 2.2)."""
import numpy as np, scipy.sparse as sps, scipy.sparse.linalg as spla
from fvsolver import *
N=2048; h=1/N; xi=(np.arange(N)+0.5)*h; t=0.05; dt=1e-5; n=int(t/dt)
p0=np.sin(np.pi*xi)
ex=np.exp(-np.pi**2*t)*np.sin(np.pi*xi)
for Bv in [1e-2,1e-3,1e-4,1e-5]:
    M=((h/dt)*sps.identity(N)+bgc_matrix(N,Bv)).tocsc(); lu=spla.splu(M); p=p0.copy()
    for k in range(n): p=lu.solve((h/dt)*p)
    d=np.abs(p-ex); core=(xi>0.2)&(xi<0.8)
    # layer thickness: where |dphi| peaks near boundary
    print(f"Bv={Bv:g} sqrt(Bv)={np.sqrt(Bv):.4f} max|phi-phi_heat| all={d.max():.2e} interior(0.2-0.8)={d[core].max():.2e} "
          f"argmax distance from wall={min(xi[np.argmax(d)],1-xi[np.argmax(d)]):.4f}")
