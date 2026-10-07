"""Smallest eigenvalues of the direct and mixed operators, B_v = 10 (Section 8.2)."""
import numpy as np, scipy.sparse.linalg as spla
from fvsolver import *
Bv=10
for N in [64,128,256,512,1024,2048]:
    h=1/N
    KB=bgc_matrix(N,Bv)/h
    App,Apm,Amp=tbgc_blocks(N,Bv); KT=(App-Apm@Amp)/h
    lb=np.sort(spla.eigs(KB,k=2,sigma=0,return_eigenvectors=False).real)
    lt=np.sort(spla.eigs(KT.tocsc(),k=2,sigma=0,return_eigenvectors=False).real)
    print(N, lb, lt, 'rel diff lam1', (lb[0]-lt[0])/lt[0])
