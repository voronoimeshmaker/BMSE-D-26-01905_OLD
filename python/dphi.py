"""Local dissipation diagnostic D_phi for the example of Section 3.  Usage: python3 dphi.py 0.01"""
import numpy as np
from fvsolver import *
import sys; SIG=float(sys.argv[1]); N=512; h=1/N; xi=(np.arange(N)+0.5)*h; Bv=1.0
p0=xi**2*(1-xi)**2*np.exp(-(xi-0.5)**2/(2*SIG**2))
p,_,dt=run('bgc',N,Bv,p0,2e-6)
# face quantities (interior faces with 4-point stencils)
i=np.arange(1,N-2)  # face between i and i+1
d1=(p[i-1]-p[i+2])/(24*h)+9/8*(p[i+1]-p[i])/h
d3=(p[i+2]-3*p[i+1]+3*p[i]-p[i-1])/h**3
J=-d1+Bv*d3; D=J*d1; xf=(i+1)*h
neg=xi[p<0]
print("pmax",p.max()); print('phi<0 region(s):',neg.min() if neg.size else None, neg.max() if neg.size else None, 'n cells',neg.size,'pmin',p.min())
pos=xf[D>0]
# group contiguous
grp=np.split(pos,np.where(np.diff(pos)>1.5*h)[0]+1)
print('D_phi>0 intervals:',[(round(g[0],3),round(g[-1],3)) for g in grp if g.size])
gn=np.split(neg,np.where(np.diff(neg)>1.5*h)[0]+1)
print('phi<0 intervals:',[(round(g[0],3),round(g[-1],3)) for g in gn if g.size])
