"""Whole-line Fourier solution of the BGC equation: signed second moment and mass outside [0,1] (Section 9.4, Eq. 80)."""
import numpy as np
from scipy.integrate import quad
s0=0.04; m0=0.5; tf=5e-3; N=512; h=1/N; xi=(np.arange(N)+0.5)*h
def phi_wl(x,Bv,t=tf):
    f=lambda k: np.cos(k*(x-m0))*np.exp(-s0**2*k**2/2-(k**2+Bv*k**4)*t)/np.pi
    return quad(f,0,np.inf,limit=400)[0]
sF=s0**2+2*tf
for Bv in [1e-4,5e-4,1e-3]:
    xs=np.linspace(-1,2,6001); ph=np.array([phi_wl(x,Bv) for x in xs]); dx=xs[1]-xs[0]
    M=np.trapezoid(ph,xs); s2=np.trapezoid((xs-m0)**2*ph,xs)/M
    inside=(xs>=0)&(xs<=1)
    Mout=M-np.trapezoid(ph[inside],xs[inside])
    pfv=np.load(f'gauss_bgc_{Bv:g}.npy'); pwl=np.array([phi_wl(x,Bv) for x in xi])
    print(f"Bv={Bv:g}: whole-line signed s2/sF={s2/sF:.6f}, mass outside [0,1]={Mout:.2e}, phi_wl(0)={phi_wl(0,Bv):.2e}, "
          f"pmin_wl={ph.min():.3e}, pmax_wl={ph.max():.4f}, max|phi_FV-phi_wl|={np.abs(pfv-pwl).max():.2e}")
# Fickian whole line mass outside
from math import erfc,sqrt
print('Fick mass outside', erfc(0.5/sqrt(2*sF)))
