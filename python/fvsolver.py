"""Re-implementation of the BGC (Appendix A) and TBGC (Appendix B) schemes from the
printed coefficient tables, homogeneous Dirichlet-Neumann BCs (R_bc = 0), backward Euler.
Also: closed-system TBGC with symmetric Neumann L_h (Proposition 1)."""
import numpy as np, scipy.sparse as sps, scipy.sparse.linalg as spla

def bgc_matrix(N, Bv):
    h = 1.0/N
    A = sps.lil_matrix((N, N))
    for i in range(N):
        for off, cf in zip((-2, -1, 0, 1, 2), (1/(24*h)+Bv/h**3, -7/(6*h)-4*Bv/h**3, 9/(4*h)+6*Bv/h**3,
                                                -7/(6*h)-4*Bv/h**3, 1/(24*h)+Bv/h**3)):
            j = i+off
            if 0 <= j < N: A[i, j] = cf
    def setrow(i, d):
        A[i, :] = 0
        for j, v in d.items(): A[i, j] = v
    f = {0: 1/h+48*Bv/h**3, 1: -10/(9*h)-32/3*Bv/h**3, 2: 1/(25*h)+48/25*Bv/h**3}
    s = {0: -25/(24*h)-Bv/h**3, 1: 161/(72*h)+17/3*Bv/h**3, 2: -233/(200*h)-99/25*Bv/h**3, 3: 1/(24*h)+Bv/h**3}
    setrow(0, f); setrow(1, s)
    setrow(N-1, {N-1-k: v for k, v in f.items()}); setrow(N-2, {N-1-k: v for k, v in s.items()})
    return A.tocsc()

def tbgc_blocks(N, Bv):
    h = 1.0/N
    App = sps.lil_matrix((N, N)); Apm = sps.lil_matrix((N, N)); Amp = sps.lil_matrix((N, N))
    for i in range(N):
        Apm[i, i] = 2/h
        if i > 0: Apm[i, i-1] = -1/h; Amp[i, i-1] = Bv/h**2
        if i < N-1: Apm[i, i+1] = -1/h; Amp[i, i+1] = Bv/h**2
        Amp[i, i] = -(1+2*Bv/h**2)
    for (i, s) in ((0, 1), (N-1, -1)):
        for r in (App, Apm, Amp): r[i, :] = 0
        App[i, i] = 40*Bv/h**3; App[i, i+s] = -80/27*Bv/h**3; App[i, i+2*s] = 8/25*Bv/h**3
        Apm[i, i] = 4/h; Apm[i, i+s] = -4/(3*h)
        Amp[i, i] = -(1+3*Bv/h**2); Amp[i, i+s] = 14/9*Bv/h**2; Amp[i, i+2*s] = -3/25*Bv/h**2
    return App.tocsc(), Apm.tocsc(), Amp.tocsc()

def neumann_Lh(N):
    h = 1.0/N
    main = -2*np.ones(N); main[0] = main[-1] = -1
    return sps.diags([np.ones(N-1), main, np.ones(N-1)], [-1, 0, 1]).tocsc()/h**2

def Fh(phi, Bv, h):
    return 0.5*(h*np.sum(phi**2) + Bv*h*np.sum((np.diff(phi)/h)**2))

def run(scheme, N, Bv, phi0, tf, C=0.1024, monitor=None, every=1):
    h = 1.0/N; dt = C*h*h; nsteps = int(round(tf/dt)); dt = tf/nsteps
    I = sps.identity(N, format='csc')
    if scheme == 'bgc':
        M = (h/dt)*I + bgc_matrix(N, Bv)
        lu = spla.splu(M.tocsc())
        step = lambda p: lu.solve((h/dt)*p)
    elif scheme == 'tbgc':
        App, Apm, Amp = tbgc_blocks(N, Bv)
        M = sps.bmat([[(h/dt)*I + App, Apm], [Amp, I]]).tocsc()
        lu = spla.splu(M)
        step = lambda p: lu.solve(np.concatenate([(h/dt)*p, np.zeros(N)]))[:N]
    elif scheme == 'tbgc_noflux':      # Proposition 1 closure, phi_xi = mu_xi = 0
        L = neumann_Lh(N); A = I - Bv*L
        M = sps.bmat([[I/dt, -L], [-A, I]]).tocsc()
        lu = spla.splu(M)
        step = lambda p: lu.solve(np.concatenate([p/dt, np.zeros(N)]))[:N]
    phi = phi0.copy(); out = []
    if monitor: out.append(monitor(0.0, phi))
    for n in range(1, nsteps+1):
        phi = step(phi)
        if monitor and (n % every == 0 or n == nsteps): out.append(monitor(n*dt, phi))
    return phi, out, dt
