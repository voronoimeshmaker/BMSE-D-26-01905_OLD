#!/usr/bin/env python3
"""
Symbolic verification of the finite-volume coefficients of manuscript
BMSE-D-26-01905 (BGC direct scheme, Appendix A; mixed TBGC scheme, Appendix B).

Closure hypothesis, reconstructed from the tables and confirmed here: near the
western boundary, phi is represented by the quartic Taylor polynomial about xi=0
whose value phi_w and slope phi'_w are fixed by the two boundary conditions and
whose remaining coefficients are fitted to the first three cell values
(xi = h/2, 3h/2, 5h/2). Interior faces use the standard 4th-order (first
derivative) and 2nd-order (third derivative) four-point formulas.

Formulas are checked as printed in the revised manuscript.

Run:  python3 verify_closures_sympy.py      (requires sympy)
"""
import sympy as sp

h, Bv = sp.symbols('h B_v', positive=True)
a, b, c, d, e, x = sp.symbols('phi_w dphi_w c d e x')
P1, P2, P3, P4 = sp.symbols('phi_1 phi_2 phi_3 phi_4')       # first four cells
mu1, mu2 = sp.symbols('mu_1 mu_2')
a1, e1, g1, a2, e2, g2 = sp.symbols('alpha_w1 eta_w1 gamma_w1 alpha_w2 eta_w2 gamma_w2')

results = []
def check(name, got, expected):
    ok = sp.simplify(sp.expand(got - expected)) == 0
    results.append((name, ok))
    print(f"[{'OK ' if ok else 'FAIL'}] {name}")
    if not ok:
        print("       derived :", sp.simplify(got))
        print("       printed :", sp.simplify(expected))

# --- quartic boundary reconstruction ------------------------------------
poly = a + b*x + c*x**2/2 + d*x**3/6 + e*x**4/24
fit = sp.solve([sp.Eq(P1, poly.subs(x, h/2)),
                sp.Eq(P2, poly.subs(x, 3*h/2)),
                sp.Eq(P3, poly.subs(x, 5*h/2))], [c, d, e], dict=True)[0]
R = lambda expr: sp.expand(expr.subs(fit))
D = lambda k, xx: R(sp.diff(poly, x, k).subs(x, xx))

# boundary value and slope from the two BCs (Eq. phiw_left)
Dw = e2*a1 - a2*e1
bc = {a: (e2*g1 - e1*g2)/Dw, b: (a1*g2 - a2*g1)/Dw}

# --- 1. Interior third-derivative formulas (Eqs. d3w/d3e_interior) ------
# face w: [3(phi_W - phi_P) + (phi_E - phi_WW)]/h^3
# face e: [3(phi_P - phi_E) + (phi_EE - phi_W)]/h^3   (exact for phi = xi^3)
WW_, W_, P_, E_, EE_ = [s**3 for s in (-2*h, -h, sp.Integer(0), h, 2*h)]
check("Eq. d3w_interior exact for xi^3", (3*(W_ - P_) + (E_ - WW_))/h**3, 6)
check("Eq. d3e_interior exact for xi^3", (3*(P_ - E_) + (EE_ - W_))/h**3, 6)

# --- 2. West-face third derivative (Eq. d3fw_left) ------------------------
cw = lambda al, et: -128*et + 45*al*h
printed = (-48*P1 + 8*P2 - sp.Rational(24, 25)*P3)/h**3 \
          - sp.Rational(8, 25)*(cw(a2, e2)*g1 - cw(a1, e1)*g2)/(Dw*h**3)
check("Eq. d3fw_left (incl. boundary term)", D(3, 0).subs(bc), printed)

# --- 3. BGC first volume (Table A1/A2) -------------------------------------
# spatial operator  -(phi'_e - phi'_w) + Bv (phi'''_e - phi'''_w)
L1 = sp.expand(-(D(1, h) - b) + Bv*(D(3, h) - D(3, 0)))
check("A1 first vol A_P",  L1.coeff(P1), 1/h + 48*Bv/h**3)
check("A1 first vol A_E",  L1.coeff(P2), -sp.Rational(10, 9)/h - sp.Rational(32, 3)*Bv/h**3)
check("A1 first vol A_EE", L1.coeff(P3), sp.Rational(1, 25)/h + sp.Rational(48, 25)*Bv/h**3)
rhs1 = sp.expand(-(L1.coeff(a)*a + L1.coeff(b)*b).subs(bc))
R1p = (sp.Rational(16, 15)*h**2 - sp.Rational(64, 5)*Bv)*a2*h - (sp.Rational(16, 225)*h**2 - sp.Rational(2944, 75)*Bv)*e2
R2p = -(sp.Rational(16, 15)*h**2 - sp.Rational(64, 5)*Bv)*a1*h + (sp.Rational(16, 225)*h**2 - sp.Rational(2944, 75)*Bv)*e1
check("A2 first vol R_bc", rhs1, (R1p*g1 + R2p*g2)/(Dw*h**3))

# --- 4. BGC second volume (west face = reconstruction at h, east face interior)
d1e = (P1 - P4)/(24*h) + sp.Rational(9, 8)*(P3 - P2)/h
d3e = (P4 - 3*P3 + 3*P2 - P1)/h**3            # correct-sign interior formula
L2 = sp.expand(-(d1e - D(1, h)) + Bv*(d3e - D(3, h)))
check("A1 second vol A_W",  L2.coeff(P1), -sp.Rational(25, 24)/h - Bv/h**3)
check("A1 second vol A_P",  L2.coeff(P2), sp.Rational(161, 72)/h + sp.Rational(17, 3)*Bv/h**3)
check("A1 second vol A_E",  L2.coeff(P3), -sp.Rational(233, 200)/h - sp.Rational(99, 25)*Bv/h**3)
check("A1 second vol A_EE", L2.coeff(P4), sp.Rational(1, 24)/h + Bv/h**3)
rhs2 = sp.expand(-(L2.coeff(a)*a + L2.coeff(b)*b).subs(bc))
R1s = -sp.Rational(1, 225)*(-16*e2 + 15*a2*h)*(h**2 + 24*Bv)
R2s = sp.Rational(1, 225)*(-16*e1 + 15*a1*h)*(h**2 + 24*Bv)
check("A2 second vol R_bc", rhs2, (R1s*g1 + R2s*g2)/(Dw*h**3))

# --- 5. TBGC first volume (Tables B1-B5) ----------------------------------
# balance: mu'(0) - mu'(h), mu quadratic through (0,mu_w),(h/2,mu_1),(3h/2,mu_2),
# with mu_w = phi_w - Bv phi''_w from the quartic reconstruction
mu_w = a - Bv*D(2, 0)
L0 = lambda s: (s - h/2)*(s - 3*h/2)/((h/2)*(3*h/2))
Lb = lambda s: s*(s - 3*h/2)/((h/2)*(-h))
Lc = lambda s: s*(s - h/2)/((3*h/2)*h)
q = mu_w*L0(x) + mu1*Lb(x) + mu2*Lc(x)
bal = sp.expand(sp.diff(q, x).subs(x, 0) - sp.diff(q, x).subs(x, h))
check("B1 A_phiphi first vol P (spatial part)", bal.coeff(P1), 40*Bv/h**3)
check("B1 A_phiphi first vol E",  bal.coeff(P2), -sp.Rational(80, 27)*Bv/h**3)
check("B1 A_phiphi first vol EE", bal.coeff(P3), sp.Rational(8, 25)*Bv/h**3)
check("B2 A_phimu first vol P", bal.coeff(mu1), 4/h)
check("B2 A_phimu first vol E", bal.coeff(mu2), -sp.Rational(4, 3)/h)
bphi = sp.expand(-(bal.coeff(a)*a + bal.coeff(b)*b).subs(bc))
K = 25216*Bv/h**3 + 1800/h
check("B4 b_phi first vol", bphi,
      ((K*e2 - 11040*Bv/h**2*a2)*g1 - (K*e1 - 11040*Bv/h**2*a1)*g2)/(675*Dw))
cons = sp.expand(mu1 - (P1 - Bv*D(2, h/2)))
check("B3 A_muphi first vol P",  cons.coeff(P1), -(1 + 3*Bv/h**2))
check("B3 A_muphi first vol E",  cons.coeff(P2), sp.Rational(14, 9)*Bv/h**2)
check("B3 A_muphi first vol EE", cons.coeff(P3), -sp.Rational(3, 25)*Bv/h**2)
bmu = sp.expand(-(cons.coeff(a)*a + cons.coeff(b)*b).subs(bc))
check("B5 b_mu first vol", bmu,
      (-(352*Bv/h**2*e2 + 120*Bv/h*a2)*g1 + (352*Bv/h**2*e1 + 120*Bv/h*a1)*g2)/(225*Dw))

# --- 6. Manufactured sources ----------------------------------------------
xi, tau = sp.symbols('xi tau')
ph = sp.exp(-tau)*xi**2*(1 - xi)**2
s1 = sp.diff(ph, tau) - sp.diff(ph, xi, 2) + Bv*sp.diff(ph, xi, 4)
check("MMS-1 source (Eq. test1_source_bgc)", s1,
      sp.exp(-tau)*(24*Bv - xi**4 + 2*xi**3 - 13*xi**2 + 12*xi - 2))
check("MMS-1 phi_x*phi_xxx (Eq. test1_phi_x_phi_xxx)",
      sp.diff(ph, xi)*sp.diff(ph, xi, 3), 24*sp.exp(-2*tau)*xi*(xi - 1)*(2*xi - 1)**2)
ph2 = sp.exp(-tau)*sp.cosh(2*(xi - sp.Rational(1, 3)))
s2 = sp.diff(ph2, tau) - sp.diff(ph2, xi, 2) + Bv*sp.diff(ph2, xi, 4)
check("MMS-2 source (Eq. test2_source_bgc)", s2, (16*Bv - 5)*ph2)
check("MMS-2 mu (Eq. test2_mu_exact)", ph2 - Bv*sp.diff(ph2, xi, 2), (1 - 4*Bv)*ph2)

n_ok = sum(ok for _, ok in results)
print(f"\n{n_ok}/{len(results)} checks passed")
raise SystemExit(0 if n_ok == len(results) else 1)
