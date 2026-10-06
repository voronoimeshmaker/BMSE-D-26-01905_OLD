# MMS2TSOM

MMS2TSOM verifies the TSOM finite-volume operator with the manufactured fields

```text
U(x,t)   = exp(-t) * x^2 * (1 - x)^2
V(x,t)   = exp(-t) * x^2 * (1 - x)^2
Psi(x,t) = U + V
```

The profile satisfies `g(0)=g(1)=g'(0)=g'(1)=0`, so the test exercises
homogeneous no-flux conditions for `Psi` while remaining compatible with the
current TSOM `V=0` boundary mode. The program writes convergence, field, LTE,
and setup files to the directory configured by `Dados/simulation.dat`.
