<div align="center">

# BGC mixed finite volumes

**Direct and mixed gradient-flow finite-volume solvers for the fourth-order
Bevilacqua–Galeão–Costa (BGC) diffusion equation**

[![Paper](https://img.shields.io/badge/paper-under%20review%20·%20JBSMSE-5C6672?style=flat-square)](#citation)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.XXXXXXX-2F6E72?style=flat-square)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![C++](https://img.shields.io/badge/C%2B%2B-23-2F6E72?style=flat-square&logo=cplusplus&logoColor=white)](#building-the-c-solvers)
[![PETSc](https://img.shields.io/badge/PETSc-%E2%89%A53.19-2F6E72?style=flat-square)](https://petsc.org)
[![Python](https://img.shields.io/badge/python-3.10%2B-B07D5B?style=flat-square&logo=python&logoColor=white)](#python-reference-implementation)
[![License](https://img.shields.io/badge/license-MIT-8A939D?style=flat-square)](LICENSE)

[Overview](#overview) ·
[Repository layout](#repository-layout) ·
[Quick start](#quick-start) ·
[Reproducing the paper](#reproducing-the-paper) ·
[Citation](#citation)

</div>

---

## Overview

This repository contains the code behind the paper

> J. F. Vasconcellos, *A mixed gradient-flow finite-volume formulation for a
> fourth-order non-Fickian diffusion equation: free-energy analysis and
> numerical verification*, Journal of the Brazilian Society of Mechanical
> Sciences and Engineering (manuscript BMSE-D-26-01905, under review).

In dimensionless form the BGC equation reads

$$
\frac{\partial \phi}{\partial \tau}
= \frac{\partial^2 \phi}{\partial \xi^2}
- B_v\,\frac{\partial^4 \phi}{\partial \xi^4},
\qquad \xi \in [0,1],
$$

where $B_v$ is the Bevilacqua number. Two discretisations are provided:

| Scheme | Unknowns | Idea |
|---|---|---|
| **BGC** (direct) | $\phi$ | Fourth-order operator discretised directly, with quartic boundary reconstruction |
| **TBGC** (mixed) | $\phi,\ \mu$ | Two-field gradient-flow form $\mu=\phi-B_v\phi_{\xi\xi}$, $\ \phi_\tau=\mu_{\xi\xi}$ |

With no-flux boundary conditions, the fully implicit mixed scheme satisfies
a discrete free-energy inequality for any time step. Both schemes produce
the same sign-changing solutions, which shows that the negative values come
from the fourth-order model and not from the discretisation.

## Repository layout

```text
.
├── AnomalousDiffusion/
│   ├── CMakeLists.txt          main build (library + programs)
│   ├── bgclib/                 C++/PETSc library: BGC (direct) and TBGC (mixed) models
│   ├── programs/
│   │   ├── MMS/BGC/            MMS1BGC, MMS2BGC, MMS4BGC
│   │   ├── MMS/TBGC/           MMS1TBGC, MMS2TBGC, MMS4TBGC
│   │   └── examples/           minimal BGC example
│   └── ModelosOld/             legacy library and the cases of Sections 3 and 8
│       ├── programs/cases/     case2 (Section 3), case3 (Section 8)
│       └── standalone/         CMake entry point for the legacy build
├── python/                     NumPy/SciPy re-implementation of both schemes
├── verification/               symbolic check of Appendices A–B (SymPy)
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

Each program reads its parameters from `Dados/simulation.dat` and writes
results to `Saida/` (or the `output_dir` set in that file) inside its own
directory.

## Quick start

The Python tools run on any machine with Python 3.10 or later:

```bash
git clone https://github.com/voronoimeshmaker/BMSE-D-26-01905.git
cd BMSE-D-26-01905
python -m pip install -r requirements.txt

python verification/verify_closures_sympy.py   # 26/26 checks passed
cd python && python gauss.py                   # Gaussian-pulse benchmark
```

## Building the C++ solvers

**Requirements:** a C++23 compiler (GCC ≥ 13 or Clang ≥ 17), CMake ≥ 3.25,
MPI, PETSc ≥ 3.19 with MUMPS (found through `pkg-config`), and yaml-cpp.

On Ubuntu 24.04:

```bash
sudo apt install build-essential cmake pkg-config petsc-dev libyaml-cpp-dev
```

For a custom PETSc installation, set `PETSC_DIR` (and `PETSC_ARCH`) before
configuring.

```bash
cd AnomalousDiffusion

# main build: bgclib and the MMS programs
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# legacy build: cases of Sections 3 and 8
cmake -S ModelosOld/standalone -B build-legacy -DCMAKE_BUILD_TYPE=Release
cmake --build build-legacy -j
```

Each program must run from its own directory. The `run_<program>` targets do
this automatically:

```bash
cmake --build build --target run_MMS1BGC
# or, by hand
cd programs/MMS/BGC/MMS1BGC && ../../../../build/bin/MMS1BGC
```

## Reproducing the paper

| Paper | Content | Program | Python |
|---|---|---|---|
| Section 3 | Sign-changing example, $N=512$ | `ModelosOld/programs/cases/case2/Case2_BGC`, `Case2_TBGC` | `dphi.py` |
| Section 7, Tables 1–4 | Manufactured solution, homogeneous BCs | `MMS1BGC`, `MMS1TBGC` | — |
| Section 7, Tables 5–8 | Manufactured solution, non-homogeneous BCs | `MMS2BGC`, `MMS2TBGC` | — |
| Section 8, Table 9 | Free-energy comparison, $B_v=10$, $\phi_0=\sin^2(\pi\xi)$ | `ModelosOld/programs/cases/case3/Case3_BGC`, `Case3_TBGC` | `energy.py`, `eig.py` |
| Section 9, Tables 10–11 | Gaussian pulse, $B_v\in\{10^{-4},5\times10^{-4},10^{-3}\}$ | `MMS4BGC`, `MMS4TBGC` | `gauss.py`, `wholeline.py` |
| Section 6.3 | Discrete energy inequality, closed system | — | `noflux.py` |
| Appendices A–B | Boundary-closure coefficients | — | `verification/verify_closures_sympy.py` |

All runs use backward Euler with $\Delta\tau=\tau_f/n$,
$n=\lceil\tau_f/(0.1024\,h^2)\rceil$, on uniform meshes of $N$ control volumes.
The MMS programs write the $L_1$ error to `*_convergence.csv`; the cell-centre
errors used for the $L_2$ and $L_\infty$ norms are in `*_fields_N*.dat`.
The Section 8 cases run one mesh at a time: set `nx` in `Dados/simulation.dat`
to $16, 32, \dots, 256$.

> [!NOTE]
> The Python scripts rebuild both schemes from the coefficient tables printed
> in the paper and reproduce Tables 9–11 without PETSc. They implement
> homogeneous boundary data only, so the non-homogeneous verification
> (Tables 5–8) requires the C++ solvers.

## Python reference implementation

`python/fvsolver.py` assembles both operators from Appendices A–B:

```python
import numpy as np
from fvsolver import run

N = 256
xi = (np.arange(N) + 0.5) / N
phi0 = np.exp(-(xi - 0.5)**2 / (2 * 0.04**2)) / (0.04 * np.sqrt(2 * np.pi))

# scheme: 'bgc' (direct), 'tbgc' (mixed) or 'tbgc_noflux' (closed system)
phi, history, dt = run('tbgc', N=N, Bv=1e-3, phi0=phi0, tf=5e-3)
```

`verification/verify_closures_sympy.py` rebuilds every boundary coefficient
from the quartic reconstruction described in Sections 5.2 and 6.4. It checks
the result against the printed tables and the manufactured source terms, and
exits with a non-zero code if any check fails.

## Citation

If you use this code, please cite the paper and the archived software:

```bibtex
@article{Vasconcellos2026MixedBGC,
  author  = {Vasconcellos, Jo{\~a}o Fl{\'a}vio},
  title   = {A mixed gradient-flow finite-volume formulation for a fourth-order
             non-{F}ickian diffusion equation: free-energy analysis and
             numerical verification},
  journal = {Journal of the Brazilian Society of Mechanical Sciences and Engineering},
  year    = {2026},
  note    = {Under review, manuscript BMSE-D-26-01905}
}

@software{Vasconcellos2026MixedBGCCode,
  author    = {Vasconcellos, Jo{\~a}o Fl{\'a}vio},
  title     = {{BGC mixed finite volumes}},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.XXXXXXX},
  url       = {https://github.com/voronoimeshmaker/BMSE-D-26-01905}
}
```

GitHub also reads [`CITATION.cff`](CITATION.cff) and shows a **Cite this
repository** button.

## License

Released under the MIT License. See [`LICENSE`](LICENSE).

## Contact

João Flávio Vasconcellos ·
Instituto Politécnico do Rio de Janeiro (IPRJ/UERJ) ·
[jflavio@iprj.uerj.br](mailto:jflavio@iprj.uerj.br)
