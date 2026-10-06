#include <petsc.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Params {
    std::string name;
    double alpha {0.0};
    double rho {0.0};
    double lambdaC {0.0};
    double lambdaR {0.0};
    double theta {1.0};
};

struct Mesh {
    int n {0};
    double h {0.0};
    double dt {0.0};
    double tf {0.0};
    int steps {0};
};

struct State {
    std::vector<double> psi;
    std::vector<double> v;
};

struct Matrix2 {
    double a00 {0.0};
    double a01 {0.0};
    double a10 {0.0};
    double a11 {0.0};
};

struct Vec2 {
    double x0 {0.0};
    double x1 {0.0};
};

struct Coeff {
    double d11 {0.0};
    double d12 {0.0};
    double d21 {0.0};
    double d22 {0.0};
};

std::filesystem::path outDir() {
    const auto dir = std::filesystem::path(__FILE__).parent_path() / "Saida";
    std::filesystem::create_directories(dir);
    return dir;
}

std::ofstream csv(const std::filesystem::path& path) {
    std::ofstream f(path);
    if (!f) {
        throw std::runtime_error("Could not open " + path.string());
    }
    f << std::setprecision(16);
    return f;
}

Mesh makeMesh(const int n, const double tf, const double cdt = 0.1024) {
    const double h = 1.0 / static_cast<double>(n);
    const double requested = cdt * h * h;
    const int steps = static_cast<int>(std::ceil(tf / requested));
    return {.n = n, .h = h, .dt = tf / static_cast<double>(steps), .tf = tf, .steps = steps};
}

double xCell(const Mesh& m, const int i) {
    return (static_cast<double>(i) + 0.5) * m.h;
}

Coeff coeffs(const Params& p) {
    const double wpsi2 = 1.0 - (p.alpha + p.rho) * p.theta;
    return {
        .d11 = 1.0 - p.alpha * p.theta,
        .d12 = wpsi2,
        .d21 = p.alpha * (1.0 - p.theta) * (1.0 - p.alpha * p.theta),
        .d22 = -p.alpha * (1.0 - p.theta) * wpsi2,
    };
}

double equilibriumFraction(const Params& p) {
    return p.alpha / (p.alpha + p.rho);
}

double qOne(const Params& p) {
    return 1.0 - p.alpha - p.rho;
}

double sin2Profile(const double x) {
    const double s = std::sin(std::numbers::pi * x);
    return s * s;
}

double poly(const double x) {
    const double y = 1.0 - x;
    return x * x * y * y;
}

double polySecond(const double x) {
    return 2.0 * (1.0 - 6.0 * x + 6.0 * x * x);
}

double gaussian(const double x, const double sigma = 0.04) {
    const double z = (x - 0.5) / sigma;
    return std::exp(-0.5 * z * z) / (sigma * std::sqrt(2.0 * std::numbers::pi));
}

double gaussianDefault(const double x) {
    return gaussian(x);
}

double mass(const Mesh& m, const State& s) {
    double sum = 0.0;
    for (const double v : s.psi) {
        sum += v;
    }
    return m.h * sum;
}

double variance(const Mesh& m, const State& s) {
    const double total = mass(m, s);
    double sum = 0.0;
    for (int i = 0; i < m.n; ++i) {
        const double dx = xCell(m, i) - 0.5;
        sum += dx * dx * s.psi[static_cast<std::size_t>(i)];
    }
    return m.h * sum / total;
}

double energy(const Mesh& m, const Params& p, const State& s) {
    const double eta = p.rho / p.alpha;
    double sum = 0.0;
    for (int i = 0; i < m.n; ++i) {
        const auto idx = static_cast<std::size_t>(i);
        const double u = s.psi[idx] - s.v[idx];
        sum += u * u + eta * s.v[idx] * s.v[idx];
    }
    return 0.5 * m.h * sum;
}

State makeState(const Mesh& m, double (*psiProfile)(double), const double vFraction) {
    State s;
    s.psi.assign(static_cast<std::size_t>(m.n), 0.0);
    s.v.assign(static_cast<std::size_t>(m.n), 0.0);
    for (int i = 0; i < m.n; ++i) {
        const auto idx = static_cast<std::size_t>(i);
        s.psi[idx] = psiProfile(xCell(m, i));
        s.v[idx] = vFraction * s.psi[idx];
    }
    return s;
}

Matrix2 zero2() {
    return {};
}

Matrix2 subtract(const Matrix2& a, const Matrix2& b) {
    return {.a00 = a.a00 - b.a00, .a01 = a.a01 - b.a01, .a10 = a.a10 - b.a10, .a11 = a.a11 - b.a11};
}

Matrix2 multiply(const Matrix2& a, const Matrix2& b) {
    return {
        .a00 = a.a00 * b.a00 + a.a01 * b.a10,
        .a01 = a.a00 * b.a01 + a.a01 * b.a11,
        .a10 = a.a10 * b.a00 + a.a11 * b.a10,
        .a11 = a.a10 * b.a01 + a.a11 * b.a11,
    };
}

Vec2 multiply(const Matrix2& a, const Vec2& x) {
    return {.x0 = a.a00 * x.x0 + a.a01 * x.x1, .x1 = a.a10 * x.x0 + a.a11 * x.x1};
}

Vec2 subtract(const Vec2& a, const Vec2& b) {
    return {.x0 = a.x0 - b.x0, .x1 = a.x1 - b.x1};
}

Matrix2 inverse(const Matrix2& a) {
    const double det = a.a00 * a.a11 - a.a01 * a.a10;
    if (std::abs(det) < 1.0e-300) {
        throw std::runtime_error("Singular 2x2 block in block Thomas solver.");
    }
    return {.a00 = a.a11 / det, .a01 = -a.a01 / det, .a10 = -a.a10 / det, .a11 = a.a00 / det};
}

Matrix2 blockForL(const Params& p,
                  const Coeff& c,
                  const double dt,
                  const double lcoef,
                  const bool diagonal) {
    Matrix2 a {};
    a.a00 = -dt * c.d11 * lcoef;
    a.a01 = -dt * c.d12 * lcoef;
    a.a10 = -dt * c.d21 * lcoef;
    a.a11 = -dt * c.d22 * lcoef;
    if (diagonal) {
        a.a00 += 1.0;
        a.a10 += -dt * p.lambdaC;
        a.a11 += 1.0 + dt * (p.lambdaC + p.lambdaR);
    }
    return a;
}

void solveStep(const Mesh& m,
               const Params& p,
               State& s,
               const std::vector<double>* sourcePsi = nullptr,
               const std::vector<double>* sourceV = nullptr,
               const double sourceFactor = 1.0) {
    const Coeff c = coeffs(p);
    std::vector<Matrix2> lower(static_cast<std::size_t>(m.n), zero2());
    std::vector<Matrix2> diag(static_cast<std::size_t>(m.n), zero2());
    std::vector<Matrix2> upper(static_cast<std::size_t>(m.n), zero2());
    std::vector<Vec2> rhs(static_cast<std::size_t>(m.n));
    const double h2inv = 1.0 / (m.h * m.h);

    for (int i = 0; i < m.n; ++i) {
        const double ldiag = (i == 0 || i == m.n - 1) ? -h2inv : -2.0 * h2inv;
        diag[static_cast<std::size_t>(i)] = blockForL(p, c, m.dt, ldiag, true);
        if (i > 0) {
            lower[static_cast<std::size_t>(i)] = blockForL(p, c, m.dt, h2inv, false);
        }
        if (i + 1 < m.n) {
            upper[static_cast<std::size_t>(i)] = blockForL(p, c, m.dt, h2inv, false);
        }
        const auto idx = static_cast<std::size_t>(i);
        rhs[idx] = {.x0 = s.psi[idx], .x1 = s.v[idx]};
        if (sourcePsi != nullptr && sourceV != nullptr) {
            rhs[idx].x0 += m.dt * sourceFactor * (*sourcePsi)[idx];
            rhs[idx].x1 += m.dt * sourceFactor * (*sourceV)[idx];
        }
    }

    std::vector<Matrix2> cp(static_cast<std::size_t>(m.n), zero2());
    std::vector<Vec2> dp(static_cast<std::size_t>(m.n));
    Matrix2 inv0 = inverse(diag[0]);
    cp[0] = multiply(inv0, upper[0]);
    dp[0] = multiply(inv0, rhs[0]);
    for (int i = 1; i < m.n; ++i) {
        const auto idx = static_cast<std::size_t>(i);
        const Matrix2 denom = subtract(diag[idx], multiply(lower[idx], cp[idx - 1]));
        const Matrix2 invDenom = inverse(denom);
        cp[idx] = (i + 1 < m.n) ? multiply(invDenom, upper[idx]) : zero2();
        dp[idx] = multiply(invDenom, subtract(rhs[idx], multiply(lower[idx], dp[idx - 1])));
    }

    std::vector<Vec2> x(static_cast<std::size_t>(m.n));
    x.back() = dp.back();
    for (int i = m.n - 2; i >= 0; --i) {
        const auto idx = static_cast<std::size_t>(i);
        x[idx] = subtract(dp[idx], multiply(cp[idx], x[idx + 1]));
    }
    for (int i = 0; i < m.n; ++i) {
        const auto idx = static_cast<std::size_t>(i);
        s.psi[idx] = x[idx].x0;
        s.v[idx] = x[idx].x1;
    }
}

std::vector<double> sourcePsiMms(const Mesh& m, const Params& p, const double f) {
    const Coeff c = coeffs(p);
    std::vector<double> source(static_cast<std::size_t>(m.n), 0.0);
    for (int i = 0; i < m.n; ++i) {
        const auto idx = static_cast<std::size_t>(i);
        const double x = xCell(m, i);
        source[idx] = -poly(x) - (c.d11 + c.d12 * f) * polySecond(x);
    }
    return source;
}

std::vector<double> sourceVMms(const Mesh& m, const Params& p, const double f) {
    const Coeff c = coeffs(p);
    std::vector<double> source(static_cast<std::size_t>(m.n), 0.0);
    for (int i = 0; i < m.n; ++i) {
        const auto idx = static_cast<std::size_t>(i);
        const double x = xCell(m, i);
        source[idx] = (-f - p.lambdaC + (p.lambdaC + p.lambdaR) * f) * poly(x) -
                      (c.d21 + c.d22 * f) * polySecond(x);
    }
    return source;
}

State exactMms(const Mesh& m, const double time, const double f) {
    State s;
    s.psi.assign(static_cast<std::size_t>(m.n), 0.0);
    s.v.assign(static_cast<std::size_t>(m.n), 0.0);
    const double e = std::exp(-time);
    for (int i = 0; i < m.n; ++i) {
        const auto idx = static_cast<std::size_t>(i);
        s.psi[idx] = e * poly(xCell(m, i));
        s.v[idx] = f * s.psi[idx];
    }
    return s;
}

std::array<double, 3> norms(const Mesh& m,
                            const std::vector<double>& numerical,
                            const std::vector<double>& exact) {
    double l1 = 0.0;
    double l2 = 0.0;
    double linf = 0.0;
    for (int i = 0; i < m.n; ++i) {
        const auto idx = static_cast<std::size_t>(i);
        const double e = std::abs(numerical[idx] - exact[idx]);
        l1 += e;
        l2 += e * e;
        linf = std::max(linf, e);
    }
    return {m.h * l1, std::sqrt(m.h * l2), linf};
}

double order(const double previous, const double current) {
    return std::log(previous / current) / std::log(2.0);
}

void writeValueOrder(std::ofstream& f, const double previous, const double current) {
    f << "," << current << ",";
    if (previous > 0.0 && current > 0.0) {
        f << order(previous, current);
    } else {
        f << "nan";
    }
}

void runMmsCase(const Params& p, std::ofstream& out) {
    const std::vector<int> ns {8, 16, 32, 64, 128, 256, 512};
    double prevPsi[3] {};
    double prevU[3] {};
    double prevV[3] {};
    for (const int n : ns) {
        const Mesh m = makeMesh(n, 1.0);
        const double f = 0.5;
        State s = exactMms(m, 0.0, f);
        const std::vector<double> sPsi = sourcePsiMms(m, p, f);
        const std::vector<double> sV = sourceVMms(m, p, f);
        for (int step = 1; step <= m.steps; ++step) {
            const double t = static_cast<double>(step) * m.dt;
            solveStep(m, p, s, &sPsi, &sV, std::exp(-t));
        }
        const State ex = exactMms(m, 1.0, f);
        std::vector<double> u(static_cast<std::size_t>(n), 0.0);
        std::vector<double> uex(static_cast<std::size_t>(n), 0.0);
        for (int i = 0; i < n; ++i) {
            const auto idx = static_cast<std::size_t>(i);
            u[idx] = s.psi[idx] - s.v[idx];
            uex[idx] = ex.psi[idx] - ex.v[idx];
        }
        const auto nPsi = norms(m, s.psi, ex.psi);
        const auto nU = norms(m, u, uex);
        const auto nV = norms(m, s.v, ex.v);
        out << p.name << "," << p.alpha << "," << p.rho << "," << p.lambdaC << ","
            << p.lambdaR << "," << p.alpha + p.rho << "," << qOne(p) << ","
            << n << "," << m.h << "," << m.dt << "," << m.steps;
        for (int k = 0; k < 3; ++k) {
            writeValueOrder(out, prevPsi[k], nPsi[static_cast<std::size_t>(k)]);
        }
        for (int k = 0; k < 3; ++k) {
            writeValueOrder(out, prevU[k], nU[static_cast<std::size_t>(k)]);
        }
        for (int k = 0; k < 3; ++k) {
            writeValueOrder(out, prevV[k], nV[static_cast<std::size_t>(k)]);
        }
        out << "\n";
        for (int k = 0; k < 3; ++k) {
            prevPsi[k] = nPsi[static_cast<std::size_t>(k)];
            prevU[k] = nU[static_cast<std::size_t>(k)];
            prevV[k] = nV[static_cast<std::size_t>(k)];
        }
    }
}

void runGroupA(const std::filesystem::path& out) {
    const std::vector<Params> cases {
        {.name = "A1", .alpha = 0.75, .rho = 0.25, .lambdaC = 1.0, .lambdaR = 1.0 / 3.0},
        {.name = "A2", .alpha = 0.30, .rho = 0.40, .lambdaC = 1.0, .lambdaR = 4.0 / 3.0},
    };
    auto mms = csv(out / "GroupA_MMS_convergence.csv");
    mms << "case,alpha,rho,lambda_c,lambda_r,alpha_plus_rho,q1,n,h,dt,steps,"
        << "l1_psi,order_l1_psi,l2_psi,order_l2_psi,linf_psi,order_linf_psi,"
        << "l1_u,order_l1_u,l2_u,order_l2_u,linf_u,order_linf_u,"
        << "l1_v,order_l1_v,l2_v,order_l2_v,linf_v,order_linf_v\n";
    for (const Params& p : cases) {
        runMmsCase(p, mms);
    }

    auto massFile = csv(out / "GroupA_mass_conservation.csv");
    massFile << "case,n,h,dt,steps,mass0,mass_final,final_drift,max_abs_drift\n";
    for (const Params& p : cases) {
        for (const int n : {8, 16, 32, 64, 128, 256, 512}) {
            const Mesh m = makeMesh(n, 5.0);
            State s = makeState(m, sin2Profile, 0.0);
            const double m0 = mass(m, s);
            double maxDrift = 0.0;
            for (int step = 1; step <= m.steps; ++step) {
                solveStep(m, p, s);
                maxDrift = std::max(maxDrift, std::abs(mass(m, s) - m0));
            }
            massFile << p.name << "," << n << "," << m.h << "," << m.dt << ","
                     << m.steps << "," << m0 << "," << mass(m, s) << ","
                     << mass(m, s) - m0 << "," << maxDrift << "\n";
        }
    }
}

void writeStateRows(std::ofstream& f, const std::string& group, const Params& p, const Mesh& m, const State& s, const double time) {
    for (int i = 0; i < m.n; ++i) {
        const auto idx = static_cast<std::size_t>(i);
        f << group << "," << p.name << "," << p.alpha << "," << p.rho << ","
          << p.lambdaC << "," << p.lambdaR << "," << p.theta << "," << time << ","
          << xCell(m, i) << "," << s.psi[idx] << "," << s.v[idx] << ","
          << s.psi[idx] - s.v[idx] << "," << s.v[idx] / s.psi[idx] << "\n";
    }
}

bool isSampleStep(const int step, const Mesh& m, const std::vector<double>& times, std::size_t& next) {
    if (next >= times.size()) {
        return false;
    }
    const double t = static_cast<double>(step) * m.dt;
    if (t + 0.5 * m.dt >= times[next]) {
        ++next;
        return true;
    }
    return false;
}

void runGroupB(const std::filesystem::path& out) {
    const std::vector<Params> cases {
        {.name = "B1", .alpha = 0.20, .rho = 0.20, .lambdaC = 2.0, .lambdaR = 2.0},
        {.name = "B2", .alpha = 0.30, .rho = 0.30, .lambdaC = 2.0, .lambdaR = 2.0},
        {.name = "B3", .alpha = 0.40, .rho = 0.40, .lambdaC = 2.0, .lambdaR = 2.0},
        {.name = "B4", .alpha = 0.50, .rho = 0.50, .lambdaC = 2.0, .lambdaR = 2.0},
    };
    const std::vector<double> times {0.0, 0.1, 0.5, 1.0, 5.0};
    auto profiles = csv(out / "GroupB_profiles.csv");
    profiles << "group,case,alpha,rho,lambda_c,lambda_r,theta,time,x,psi,v,u,ratio\n";
    auto summary = csv(out / "GroupB_summary.csv");
    summary << "case,alpha,rho,alpha_plus_rho,q1,eq_fraction,mass_final,variance_final\n";
    for (const Params& p : cases) {
        const Mesh m = makeMesh(256, 5.0, 0.5);
        State s = makeState(m, sin2Profile, 0.0);
        std::size_t next = 0;
        writeStateRows(profiles, "B", p, m, s, 0.0);
        next = 1;
        for (int step = 1; step <= m.steps; ++step) {
            solveStep(m, p, s);
            if (isSampleStep(step, m, times, next)) {
                writeStateRows(profiles, "B", p, m, s, times[next - 1]);
            }
        }
        summary << p.name << "," << p.alpha << "," << p.rho << ","
                << p.alpha + p.rho << "," << qOne(p) << ","
                << equilibriumFraction(p) << "," << mass(m, s) << ","
                << variance(m, s) << "\n";
    }
}

void runGroupC(const std::filesystem::path& out) {
    const std::vector<Params> cases {
        {.name = "C1", .alpha = 0.10, .rho = 0.50, .lambdaC = 2.0, .lambdaR = 10.0},
        {.name = "C2", .alpha = 0.20, .rho = 0.40, .lambdaC = 2.0, .lambdaR = 4.0},
        {.name = "C3", .alpha = 0.30, .rho = 0.30, .lambdaC = 2.0, .lambdaR = 2.0},
        {.name = "C4", .alpha = 0.40, .rho = 0.20, .lambdaC = 2.0, .lambdaR = 1.0},
        {.name = "C5", .alpha = 0.50, .rho = 0.10, .lambdaC = 2.0, .lambdaR = 0.4},
    };
    auto series = csv(out / "GroupC_center_ratio.csv");
    series << "case,alpha,rho,lambda_c,lambda_r,eq_fraction,q1,time,center_ratio,energy_ratio,mass\n";
    for (const Params& p : cases) {
        const Mesh m = makeMesh(256, 10.0, 0.5);
        State s = makeState(m, sin2Profile, 0.0);
        const double e0 = energy(m, p, s);
        const int stride = std::max(1, m.steps / 1000);
        for (int step = 0; step <= m.steps; ++step) {
            if (step == 0 || step % stride == 0 || step == m.steps) {
                const int center = m.n / 2;
                const auto idx = static_cast<std::size_t>(center);
                series << p.name << "," << p.alpha << "," << p.rho << ","
                       << p.lambdaC << "," << p.lambdaR << ","
                       << equilibriumFraction(p) << "," << qOne(p) << ","
                       << static_cast<double>(step) * m.dt << ","
                       << s.v[idx] / s.psi[idx] << "," << energy(m, p, s) / e0
                       << "," << mass(m, s) << "\n";
            }
            if (step < m.steps) {
                solveStep(m, p, s);
            }
        }
    }
}

void runGroupD(const std::filesystem::path& out) {
    const std::vector<Params> cases {
        {.name = "D1", .alpha = 0.30, .rho = 0.30, .lambdaC = 0.25, .lambdaR = 0.25},
        {.name = "D2", .alpha = 0.30, .rho = 0.30, .lambdaC = 1.00, .lambdaR = 1.00},
        {.name = "D3", .alpha = 0.30, .rho = 0.30, .lambdaC = 5.00, .lambdaR = 5.00},
        {.name = "D4", .alpha = 0.30, .rho = 0.30, .lambdaC = 25.0, .lambdaR = 25.0},
    };
    auto series = csv(out / "GroupD_energy_ratio.csv");
    series << "case,alpha,rho,lambda_c,lambda_r,tau_exchange,time,center_ratio,energy_ratio,mass\n";
    auto summary = csv(out / "GroupD_t95_summary.csv");
    summary << "case,lambda_c,lambda_r,tau_exchange,t95,final_center_ratio,energy_ratio_final\n";
    for (const Params& p : cases) {
        const Mesh m = makeMesh(256, 10.0, 0.5);
        State s = makeState(m, sin2Profile, 0.0);
        const double e0 = energy(m, p, s);
        const double eq = equilibriumFraction(p);
        double t95 = -1.0;
        const int stride = std::max(1, m.steps / 1000);
        for (int step = 0; step <= m.steps; ++step) {
            const int center = m.n / 2;
            const auto idx = static_cast<std::size_t>(center);
            const double ratio = s.v[idx] / s.psi[idx];
            const double t = static_cast<double>(step) * m.dt;
            if (t95 < 0.0 && ratio >= 0.95 * eq) {
                t95 = t;
            }
            if (step == 0 || step % stride == 0 || step == m.steps) {
                series << p.name << "," << p.alpha << "," << p.rho << ","
                       << p.lambdaC << "," << p.lambdaR << ","
                       << 1.0 / (p.lambdaC + p.lambdaR) << "," << t << ","
                       << ratio << "," << energy(m, p, s) / e0 << ","
                       << mass(m, s) << "\n";
            }
            if (step < m.steps) {
                solveStep(m, p, s);
            }
        }
        const int center = m.n / 2;
        const auto idx = static_cast<std::size_t>(center);
        summary << p.name << "," << p.lambdaC << "," << p.lambdaR << ","
                << 1.0 / (p.lambdaC + p.lambdaR) << "," << t95 << ","
                << s.v[idx] / s.psi[idx] << "," << energy(m, p, s) / e0 << "\n";
    }
}

void runGroupE(const std::filesystem::path& out) {
    const std::vector<Params> cases {
        {.name = "E1", .alpha = 0.10, .rho = 0.30, .lambdaC = 2.0, .lambdaR = 6.0},
        {.name = "E2", .alpha = 0.30, .rho = 0.30, .lambdaC = 2.0, .lambdaR = 2.0},
        {.name = "E3", .alpha = 0.50, .rho = 0.30, .lambdaC = 2.0, .lambdaR = 1.2},
        {.name = "E4", .alpha = 0.70, .rho = 0.30, .lambdaC = 2.0, .lambdaR = 0.857},
    };
    auto series = csv(out / "GroupE_variance.csv");
    series << "case,alpha,rho,lambda_c,lambda_r,d_effective,eq_fraction,q1,time,variance,mass,center_ratio\n";
    for (const Params& p : cases) {
        const Mesh m = makeMesh(512, 0.01, 0.5);
        State s = makeState(m, gaussianDefault, 0.0);
        const int stride = std::max(1, m.steps / 300);
        for (int step = 0; step <= m.steps; ++step) {
            if (step == 0 || step % stride == 0 || step == m.steps) {
                const int center = m.n / 2;
                const auto idx = static_cast<std::size_t>(center);
                series << p.name << "," << p.alpha << "," << p.rho << ","
                       << p.lambdaC << "," << p.lambdaR << "," << 1.0 - p.alpha
                       << "," << equilibriumFraction(p) << "," << qOne(p) << ","
                       << static_cast<double>(step) * m.dt << "," << variance(m, s)
                       << "," << mass(m, s) << "," << s.v[idx] / s.psi[idx] << "\n";
            }
            if (step < m.steps) {
                solveStep(m, p, s);
            }
        }
    }
}

void runGroupF(const std::filesystem::path& out) {
    const Params p {.name = "F", .alpha = 0.42, .rho = 0.28, .lambdaC = 2.0, .lambdaR = 4.0 / 3.0};
    const std::vector<double> fractions {0.0, 0.3, 1.0};
    auto series = csv(out / "GroupF_initial_partitions.csv");
    series << "case,initial_fraction,eq_fraction,time,center_ratio,energy_ratio,mass\n";
    for (const double initialFraction : fractions) {
        const Mesh m = makeMesh(256, 5.0, 0.5);
        State s = makeState(m, sin2Profile, initialFraction);
        const double e0 = energy(m, p, s);
        const int stride = std::max(1, m.steps / 1000);
        for (int step = 0; step <= m.steps; ++step) {
            if (step == 0 || step % stride == 0 || step == m.steps) {
                const int center = m.n / 2;
                const auto idx = static_cast<std::size_t>(center);
                series << p.name << "," << initialFraction << ","
                       << equilibriumFraction(p) << "," << static_cast<double>(step) * m.dt
                       << "," << s.v[idx] / s.psi[idx] << ","
                       << energy(m, p, s) / e0 << "," << mass(m, s) << "\n";
            }
            if (step < m.steps) {
                solveStep(m, p, s);
            }
        }
    }
}

void runGroupG(const std::filesystem::path& out) {
    const std::vector<Params> cases {
        {.name = "G1", .alpha = 0.30, .rho = 0.40, .lambdaC = 2.0, .lambdaR = 2.667, .theta = 1.0},
        {.name = "G2", .alpha = 0.50, .rho = 0.50, .lambdaC = 2.0, .lambdaR = 2.0, .theta = 1.0},
        {.name = "G3", .alpha = 0.20, .rho = 0.30, .lambdaC = 2.0, .lambdaR = 3.0, .theta = 0.5},
    };
    auto series = csv(out / "GroupG_energy_decay.csv");
    series << "case,theta,alpha,rho,lambda_c,lambda_r,alpha_plus_rho,time,energy_ratio,mass,center_ratio\n";
    auto summary = csv(out / "GroupG_energy_summary.csv");
    summary << "case,theta,completed,stop_time,energy_ratio_final,max_energy_increase,positive_energy_increments,final_center_ratio,mass_final\n";
    for (const Params& p : cases) {
        const Mesh m = makeMesh(256, 5.0, 0.1);
        State s = makeState(m, sin2Profile, 0.0);
        const double e0 = energy(m, p, s);
        double maxIncrease = 0.0;
        int positive = 0;
        bool completed = true;
        double stopTime = m.tf;
        double previous = e0;
        const int stride = std::max(1, m.steps / 1500);
        for (int step = 0; step <= m.steps; ++step) {
            const double current = energy(m, p, s);
            const int center = m.n / 2;
            const auto idx = static_cast<std::size_t>(center);
            if (step == 0 || step % stride == 0 || step == m.steps) {
                series << p.name << "," << p.theta << "," << p.alpha << "," << p.rho
                       << "," << p.lambdaC << "," << p.lambdaR << "," << p.alpha + p.rho
                       << "," << static_cast<double>(step) * m.dt << ","
                       << current / e0 << "," << mass(m, s) << ","
                       << s.v[idx] / s.psi[idx] << "\n";
            }
            if (!std::isfinite(current) || !std::isfinite(mass(m, s))) {
                completed = false;
                stopTime = static_cast<double>(step) * m.dt;
                break;
            }
            if (step > 0) {
                const double inc = current - previous;
                maxIncrease = std::max(maxIncrease, inc);
                if (inc > 32.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, e0)) {
                    ++positive;
                }
                previous = current;
            }
            if (step < m.steps) {
                solveStep(m, p, s);
            }
        }
        const int center = m.n / 2;
        const auto idx = static_cast<std::size_t>(center);
        summary << p.name << "," << p.theta << "," << (completed ? 1 : 0) << ","
                << stopTime << "," << energy(m, p, s) / e0 << "," << maxIncrease
                << "," << positive << "," << s.v[idx] / s.psi[idx] << ","
                << mass(m, s) << "\n";
    }
}

void writeReadme(const std::filesystem::path& out) {
    std::ofstream r(out / "README.txt");
    r << "New paper TSOMPsiV sensitivity suite.\n"
      << "All cases use theta=1 except Group G case G3, which uses theta=1/2.\n"
      << "CSV files are grouped by GroupA ... GroupG.\n";
}

} // namespace

int main(int argc, char** argv) {
    PetscCall(PetscInitialize(&argc, &argv, nullptr, nullptr));
    try {
        const auto out = outDir();
        runGroupA(out);
        runGroupB(out);
        runGroupC(out);
        runGroupD(out);
        runGroupE(out);
        runGroupF(out);
        runGroupG(out);
        writeReadme(out);
        std::cout << "New paper suite completed. Output: " << out << "\n";
    } catch (const std::exception& ex) {
        std::cerr << "NewPaperPsiV failed: " << ex.what() << "\n";
        PetscCall(PetscFinalize());
        return 1;
    }
    PetscCall(PetscFinalize());
    return 0;
}
