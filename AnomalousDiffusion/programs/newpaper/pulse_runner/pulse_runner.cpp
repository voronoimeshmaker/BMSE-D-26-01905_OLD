#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numbers>
#include <sstream>
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
    int n {0};
    double cdt {0.0};
    double tf {0.0};
    int steps {0};
    double sigma0 {0.04};
    double mu0 {0.5};
    double v0Fraction {0.0};
    double dEff {0.0};
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

struct ReferenceRow {
    int cell {0};
    double x {0.0};
    double psi0 {0.0};
    double psiFinal {0.0};
};

std::filesystem::path sourceDir() {
    return std::filesystem::path(__FILE__).parent_path().parent_path();
}

std::filesystem::path inputDir() {
    return sourceDir() / "pulse_inputs";
}

std::filesystem::path outputDir() {
    const auto dir = sourceDir() / "pulse_results";
    std::filesystem::create_directories(dir);
    return dir;
}

std::ofstream openCsv(const std::filesystem::path& path) {
    std::ofstream f(path);
    if (!f) {
        throw std::runtime_error("Could not open " + path.string());
    }
    f << std::setprecision(16);
    return f;
}

Params readParams(const std::filesystem::path& yamlPath) {
    const YAML::Node y = YAML::LoadFile(yamlPath.string());
    Params p;
    p.theta = y["model"]["theta"].as<double>();
    p.alpha = y["model"]["alpha"].as<double>();
    p.rho = y["model"]["rho"].as<double>();
    p.lambdaC = y["model"]["lambda_c"].as<double>();
    p.lambdaR = y["model"]["lambda_r"].as<double>();
    p.n = y["mesh"]["N"].as<int>();
    p.cdt = y["time"]["C"].as<double>();
    p.tf = y["time"]["tau_f"].as<double>();
    p.steps = y["time"]["steps"].as<int>();
    p.sigma0 = y["initial_condition"]["sigma0"].as<double>();
    p.mu0 = y["initial_condition"]["mu0"].as<double>();
    p.v0Fraction = y["initial_condition"]["V0"].as<double>();
    p.name = y["output"]["case_name"].as<std::string>();
    p.dEff = y["output"]["D_eff"].as<double>();
    return p;
}

Mesh meshFrom(const Params& p) {
    const double h = 1.0 / static_cast<double>(p.n);
    const double requested = p.cdt * h * h;
    const int computedSteps = static_cast<int>(std::ceil(p.tf / requested));
    const int steps = p.steps > 0 ? p.steps : computedSteps;
    return {.n = p.n, .h = h, .dt = p.tf / static_cast<double>(steps), .tf = p.tf, .steps = steps};
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

double gaussian(const double x, const double sigma, const double mu) {
    const double z = (x - mu) / sigma;
    return std::exp(-0.5 * z * z) / (sigma * std::sqrt(2.0 * std::numbers::pi));
}

State initialState(const Mesh& m, const Params& p) {
    State s;
    s.psi.assign(static_cast<std::size_t>(m.n), 0.0);
    s.v.assign(static_cast<std::size_t>(m.n), 0.0);
    for (int i = 0; i < m.n; ++i) {
        const auto idx = static_cast<std::size_t>(i);
        s.psi[idx] = gaussian(xCell(m, i), p.sigma0, p.mu0);
        s.v[idx] = p.v0Fraction * s.psi[idx];
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
        throw std::runtime_error("Singular block.");
    }
    return {.a00 = a.a11 / det, .a01 = -a.a01 / det, .a10 = -a.a10 / det, .a11 = a.a00 / det};
}

Matrix2 blockForL(const Params& p,
                  const Coeff& c,
                  const double dt,
                  const double lcoef,
                  const bool diagonal) {
    Matrix2 a;
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

void solveStep(const Mesh& m, const Params& p, State& s) {
    const Coeff c = coeffs(p);
    const double h2inv = 1.0 / (m.h * m.h);
    std::vector<Matrix2> lower(static_cast<std::size_t>(m.n), zero2());
    std::vector<Matrix2> diag(static_cast<std::size_t>(m.n), zero2());
    std::vector<Matrix2> upper(static_cast<std::size_t>(m.n), zero2());
    std::vector<Vec2> rhs(static_cast<std::size_t>(m.n));

    for (int i = 0; i < m.n; ++i) {
        const auto idx = static_cast<std::size_t>(i);
        const double ldiag = (i == 0 || i == m.n - 1) ? -h2inv : -2.0 * h2inv;
        diag[idx] = blockForL(p, c, m.dt, ldiag, true);
        if (i > 0) {
            lower[idx] = blockForL(p, c, m.dt, h2inv, false);
        }
        if (i + 1 < m.n) {
            upper[idx] = blockForL(p, c, m.dt, h2inv, false);
        }
        rhs[idx] = {.x0 = s.psi[idx], .x1 = s.v[idx]};
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

double mass(const Mesh& m, const std::vector<double>& values) {
    double sum = 0.0;
    for (const double value : values) {
        sum += value;
    }
    return m.h * sum;
}

double variance(const Mesh& m, const std::vector<double>& psi) {
    const double total = mass(m, psi);
    double sum = 0.0;
    for (int i = 0; i < m.n; ++i) {
        const double dx = xCell(m, i) - 0.5;
        sum += dx * dx * psi[static_cast<std::size_t>(i)];
    }
    return m.h * sum / total;
}

double minValue(const std::vector<double>& values) {
    return *std::min_element(values.begin(), values.end());
}

double maxValue(const std::vector<double>& values) {
    return *std::max_element(values.begin(), values.end());
}

std::map<std::string, std::vector<ReferenceRow>> readReference(const std::filesystem::path& path) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("Could not open reference " + path.string());
    }
    std::map<std::string, std::vector<ReferenceRow>> data;
    std::string line;
    std::getline(f, line);
    while (std::getline(f, line)) {
        std::stringstream ss(line);
        std::vector<std::string> fields;
        std::string field;
        while (std::getline(ss, field, ',')) {
            fields.push_back(field);
        }
        if (fields.size() < 8) {
            continue;
        }
        data[fields[0]].push_back({
            .cell = std::stoi(fields[1]),
            .x = std::stod(fields[2]),
            .psi0 = std::stod(fields[3]),
            .psiFinal = std::stod(fields[4]),
        });
    }
    return data;
}

double l2AgainstReference(const Mesh& m,
                          const std::vector<double>& psi,
                          const std::vector<ReferenceRow>& ref) {
    if (ref.size() != psi.size()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double sum = 0.0;
    for (int i = 0; i < m.n; ++i) {
        const double e = psi[static_cast<std::size_t>(i)] - ref[static_cast<std::size_t>(i)].psiFinal;
        sum += e * e;
    }
    return std::sqrt(m.h * sum);
}

void writeProfile(const std::filesystem::path& path,
                  const Mesh& m,
                  const Params& p,
                  const State& initial,
                  const State& final,
                  const std::vector<ReferenceRow>* ref) {
    auto f = openCsv(path);
    f << "case,cell,x,psi0,v0,u0,psi_final,v_final,u_final,ratio_final,fickian_ref\n";
    for (int i = 0; i < m.n; ++i) {
        const auto idx = static_cast<std::size_t>(i);
        const double refValue = ref != nullptr && ref->size() == final.psi.size()
            ? (*ref)[idx].psiFinal
            : std::numeric_limits<double>::quiet_NaN();
        f << p.name << "," << i + 1 << "," << xCell(m, i) << ","
          << initial.psi[idx] << "," << initial.v[idx] << ","
          << initial.psi[idx] - initial.v[idx] << ","
          << final.psi[idx] << "," << final.v[idx] << ","
          << final.psi[idx] - final.v[idx] << ","
          << final.v[idx] / final.psi[idx] << "," << refValue << "\n";
    }
}

std::vector<std::filesystem::path> inputYamlFiles() {
    std::vector<std::filesystem::path> files;
    for (const auto& entry : std::filesystem::directory_iterator(inputDir())) {
        if (entry.path().extension() == ".yaml") {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto out = outputDir();
        const auto refs = readReference(inputDir() / "fickian_reference.csv");
        auto summary = openCsv(out / "pulse_summary.csv");
        summary << "case,alpha,rho,lambda_c,lambda_r,theta,n,dt,steps,d_eff,"
                << "mass0,mass_final,mass_drift,min_psi,min_u,min_v,max_psi,"
                << "variance_final,l2_error_fickian,center_ratio,eq_fraction\n";

        std::vector<std::filesystem::path> yamls;
        if (argc > 1) {
            for (int i = 1; i < argc; ++i) {
                yamls.emplace_back(argv[i]);
            }
        } else {
            yamls = inputYamlFiles();
        }

        for (const auto& yaml : yamls) {
            const std::filesystem::path fullYaml =
                yaml.is_absolute() ? yaml : inputDir() / yaml;
            const Params p = readParams(fullYaml);
            const Mesh m = meshFrom(p);
            State s = initialState(m, p);
            const State initial = s;
            const double m0 = mass(m, s.psi);
            for (int step = 0; step < m.steps; ++step) {
                solveStep(m, p, s);
            }

            std::vector<double> u(static_cast<std::size_t>(m.n), 0.0);
            for (int i = 0; i < m.n; ++i) {
                const auto idx = static_cast<std::size_t>(i);
                u[idx] = s.psi[idx] - s.v[idx];
            }
            const auto refIt = refs.find(p.name);
            const std::vector<ReferenceRow>* ref = refIt != refs.end() ? &refIt->second : nullptr;
            const double l2 = ref != nullptr
                ? l2AgainstReference(m, s.psi, *ref)
                : std::numeric_limits<double>::quiet_NaN();
            const int center = m.n / 2;
            const auto centerIdx = static_cast<std::size_t>(center);
            summary << p.name << "," << p.alpha << "," << p.rho << ","
                    << p.lambdaC << "," << p.lambdaR << "," << p.theta << ","
                    << p.n << "," << m.dt << "," << m.steps << "," << p.dEff << ","
                    << m0 << "," << mass(m, s.psi) << "," << mass(m, s.psi) - m0 << ","
                    << minValue(s.psi) << "," << minValue(u) << "," << minValue(s.v) << ","
                    << maxValue(s.psi) << "," << variance(m, s.psi) << "," << l2 << ","
                    << s.v[centerIdx] / s.psi[centerIdx] << ","
                    << p.alpha / (p.alpha + p.rho) << "\n";
            writeProfile(out / (p.name + "_profile.csv"), m, p, initial, s, ref);
            std::cout << "Completed " << p.name << "\n";
        }

        std::ofstream readme(out / "README.txt");
        readme << "Pulse benchmark outputs generated by pulse_runner.\n"
               << "pulse_summary.csv contains one row per YAML input.\n"
               << "<case>_profile.csv contains initial/final Psi,V,U and Fickian reference.\n";
        std::cout << "Pulse results written to " << out << "\n";
    } catch (const std::exception& ex) {
        std::cerr << "pulse_runner failed: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
