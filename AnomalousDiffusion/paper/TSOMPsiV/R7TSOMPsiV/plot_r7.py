from pathlib import Path
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(path):
    with path.open() as file:
        return list(csv.DictReader(file))


out = Path(__file__).resolve().parent / "Saida"
rows = read_csv(out / "partition_sensitivity.csv")


def plot_subset(filename, scenarios, labels):
    plt.figure(figsize=(7.2, 4.6), dpi=180)
    colors = ["#1b7f5a", "#2f5eaa", "#8c4a9b", "#c46a2b"]
    for scenario, label, color in zip(scenarios, labels, colors):
        subset = [row for row in rows if row["scenario"] == scenario]
        t = [float(row["time"]) for row in subset]
        r = [float(row["v_over_psi_center"]) for row in subset]
        eq = float(subset[0]["v_equilibrium"])
        plt.plot(t, r, lw=2.0, color=color, label=label)
        plt.axhline(eq, color=color, lw=0.9, ls=":", alpha=0.7)
    plt.xlabel(r"$\tau$")
    plt.ylabel(r"$V(0.5,\tau)/\Psi(0.5,\tau)$")
    plt.xlim(0.0, 5.0)
    plt.ylim(-0.02, 1.02)
    plt.grid(True, color="0.88", lw=0.7)
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(out / f"{filename}.png")
    plt.savefig(out / f"{filename}.pdf")
    plt.close()


plot_subset(
    "partition_equilibrium_alpha_rho",
    [
        "alpha090_rho010_Lc1",
        "alpha075_rho025_Lc1",
        "alpha050_rho050_Lc1",
        "alpha025_rho075_Lc1",
    ],
    [
        r"$\alpha=0.90,\ \rho=0.10$",
        r"$\alpha=0.75,\ \rho=0.25$",
        r"$\alpha=0.50,\ \rho=0.50$",
        r"$\alpha=0.25,\ \rho=0.75$",
    ],
)

plot_subset(
    "partition_exchange_rate",
    [
        "alpha075_rho025_Lc0p5",
        "alpha075_rho025_Lc1",
        "alpha075_rho025_Lc2",
        "alpha075_rho025_Lc10",
    ],
    [
        r"$\Lambda_c=0.5$",
        r"$\Lambda_c=1$",
        r"$\Lambda_c=2$",
        r"$\Lambda_c=10$",
    ],
)

summary = read_csv(out / "summary.csv")
plt.figure(figsize=(5.8, 4.2), dpi=180)
subset = [
    row
    for row in summary
    if row["scenario"]
    in {
        "alpha075_rho025_Lc0p5",
        "alpha075_rho025_Lc1",
        "alpha075_rho025_Lc2",
        "alpha075_rho025_Lc10",
    }
]
x = [1.0 / float(row["lambda_exchange"]) for row in subset]
y = [float(row["t95"]) for row in subset]
labels = [row["lambda_c"] for row in subset]
plt.plot(x, y, color="#1b7f5a", marker="o", lw=2.0)
for xi, yi, label in zip(x, y, labels):
    plt.annotate(r"$\Lambda_c=" + label + r"$", (xi, yi), xytext=(5, 4),
                 textcoords="offset points", fontsize=8)
plt.xlabel(r"$1/(\Lambda_c+\Lambda_r)$")
plt.ylabel(r"$t_{95}$")
plt.grid(True, color="0.88", lw=0.7)
plt.tight_layout()
plt.savefig(out / "partition_t95_exchange_scale.png")
plt.savefig(out / "partition_t95_exchange_scale.pdf")
plt.close()

print(out / "partition_equilibrium_alpha_rho.png")
print(out / "partition_exchange_rate.png")
print(out / "partition_t95_exchange_scale.png")
