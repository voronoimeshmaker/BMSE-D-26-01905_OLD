from pathlib import Path
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(path):
    with path.open() as file:
        return list(csv.DictReader(file))


out = Path(__file__).resolve().parent / "Saida"
ratio_rows = read_csv(out / "center_ratio_and_exchange.csv")
energy_rows = read_csv(out / "energy.csv")

labels = {
    "R5a": r"$V_0=0$",
    "R5b": r"$V_0=\Psi_0/2$",
    "R5c": r"$V_0=\Psi_0$",
}
colors = {"R5a": "#1b7f5a", "R5b": "#2f5eaa", "R5c": "#9b2d30"}


def subset(rows, case):
    return [row for row in rows if row["case"] == case]


plt.figure(figsize=(7.2, 4.6), dpi=180)
for case in ["R5a", "R5b", "R5c"]:
    rows = subset(ratio_rows, case)
    t = [float(row["time"]) for row in rows]
    r = [float(row["v_over_psi_center"]) for row in rows]
    plt.plot(t, r, lw=2.0, color=colors[case], label=labels[case])
plt.axhline(0.75, color="0.2", lw=1.0, ls=":", label=r"equilibrium $0.75$")
plt.xlabel(r"$\tau$")
plt.ylabel(r"$V(0.5,\tau)/\Psi(0.5,\tau)$")
plt.xlim(0.0, 5.0)
plt.ylim(-0.05, 1.70)
plt.grid(True, color="0.88", lw=0.7)
plt.legend(frameon=False, fontsize=9)
plt.tight_layout()
plt.savefig(out / "initial_partition_center_ratio.png")
plt.savefig(out / "initial_partition_center_ratio.pdf")
plt.close()

plt.figure(figsize=(7.2, 4.6), dpi=180)
for case in ["R5a", "R5b", "R5c"]:
    rows = subset(ratio_rows, case)
    t = [float(row["time"]) for row in rows if float(row["time"]) > 0.0]
    y = [float(row["log_abs_ratio_minus_v_equilibrium"]) for row in rows if float(row["time"]) > 0.0]
    plt.plot(t, y, lw=2.0, color=colors[case], label=labels[case])
plt.xlabel(r"$\tau$")
plt.ylabel(r"$\log |V(0.5,\tau)/\Psi(0.5,\tau)-0.75|$")
plt.xlim(0.0, 5.0)
plt.grid(True, color="0.88", lw=0.7)
plt.legend(frameon=False, fontsize=9)
plt.tight_layout()
plt.savefig(out / "initial_partition_log_relaxation.png")
plt.savefig(out / "initial_partition_log_relaxation.pdf")
plt.close()

plt.figure(figsize=(7.2, 4.6), dpi=180)
for case in ["R5a", "R5b", "R5c"]:
    rows = subset(energy_rows, case)
    t = [float(row["time"]) for row in rows]
    e0 = float(rows[0]["free_energy"])
    e = [float(row["free_energy"]) / e0 for row in rows]
    plt.plot(t, e, lw=2.0, color=colors[case], label=labels[case])
plt.xlabel(r"$\tau$")
plt.ylabel(r"$\mathcal{F}(\tau)/\mathcal{F}(0)$")
plt.xlim(0.0, 5.0)
plt.grid(True, color="0.88", lw=0.7)
plt.legend(frameon=False, fontsize=9)
plt.tight_layout()
plt.savefig(out / "initial_partition_energy.png")
plt.savefig(out / "initial_partition_energy.pdf")
plt.close()

print(out / "initial_partition_center_ratio.png")
print(out / "initial_partition_log_relaxation.png")
print(out / "initial_partition_energy.png")
