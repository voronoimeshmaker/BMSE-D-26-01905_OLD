from pathlib import Path
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(path):
    with path.open() as file:
        return list(csv.DictReader(file))


out = Path(__file__).resolve().parent / "Saida"

rows = read_csv(out / "tsom_v0_uv_final.csv")
x = [float(row["x"]) for row in rows]
component_series = {
    "Psi": [float(row["psi"]) for row in rows],
    "U = Psi - V": [float(row["u"]) for row in rows],
    "V": [float(row["v"]) for row in rows],
}

plt.figure(figsize=(7.2, 4.6), dpi=180)
for name, color in [
    ("Total concentration $\\Psi$", "#1b7f5a"),
    ("Mobile state $U=\\Psi-V$", "#2f5eaa"),
    ("Retained state $V$", "#8c4a9b"),
]:
    source_name = "Psi" if name.startswith("Total") else "U = Psi - V" if name.startswith("Mobile") else "V"
    plt.plot(x, component_series[source_name], label=name, lw=2.0, color=color)
plt.axhline(0.0, color="0.35", lw=0.8)
plt.xlabel(r"$\xi$")
plt.ylabel("field value")
plt.xlim(0.25, 0.75)
plt.grid(True, color="0.88", lw=0.7)
plt.legend(frameon=False, fontsize=9)
plt.tight_layout()
plt.savefig(out / "internal_partition_gaussian_v0.png")
plt.savefig(out / "internal_partition_gaussian_v0.pdf")
plt.close()

print(out / "internal_partition_gaussian_v0.png")
