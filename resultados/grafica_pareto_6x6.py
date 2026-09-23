"""Grafica de presentacion: frente de Pareto SOLO 6x6, MILP vs NSGA-II."""
import json
import numpy as np
import matplotlib.pyplot as plt

with open("resultados/comparacion_hv_todos.json") as fh:
    DATA = json.load(fh)

AZUL = "#0072B2"
NARANJA = "#C97A0A"
FONDO = "#FBF7EF"
TINTA = "#2B2620"
GRIS = "#8A8272"

d = DATA["6x6"]
milp = np.array(sorted(d["milp"]["puntos"], key=lambda p: p[1]))
nsga = np.array(sorted(d["nsga2"]["puntos"], key=lambda p: p[1]))

fig, ax = plt.subplots(figsize=(8, 6), facecolor=FONDO)
ax.set_facecolor(FONDO)

# los dos frentes coinciden exactamente -> microdesplazamiento vertical
# simetrico, puramente visual, para que ambas series se vean
jitter = (milp[:, 1].max() - milp[:, 1].min()) * 0.05

ax.plot(milp[:, 0], milp[:, 1] + jitter, color=AZUL, linewidth=2.6, zorder=3,
        marker="o", markersize=10, markerfacecolor=AZUL, markeredgecolor=FONDO, markeredgewidth=1.4,
        label="MILP")
ax.plot(nsga[:, 0], nsga[:, 1] - jitter, color=NARANJA, linewidth=2.6, zorder=4,
        marker="D", markersize=9, markerfacecolor=NARANJA, markeredgecolor=FONDO, markeredgewidth=1.4,
        label="NSGA-II")

for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
for spine in ("left", "bottom"):
    ax.spines[spine].set_color("#CFC7B4")
ax.tick_params(colors=GRIS, labelsize=11)
ax.grid(axis="both", color="#E4DDC9", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

ax.set_xlabel("$f_1$ — competencia", fontsize=12.5, color=TINTA, labelpad=10)
ax.set_ylabel("$f_2$ — monocultivo", fontsize=12.5, color=TINTA, labelpad=10)
ax.set_title("Frente de Pareto — 6×6", fontsize=20, color=TINTA, weight="bold", loc="left", pad=16)

leg = ax.legend(loc="upper right", fontsize=13, frameon=False, handlelength=2.2, labelcolor=TINTA)

pad_x = (milp[:, 0].max() - milp[:, 0].min()) * 0.15 + 0.04
pad_y = (milp[:, 1].max() - milp[:, 1].min()) * 0.20 + 0.3
ax.set_xlim(milp[:, 0].min() - pad_x, milp[:, 0].max() + pad_x)
ax.set_ylim(milp[:, 1].min() - pad_y, milp[:, 1].max() + pad_y)

plt.tight_layout()
plt.savefig("resultados/mapas/pareto_6x6_presentacion.png", dpi=220, bbox_inches="tight", facecolor=FONDO)
plt.savefig("resultados/mapas/pareto_6x6_presentacion.svg", bbox_inches="tight", facecolor=FONDO)
print("guardado: resultados/mapas/pareto_6x6_presentacion.png (y .svg)")
