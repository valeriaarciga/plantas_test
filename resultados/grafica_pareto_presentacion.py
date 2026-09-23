"""Grafica de presentacion: frentes de Pareto MILP vs NSGA-II, los 4 tamanos
con datos MILP reales, en paneles lado a lado. Usa los puntos exactos de
resultados/comparacion_hv_todos.json (misma corrida que sostiene la tabla
y el HV% ya reportados)."""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

with open("resultados/comparacion_hv_todos.json") as fh:
    DATA = json.load(fh)

AZUL = "#0072B2"     # MILP -- Okabe-Ito, mismo par usado en el reporte
NARANJA = "#C97A0A"  # NSGA-II
FONDO = "#FBF7EF"
TINTA = "#2B2620"
GRIS = "#8A8272"

HV_RATIO = {"6x6": 100.00, "8x8": 98.90, "10x10": 91.48, "12x12": 88.74}
MILP_CERRO = {"6x6": True, "8x8": False, "10x10": False, "12x12": False}

fig, axes = plt.subplots(1, 4, figsize=(18, 5), facecolor=FONDO)
fig.suptitle("Frente de Pareto: MILP (óptimo/incumbente) vs NSGA-II corregido",
             fontsize=19, color=TINTA, weight="bold", y=1.06, x=0.128, ha="left")
fig.text(0.128, 0.965,
         "competencia esperada ($f_1$, minimizar)  vs  aristas monoespecíficas ($f_2$, minimizar)  ·  misma instancia, mismo escenario, misma semilla",
         fontsize=11.5, color=GRIS, ha="left")

for ax, (nombre, d) in zip(axes, DATA.items()):
    ax.set_facecolor(FONDO)
    milp = np.array(sorted(d["milp"]["puntos"], key=lambda p: p[1]))
    nsga = np.array(sorted(d["nsga2"]["puntos"], key=lambda p: p[1]))

    # cuando los frentes casi coinciden (6x6: HV=100%), una serie tapa
    # completamente a la otra -- se separan con un microdesplazamiento
    # vertical simetrico, puramente visual, para que ambas se vean
    coinciden = HV_RATIO[nombre] >= 99.5
    jitter = (todo_y_range := max(milp[:, 1].max(), nsga[:, 1].max()) - min(milp[:, 1].min(), nsga[:, 1].min())) * 0.045 if coinciden else 0.0

    ax.plot(milp[:, 0], milp[:, 1] + jitter, color=AZUL, linewidth=2.4, zorder=3,
            marker="o", markersize=7, markerfacecolor=AZUL, markeredgecolor=FONDO, markeredgewidth=1.2,
            label="MILP")
    ax.plot(nsga[:, 0], nsga[:, 1] - jitter, color=NARANJA, linewidth=2.4, zorder=4,
            marker="D", markersize=6.5, markerfacecolor=NARANJA, markeredgecolor=FONDO, markeredgewidth=1.2,
            linestyle=(0, (1, 0)) if len(nsga) > 1 else "none",
            label="NSGA-II")
    if coinciden:
        ax.text(0.97, 0.93, "los dos frentes\ncoinciden exactamente", transform=ax.transAxes,
                fontsize=8.3, color=GRIS, ha="right", va="top", style="italic")

    todo_x = np.concatenate([milp[:, 0], nsga[:, 0]])
    todo_y = np.concatenate([milp[:, 1], nsga[:, 1]])
    pad_x = (todo_x.max() - todo_x.min()) * 0.18 + 0.05
    pad_y = (todo_y.max() - todo_y.min()) * 0.15 + 0.4
    ax.set_xlim(todo_x.min() - pad_x, todo_x.max() + pad_x)
    ax.set_ylim(todo_y.min() - pad_y, todo_y.max() + pad_y)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#CFC7B4")
    ax.tick_params(colors=GRIS, labelsize=9.5)
    ax.grid(axis="both", color="#E4DDC9", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    cerro = MILP_CERRO[nombre]
    badge = "MILP cerró a optimalidad" if cerro else "MILP con gap abierto (TIME_LIMIT)"
    badge_color = "#00815F" if cerro else "#9C6B00"
    ratio = HV_RATIO[nombre]
    ratio_color = "#00815F" if ratio >= 90 else "#9C6B00"

    ax.set_title(f"{nombre}", fontsize=16, color=TINTA, weight="bold", loc="left", pad=12)
    ax.text(0.0, 1.14, f"HV = {ratio:.1f}%", transform=ax.transAxes,
            fontsize=13, color=ratio_color, weight="bold", ha="left")
    ax.text(0.0, -0.20, badge, transform=ax.transAxes, fontsize=8.7, color=badge_color, ha="left")

    ax.set_xlabel("$f_1$  competencia", fontsize=10, color=GRIS)
    if ax is axes[0]:
        ax.set_ylabel("$f_2$  aristas monoespecíficas", fontsize=10, color=GRIS)

handles = [
    plt.Line2D([0], [0], color=AZUL, marker="o", markersize=8, linewidth=2.4,
               markerfacecolor=AZUL, markeredgecolor=FONDO, label="MILP (óptimo / mejor incumbente)"),
    plt.Line2D([0], [0], color=NARANJA, marker="D", markersize=7, linewidth=2.4,
               markerfacecolor=NARANJA, markeredgecolor=FONDO, label="NSGA-II (corregido)"),
]
leg = fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=12,
                  frameon=False, bbox_to_anchor=(0.5, -0.06), handlelength=2.2)
for t in leg.get_texts():
    t.set_color(TINTA)

fig.text(0.128, -0.14,
         "Escenario semilla 0 en cada instancia. HV = hipervolumen de NSGA-II como % del hipervolumen del MILP,\n"
         "punto de referencia fijo e independiente de ambos algoritmos. 12×12: NSGA-II no exploró f2<4 aunque el MILP encontró f2=0 factible.",
         fontsize=9.5, color=GRIS, ha="left")

plt.tight_layout(rect=[0, 0.02, 1, 0.94])
plt.savefig("resultados/mapas/pareto_presentacion.png", dpi=220, bbox_inches="tight", facecolor=FONDO)
plt.savefig("resultados/mapas/pareto_presentacion.svg", bbox_inches="tight", facecolor=FONDO)
print("guardado: resultados/mapas/pareto_presentacion.png (y .svg)")
