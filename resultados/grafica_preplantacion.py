"""Diagrama de la vegetacion PRE-EXISTENTE (antes de sembrar), tal como la
genera la simulacion de Montecarlo (P.escenario(seed)) -- nodos libres se
muestran vacios (aun no hay decision de que plantar ahi), nodos ocupados
se colorean por la especie que ya esta en el terreno. Reutiliza PALETA,
FONDO y _coords_hex de visualizar_bello.py. Layout con ejes posicionados a
mano (fig.add_axes) para que aspect='equal' no deje margenes muertos."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, Patch
from core import Problema, SP, P_OCU
from visualizar_bello import PALETA, FONDO, COLOR_TEXTO, _coords_hex

R, C = 8, 8
SEED = 0
P = Problema(R, C, t=0.10)
esc = P.escenario(seed=SEED)

VACIO = "#EFE7D4"
BORDE_VACIO = "#D8CCAE"
radio = 0.58

# --- contenido en "unidades de hexagono" ---
w_units = C + 0.9
h_units = R * np.sqrt(3) / 2 + 0.6
escala = 0.66  # pulgadas por unidad
plot_w = w_units * escala
plot_h = h_units * escala

TITULO_IN = 0.95   # alto reservado arriba, en pulgadas
LEYENDA_IN = 0.75  # alto reservado abajo, en pulgadas
MARGEN_IN = 0.15

fig_w = plot_w + 2 * MARGEN_IN
fig_h = plot_h + TITULO_IN + LEYENDA_IN + MARGEN_IN
fig = plt.figure(figsize=(fig_w, fig_h), facecolor=FONDO)

ax = fig.add_axes([
    MARGEN_IN / fig_w,
    LEYENDA_IN / fig_h,
    plot_w / fig_w,
    plot_h / fig_h,
])
ax.set_facecolor(FONDO)

for r in range(R):
    for c in range(C):
        v = r * C + c
        x, y = _coords_hex(r, c)
        ocupado = esc[v] >= 0
        if ocupado:
            color, edge, lw, z = PALETA[SP[esc[v]]], FONDO, 1.1, 2
        else:
            color, edge, lw, z = VACIO, BORDE_VACIO, 0.8, 1
        hexagon = RegularPolygon((x, y), numVertices=6, radius=radio, orientation=np.pi / 6,
                                  facecolor=color, edgecolor=edge, linewidth=lw, zorder=z)
        ax.add_patch(hexagon)

ax.set_xlim(-0.75, C + 0.15)
ax.set_ylim(-R * np.sqrt(3) / 2 - 0.45, 0.75)
ax.set_aspect("equal")
ax.axis("off")

n_ocupados = int((esc >= 0).sum())
fig.text(0.5, 1 - (TITULO_IN * 0.42) / fig_h, "Vegetación preexistente antes de sembrar",
          ha="center", va="center", fontsize=17, color=COLOR_TEXTO, weight="bold", family="sans-serif")
fig.text(0.5, 1 - (TITULO_IN * 0.78) / fig_h,
          f"simulación de Montecarlo (semilla {SEED})  ·  {n_ocupados}/{P.nV} nodos ocupados "
          f"({n_ocupados/P.nV:.1%}, ~{P_OCU:.1%} esperado)  ·  el resto queda libre para plantar",
          ha="center", va="center", fontsize=10.5, color="#6B6459", family="sans-serif")

handles = [Patch(facecolor=PALETA[sp], edgecolor=FONDO, label=sp) for sp in SP]
handles.append(Patch(facecolor=VACIO, edgecolor=BORDE_VACIO, label="libre"))
leg = fig.legend(handles=handles, loc="center", ncol=11, fontsize=8.5, frameon=False,
                  bbox_to_anchor=(0.5, (LEYENDA_IN * 0.5) / fig_h),
                  handlelength=1.3, handleheight=1.3, columnspacing=1.2)
for t in leg.get_texts():
    t.set_color(COLOR_TEXTO)

out = f"resultados/mapas/preplantacion_{R}x{C}"
plt.savefig(out + ".png", dpi=200, facecolor=FONDO, bbox_inches="tight", pad_inches=0.2)
plt.savefig(out + ".svg", facecolor=FONDO, bbox_inches="tight", pad_inches=0.2)
print(f"guardado: {out}.png  ({n_ocupados}/{P.nV} ocupados)  figsize={fig_w:.2f}x{fig_h:.2f}in")
