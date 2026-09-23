"""Genera los paneles MILP/NSGA-II (y el panel de diferencias para 12x12)
reutilizando graficar_bello() y sus piezas internas (PALETA, FONDO, _coords_hex)
de visualizar_bello.py -- no reimplementa el motor de dibujo hexagonal."""
import sys, os, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, Circle, Patch
from core import SP
from visualizar_bello import graficar_bello, PALETA, FONDO, COLOR_TEXTO, _coords_hex

OUT = "resultados/mapas"
os.makedirs(OUT, exist_ok=True)


def graficar_diferencias(P, s_a, s_b, esc, label_a, label_b, subtitulo, archivo,
                          mostrar_texto=False, radio=0.58, dpi=170):
    """Mapa coloreado por s_b (NSGA-II), con un marcador distintivo en los
    nodos donde s_a (MILP) asigno una especie distinta -- el 'gap' visible."""
    R, C = P.R, P.C
    dif = (s_a != s_b) & (esc < 0)  # solo cuenta diferencias en nodos libres
    fig_w = min(20, max(7, C * 0.40))
    fig_h = min(14, max(5.5, R * 0.40)) + 1.6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=FONDO)
    ax.set_facecolor(FONDO)

    for r in range(R):
        for c in range(C):
            v = r * C + c
            x, y = _coords_hex(r, c)
            especie = s_b[v]
            fijo = esc[v] >= 0
            hexagon = RegularPolygon((x, y), numVertices=6, radius=radio, orientation=np.pi / 6,
                                      facecolor=PALETA[SP[especie]], edgecolor=FONDO,
                                      linewidth=1.1, zorder=2)
            ax.add_patch(hexagon)
            if fijo:
                ax.add_patch(Circle((x, y), radio * 0.22, facecolor=FONDO, edgecolor="none",
                                     zorder=3, alpha=0.85))
            if dif[v]:
                ax.add_patch(Circle((x, y), radio * 0.30, facecolor="none",
                                     edgecolor="#2B2620", linewidth=1.4, zorder=5))
            if mostrar_texto:
                ax.text(x, y - radio * 0.62, SP[especie], ha='center', va='top',
                        fontsize=6.2, color=COLOR_TEXTO, alpha=0.75, zorder=4, family='monospace')

    ax.set_xlim(-1, C + 1)
    ax.set_ylim(-R * np.sqrt(3) / 2 - 1.4, 1.4)
    ax.set_aspect('equal')
    ax.axis('off')

    n_dif = int(dif.sum())
    n_libres = int((esc < 0).sum())
    ax.text(0.5, 1.10, subtitulo, transform=ax.transAxes, ha='center', va='bottom',
            fontsize=15, color=COLOR_TEXTO, weight='bold', family='sans-serif')
    stats = (f"coloreado por {label_b}  ·  círculo = nodo donde {label_a} asigna otra especie  ·  "
             f"{n_dif}/{n_libres} nodos libres difieren ({n_dif/n_libres:.0%})")
    ax.text(0.5, 1.045, stats, transform=ax.transAxes, ha='center', va='bottom',
            fontsize=9.5, color="#6B6459", family='sans-serif')

    handles = [Patch(facecolor=PALETA[sp], edgecolor=FONDO, label=sp) for sp in SP]
    leg = fig.legend(handles=handles, loc='lower center', ncol=10, fontsize=7.5, frameon=False,
                      bbox_to_anchor=(0.5, -0.01), handlelength=1.3, handleheight=1.3, columnspacing=1.2)
    for t in leg.get_texts():
        t.set_color(COLOR_TEXTO)

    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    plt.savefig(archivo, dpi=dpi, bbox_inches='tight', facecolor=FONDO)
    plt.close(fig)
    return n_dif, n_libres


def graficar_con_aristas_mono(P, s, esc, subtitulo, archivo, mostrar_texto=False,
                               radio=0.58, dpi=170):
    """Mapa normal + una linea gruesa marcando cada arista monoespecifica
    (los pares que cuentan literalmente para f2) -- visualiza f2 en el
    espacio, no solo el numero."""
    R, C = P.R, P.C
    idx_mono = np.where(s[P.U] == s[P.V])[0]
    fig_w = min(20, max(7, C * 0.40))
    fig_h = min(14, max(5.5, R * 0.40)) + 1.6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=FONDO)
    ax.set_facecolor(FONDO)

    for r in range(R):
        for c in range(C):
            v = r * C + c
            x, y = _coords_hex(r, c)
            especie = s[v]
            fijo = esc[v] >= 0
            hexagon = RegularPolygon((x, y), numVertices=6, radius=radio, orientation=np.pi / 6,
                                      facecolor=PALETA[SP[especie]], edgecolor=FONDO,
                                      linewidth=1.1, zorder=2)
            ax.add_patch(hexagon)
            if fijo:
                ax.add_patch(Circle((x, y), radio * 0.22, facecolor=FONDO, edgecolor="none",
                                     zorder=3, alpha=0.85))
            if mostrar_texto:
                ax.text(x, y - radio * 0.62, SP[especie], ha='center', va='top',
                        fontsize=6.2, color=COLOR_TEXTO, alpha=0.75, zorder=4, family='monospace')

    for a in idx_mono:
        u, v = int(P.U[a]), int(P.V[a])
        xu, yu = _coords_hex(u // C, u % C)
        xv, yv = _coords_hex(v // C, v % C)
        ax.plot([xu, xv], [yu, yv], color="#B23A2E", linewidth=3.2, zorder=6,
                solid_capstyle="round")
        ax.plot([xu, xv], [yu, yv], color="#F2E4D0", linewidth=1.0, zorder=7,
                solid_capstyle="round")

    ax.set_xlim(-1, C + 1)
    ax.set_ylim(-R * np.sqrt(3) / 2 - 1.4, 1.4)
    ax.set_aspect('equal')
    ax.axis('off')

    f1, f2 = P.f1(s), P.f2(s)
    ax.text(0.5, 1.10, subtitulo, transform=ax.transAxes, ha='center', va='bottom',
            fontsize=15, color=COLOR_TEXTO, weight='bold', family='sans-serif')
    stats = (f"$f_1$={f1:.2f}   $f_2$={f2}   línea roja = arista monoespecífica "
             f"(el par que cuenta para $f_2$)" if f2 > 0 else
             f"$f_1$={f1:.2f}   $f_2$={f2}   sin aristas monoespecíficas")
    ax.text(0.5, 1.045, stats, transform=ax.transAxes, ha='center', va='bottom',
            fontsize=9.5, color="#6B6459", family='sans-serif')

    handles = [Patch(facecolor=PALETA[sp], edgecolor=FONDO, label=sp) for sp in SP]
    leg = fig.legend(handles=handles, loc='lower center', ncol=10, fontsize=7.5, frameon=False,
                      bbox_to_anchor=(0.5, -0.01), handlelength=1.3, handleheight=1.3, columnspacing=1.2)
    for t in leg.get_texts():
        t.set_color(COLOR_TEXTO)

    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    plt.savefig(archivo, dpi=dpi, bbox_inches='tight', facecolor=FONDO)
    plt.close(fig)
    return len(idx_mono)


with open("resultados/soluciones_mapas.pkl", "rb") as fh:
    sol = pickle.load(fh)

for nombre, d in sol.items():
    P, esc = d["P"], d["esc"]
    graficar_bello(P, d["s_milp"], esc,
                    subtitulo=f"MILP (óptimo/incumbente) — {nombre} — punto f2={d['f2_milp']}",
                    archivo_salida=f"{OUT}/{nombre}_milp.png",
                    mostrar_texto=(P.nV <= 130))
    graficar_bello(P, d["s_nsga"], esc,
                    subtitulo=f"NSGA-II (corregido) — {nombre} — punto f2={d['f2_nsga']}",
                    archivo_salida=f"{OUT}/{nombre}_nsga.png",
                    mostrar_texto=(P.nV <= 130))
    print(f"{nombre}: MILP f2={d['f2_milp']} f1={d['f1_milp']:.4f}  |  "
          f"NSGA-II f2={d['f2_nsga']} f1={d['f1_nsga']:.4f}  -> mapas guardados")

# Panel de diferencias, foco en 12x12 (donde HV cayo a 88.7%)
d = sol["12x12"]
n_dif, n_libres = graficar_diferencias(
    d["P"], d["s_milp"], d["s_nsga"], d["esc"], "MILP", "NSGA-II",
    subtitulo="12×12 — dónde difieren MILP y NSGA-II (HV=88.7%)",
    archivo=f"{OUT}/12x12_diferencias.png", mostrar_texto=False)
print(f"12x12 diferencias: {n_dif}/{n_libres} nodos libres difieren "
      f"(nota: alto por simetria de permutacion de especies con f1 similar, "
      f"no por una region espacial concentrada -- ver panel de aristas mono abajo)")

# Panel mas preciso: las aristas monoespecificas EXACTAS que cuentan para f2
n_mono_nsga = graficar_con_aristas_mono(
    d["P"], d["s_nsga"], d["esc"],
    subtitulo="12×12 — NSGA-II: las 4 aristas que forman su f2=4",
    archivo=f"{OUT}/12x12_nsga_aristas.png")
n_mono_milp = graficar_con_aristas_mono(
    d["P"], d["s_milp"], d["esc"],
    subtitulo="12×12 — MILP: cero aristas monoespecíficas (f2=0)",
    archivo=f"{OUT}/12x12_milp_aristas.png")
print(f"12x12 aristas monoespecificas: NSGA-II={n_mono_nsga}  MILP={n_mono_milp}")

# 14x47
with open("resultados/solucion_14x47.pkl", "rb") as fh:
    d47 = pickle.load(fh)
graficar_bello(d47["P"], d47["s"], d47["esc"],
               subtitulo=f"Escala real 14×47 (1 ha) — solución NSGA-II, punto f1 mínimo",
               archivo_salida=f"{OUT}/14x47.png", mostrar_texto=False)
print("14x47: mapa guardado")
