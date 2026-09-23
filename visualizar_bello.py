"""
visualizar_bello.py — Version presentacion del mapa hexagonal, con paleta
botanica curada, tipografia limpia y marcado elegante de preexistentes.
Reutiliza Problema de core.py, mismo mecanismo que visualizar.py.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import RegularPolygon, Patch, Circle
from matplotlib.collections import PatchCollection
from core import Problema, SP, NOMBRE

# Paleta botanica: tonos inspirados en la vegetacion real del Altiplano,
# maximamente distinguibles pero sin la saturacion "de cartoon" de tab10.
PALETA = {
    "AL":  "#4A7C59",  # Agave lechuguilla   - verde agave oscuro
    "AS":  "#8FAE5D",  # Agave salmiana      - verde oliva (especie dominante, tono suave)
    "ASc": "#2F6B4F",  # Agave scabra        - verde bosque
    "ASt": "#6FA287",  # Agave striata       - verde salvia
    "OC":  "#D98E5C",  # Opuntia cantabrigiensis - terracota
    "OE":  "#C17A4A",  # Opuntia engelmannii - terracota oscuro
    "OR":  "#E8B04B",  # Opuntia robusta     - ocre dorado
    "OS":  "#B5C9A4",  # Opuntia streptacantha - verde polvo
    "PL":  "#7B5B45",  # Prosopis laevigata  - marron mezquite (la nodriza)
    "YF":  "#C9A66B",  # Yucca filifera      - beige arena
}
COLORES = np.array([PALETA[sp] for sp in SP])
FONDO = "#FBF7EF"     # crema calido, evoca papel/terreno arido
COLOR_TEXTO = "#2B2620"


def _coords_hex(r, c):
    x = c + (0.5 if r % 2 == 1 else 0.0)
    y = -r * np.sqrt(3) / 2
    return x, y


def graficar_bello(P, s, esc, subtitulo="", archivo_salida="mapa.png",
                     radio=0.58, mostrar_texto=None, dpi=170):
    R, C = P.R, P.C
    if mostrar_texto is None:
        mostrar_texto = P.nV <= 130

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
            hexagon = RegularPolygon(
                (x, y), numVertices=6, radius=radio, orientation=np.pi / 6,
                facecolor=PALETA[SP[especie]],
                edgecolor=FONDO, linewidth=1.1, zorder=2)
            ax.add_patch(hexagon)
            if fijo:
                # marca elegante de preexistente: punto pequeno, no borde grueso
                ax.add_patch(Circle((x, y), radio * 0.22, facecolor=FONDO,
                                     edgecolor="none", zorder=3, alpha=0.85))
            if mostrar_texto:
                ax.text(x, y - radio * 0.62, SP[especie], ha='center', va='top',
                         fontsize=6.2, color=COLOR_TEXTO, alpha=0.75, zorder=4,
                         family='monospace')

    ax.set_xlim(-1, C + 1)
    ax.set_ylim(-R * np.sqrt(3) / 2 - 1.4, 1.4)
    ax.set_aspect('equal')
    ax.axis('off')

    f1, f2 = P.f1(s), P.f2(s)
    n_fijos = int((esc >= 0).sum())

    ax.text(0.5, 1.10, subtitulo or f"Asignación de especies — retículo {R}×{C}",
             transform=ax.transAxes, ha='center', va='bottom',
             fontsize=15, color=COLOR_TEXTO, weight='bold', family='sans-serif')
    stats = (f"$n_V$ = {P.nV}     $f_1$ (competencia) = {f1:.2f}     "
              f"$f_2$ (monocultivo) = {f2}     "
              f"{n_fijos} sitios preexistentes (◦)")
    ax.text(0.5, 1.045, stats, transform=ax.transAxes, ha='center', va='bottom',
             fontsize=9.5, color="#6B6459", family='sans-serif')

    # leyenda: hexagonos pequeños con nombre comun, no solo abreviatura
    handles = [Patch(facecolor=PALETA[sp], edgecolor=FONDO,
                       label=f"{sp}") for sp in SP]
    leg = fig.legend(handles=handles, loc='lower center', ncol=10, fontsize=7.5,
                       frameon=False, bbox_to_anchor=(0.5, -0.01),
                       handlelength=1.3, handleheight=1.3, columnspacing=1.2)
    for text in leg.get_texts():
        text.set_color(COLOR_TEXTO)

    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    plt.savefig(archivo_salida, dpi=dpi, bbox_inches='tight', facecolor=FONDO)
    plt.close(fig)
    return f1, f2, n_fijos


def resolver_y_graficar_bello(R, C, seed=0, iters=25000, subtitulo=None, archivo=None):
    P = Problema(R, C, t=0.10)
    esc = P.escenario(seed=seed)
    s0 = P.sembrar(esc, rng=np.random.default_rng(seed))
    s1 = P.busqueda_local(s0, w1=1.0, w2=1.0, iters=iters,
                            rng=np.random.default_rng(seed))
    archivo = archivo or f"mapa_bello_{R}x{C}.png"
    f1, f2, n_fijos = graficar_bello(P, s1, esc, subtitulo, archivo)
    print(f"{R}x{C}: nV={P.nV}  f1={f1:.2f}  f2={f2}  preexistentes={n_fijos} -> {archivo}")
    return P, s1, esc


if __name__ == "__main__":
    resolver_y_graficar_bello(8, 8, seed=0,
        subtitulo="Solución de compromiso — instancia 8×8 (validada contra MILP)")
    resolver_y_graficar_bello(14, 47, seed=0,
        subtitulo="Escala real — Línea de Transmisión Dominica–Charcas (1 ha)")
