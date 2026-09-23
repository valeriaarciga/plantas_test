"""
portafolio.py — Escalar el ejercicio a los 30 poligonos reales del sitio
(en vez de un solo hectarea sintetica). Reutiliza core.py sin modificarlo:
cada poligono es una instancia independiente (no hay competencia ENTRE
poligonos, solo dentro de cada uno), asi que el problema se descompone
exactamente y se resuelve poligono por poligono.

Dos modos de escenario:
  - MONTECARLO: usa P.escenario(seed) tal como esta (probabilidades globales).
  - REAL: usa los conteos observados de campo de ESE poligono especifico
          (datos_inecol / Calculos_reto), asignados a nodos al azar dentro
          del poligono (no se conoce la ubicacion exacta de cada planta,
          solo el conteo total por especie).

Uso:
    from portafolio import correr_portafolio
    resumen = correr_portafolio(modo="real")
"""
import numpy as np
from core import Problema, SP, R_HA, N_HA, cuotas, compra_minima

# ---------------------------------------------------------------------
# Datos reales de los 30 poligonos (datos_inecol / Calculos_reto, ya
# validados en el EDA). area_ha y conteos por especie, en el mismo orden
# que SP = ["AL","AS","ASc","ASt","OC","OE","OR","OS","PL","YF"].
# ---------------------------------------------------------------------
AREAS_HA = [1.28,6.64,6.76,1.38,8,7.82,5.53,5.64,7.11,6.11,5.64,4.92,5.05,4.75,
            7.97,7.34,5.98,5.4,6.28,7.6,8,8,7.67,1.47,4.19,7.52,8,8,7.56,5.4]

CONTEOS = {
"AL": [8,58,66,10,65,67,41,32,58,47,48,36,50,35,40,70,56,45,56,69,54,75,44,10,44,72,74,67,60,34],
"AS": [46,263,236,52,280,306,209,252,269,209,233,182,189,186,311,290,233,204,223,285,287,292,288,59,155,291,342,309,280,195],
"ASc":[16,47,51,15,66,63,43,46,58,47,41,43,49,38,52,55,39,34,51,57,70,60,69,15,28,73,52,67,61,56],
"ASt":[16,49,50,9,61,61,49,40,53,41,37,44,28,43,62,54,46,33,48,75,62,68,51,11,31,50,55,66,53,58],
"OC": [11,60,71,15,92,81,48,73,57,66,48,51,43,39,68,84,71,56,50,77,89,104,69,17,39,66,70,86,76,55],
"OE": [14,44,56,10,54,62,52,39,58,43,40,41,33,44,63,50,45,46,48,74,58,62,60,18,39,62,76,54,68,46],
"OR": [18,111,100,18,124,118,87,86,94,96,94,68,73,73,132,105,83,82,93,107,126,109,122,17,65,125,100,120,109,91],
"OS": [12,95,78,20,114,91,60,79,91,71,65,72,76,61,89,77,81,48,90,103,109,95,100,10,50,101,100,103,85,63],
"PL": [15,98,123,23,106,133,97,91,117,108,94,67,97,74,149,121,97,95,120,104,139,125,106,39,67,122,129,141,135,86],
"YF": [9,39,28,11,41,45,27,26,37,29,27,19,25,23,53,40,32,17,33,25,44,43,39,8,25,31,44,33,36,23],
}
N_POLIGONOS = 30


def dimensiones_grid(area_ha, aspecto_ref=14 / 47):
    """Aproxima (R, C) tal que R*C ~= area_ha*658, manteniendo un aspecto
       razonable. LIMITACION DECLARADA: no se conoce la geometria real del
       poligono (solo el area), asi que se asume una forma rectangular
       generica -- razonable porque la competencia solo depende de la
       vecindad local, no de la forma global del borde."""
    nV_obj = round(area_ha * N_HA)
    C = max(2, round(np.sqrt(nV_obj / aspecto_ref)))
    R = max(2, round(nV_obj / C))
    return R, C


def escenario_real(P, conteos_poligono, rng):
    """Construye el escenario de vegetacion preexistente a partir de los
       CONTEOS REALES observados en ese poligono especifico (no una
       simulacion). Como no se conoce la ubicacion exacta de cada planta
       dentro del poligono, se distribuyen al azar sobre los nV nodos --
       el conteo total por especie SI es el dato real, solo la posicion
       espacial es una suposicion necesaria."""
    conteos_poligono = np.asarray(conteos_poligono, dtype=int)
    total_real = conteos_poligono.sum()
    if total_real > P.nV:
        # el poligono real es mas denso que la capacidad nominal del grid
        # aproximado (nV redondeado); se escala proporcionalmente
        conteos_poligono = np.floor(conteos_poligono / total_real * P.nV).astype(int)
        total_real = conteos_poligono.sum()
    especies = np.repeat(np.arange(10), conteos_poligono)
    nodos = rng.choice(P.nV, size=total_real, replace=False)
    esc = np.full(P.nV, -1, dtype=int)
    esc[nodos] = especies
    return esc


def correr_portafolio(modo="real", n_poligonos=N_POLIGONOS, iters_bl=15000, seed=0):
    """Resuelve cada poligono de forma independiente (sin MILP: a esta
       escala -842 a 5264 nodos por poligono- el MILP ya es intratable
       incluso en el mas chico; ver seccion de discusion). Agrega
       resultados a nivel de PORTAFOLIO (los 30 poligonos juntos)."""
    rng = np.random.default_rng(seed)
    filas = []
    for k in range(n_poligonos):
        area = AREAS_HA[k]
        R, C = dimensiones_grid(area)
        P = Problema(R, C, t=0.10)

        if modo == "real":
            conteos_k = [CONTEOS[sp][k] for sp in SP]
            esc = escenario_real(P, conteos_k, rng)
        else:
            esc = P.escenario(seed=seed + k)

        s0 = P.sembrar(esc, rng=rng)
        s1 = P.busqueda_local(s0, w1=1.0, w2=1.0, iters=iters_bl, rng=rng)

        piso = int(((esc[P.U] >= 0) & (esc[P.U] == esc[P.V])).sum())

        filas.append({
            "poligono": k + 1, "area_ha": area, "R": R, "C": C, "nV": P.nV,
            "modo_escenario": modo,
            "nodos_preexistentes": int((esc >= 0).sum()),
            "piso_f2": piso,
            "f1_sembrado": round(P.f1(s0), 2), "f2_sembrado": P.f2(s0),
            "f1_final": round(P.f1(s1), 2), "f2_final": P.f2(s1),
            "cuota_ok": P.cuota_ok(s1),
        })

    return filas


def resumen_portafolio(filas):
    """Metricas a nivel de PORTAFOLIO: lo que le importa al socio formador
       no es un poligono aislado, es el sitio completo."""
    nV_tot = sum(f["nV"] for f in filas)
    f1_tot = sum(f["f1_final"] for f in filas)
    f2_tot = sum(f["f2_final"] for f in filas)
    piso_tot = sum(f["piso_f2"] for f in filas)
    cuotas_ok = all(f["cuota_ok"] for f in filas)
    print(f"\n{'='*72}\nRESUMEN DE PORTAFOLIO ({len(filas)} poligonos)\n{'='*72}")
    print(f"Nodos totales (aprox.):      {nV_tot:6d}   (equivalente a {nV_tot/N_HA:.2f} ha)")
    print(f"Competencia total f1:        {f1_tot:10.1f}")
    print(f"Monocultivo total f2:        {f2_tot:6d}   (piso teorico agregado: {piso_tot})")
    print(f"Cuota respetada en todos:    {cuotas_ok}")
    print(f"Competencia promedio/nodo:   {f1_tot/nV_tot:.4f}")

    # cuota agregada de portafolio: R_i escalado al total real de plantas
    # (no 658 por poligono, sino la suma real de todo el sitio)
    Ri_total = np.round(R_HA / N_HA * nV_tot).astype(int)
    compra = compra_minima(alpha=0.20, R=Ri_total)
    print(f"\nCuota agregada de portafolio (R_i escalada a {nV_tot} nodos):")
    for i, sp in enumerate(SP):
        print(f"  {sp:5s}: cuota={Ri_total[i]:5d}  compra recomendada (a=0.20)={compra[i]:5d}")
    print(f"  TOTAL: cuota={Ri_total.sum():5d}  compra={compra.sum():5d}  "
          f"sobrecompra={compra.sum()/Ri_total.sum()-1:.1%}")


if __name__ == "__main__":
    import time

    print("Dimensiones de grid aproximadas por poligono (primeros 5):")
    for k in range(5):
        R, C = dimensiones_grid(AREAS_HA[k])
        print(f"  Poligono {k+1}: area={AREAS_HA[k]:.2f} ha -> grid {R}x{C} "
              f"(nV={R*C}, objetivo={round(AREAS_HA[k]*N_HA)})")

    print(f"\n{'='*72}\nCorrida de DEMOSTRACION en 3 poligonos "
          f"(la corrida completa de 30 es analoga, solo mas larga)\n{'='*72}")
    t0 = time.perf_counter()
    filas = correr_portafolio(modo="real", n_poligonos=3, iters_bl=15000, seed=0)
    dt = time.perf_counter() - t0
    for f in filas:
        print(f"\nPoligono {f['poligono']} (area={f['area_ha']} ha, grid {f['R']}x{f['C']}, "
              f"nV={f['nV']}, {f['nodos_preexistentes']} preexistentes reales):")
        print(f"  piso_f2 teorico = {f['piso_f2']}")
        print(f"  sembrado        : f1={f['f1_sembrado']:8.1f}  f2={f['f2_sembrado']:4d}")
        print(f"  + busqueda local: f1={f['f1_final']:8.1f}  f2={f['f2_final']:4d}  "
              f"cuota_ok={f['cuota_ok']}")
    print(f"\nTiempo total para 3 poligonos: {dt:.1f}s "
          f"(proyeccion para 30: ~{dt/3*30:.0f}s, trivialmente paralelizable)")

    resumen_portafolio(filas)
