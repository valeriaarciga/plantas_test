"""Tarea 3, version corregida: compara MILP vs NSGA-II con un punto de
referencia FIJO e INDEPENDIENTE de los resultados de ambos algoritmos
(una asignacion deliberadamente mala x margen, como en escalabilidad.py).
Usar el maximo de los propios resultados como referencia distorsiona el HV
cuando un algoritmo tiene peor rango en f1 (esto se detecto en el primer
intento con 10x10: HV cayo a 49% por un artefacto del punto de referencia,
no por una diferencia real de esa magnitud en la calidad de las soluciones).
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from core import Problema
from nsga2_reforestacion import correr_nsga2
from pymoo.indicators.hv import HV


def punto_referencia_valido(P, esc, factor=1.2, n=20):
    rng = np.random.default_rng(0)
    peor = []
    for _ in range(n):
        s = P.sembrar(esc, rng=rng, perturbar=300)
        peor.append((P.f1(s), P.f2(s)))
    peor = np.array(peor)
    return float(peor[:, 0].max() * factor), float(peor[:, 1].max() * factor)


def comparar(tam, milp_json, pop=120, gens=200, seed=0):
    R, C = tam
    P = Problema(R, C, t=0.10)
    esc = P.escenario(seed=seed)

    with open(milp_json) as fh:
        frente_milp_raw = json.load(fh)
    puntos_milp = np.array([[r["f1"], r["f2"]] for r in frente_milp_raw if r["f1"] is not None])
    puntos_milp = np.unique(puntos_milp, axis=0)
    tiempo_milp = sum(r["runtime_s"] for r in frente_milp_raw)
    todos_cerraron = all(r["cerro_optimalidad"] for r in frente_milp_raw if r["f1"] is not None)

    t0 = time.perf_counter()
    res, hv_hist, gen_conv = correr_nsga2(P, esc, pop=pop, gens=gens, seed=seed,
                                           ref_point=(1e9, 1e9))  # no se usa; HV real abajo
    dt_nsga2 = time.perf_counter() - t0
    puntos_nsga2 = np.unique(np.round(res.F, 6), axis=0)

    ref = punto_referencia_valido(P, esc)
    # el ref debe dominar TODO punto observado (MILP y NSGA-II); si algun f1/f2
    # supera el ref por construccion aleatoria, se amplia con margen adicional
    todos = np.vstack([puntos_milp, puntos_nsga2])
    ref = (max(ref[0], float(todos[:, 0].max()) * 1.05),
           max(ref[1], float(todos[:, 1].max()) * 1.05))

    hv_calc = HV(ref_point=np.array(ref))
    hv_milp = hv_calc(puntos_milp)
    hv_nsga2 = hv_calc(puntos_nsga2)
    razon = hv_nsga2 / hv_milp

    print(f"\n{'='*70}\nInstancia {R}x{C} (nV={P.nV}, nA={P.nA})  ref_point={ref}")
    print(f"MILP: todos los puntos cerraron a optimalidad = {todos_cerraron}")
    print(f"MILP:    {len(puntos_milp)} puntos  tiempo_total={tiempo_milp:8.1f}s  HV={hv_milp:10.4f}")
    for f1, f2 in puntos_milp[np.argsort(puntos_milp[:, 1])]:
        print(f"    f1={f1:9.4f}  f2={f2:.0f}")
    print(f"NSGA-II: {len(puntos_nsga2)} puntos  tiempo={dt_nsga2:8.1f}s  HV={hv_nsga2:10.4f}")
    for f1, f2 in puntos_nsga2[np.argsort(puntos_nsga2[:, 1])]:
        print(f"    f1={f1:9.4f}  f2={f2:.0f}")
    print(f"Razon HV(NSGA-II)/HV(MILP) = {razon:.4%}")
    print(f"Tiempo MILP / Tiempo NSGA-II = {tiempo_milp/dt_nsga2:.1f}x")

    # gap relativo de f1 al f2 minimo comun (metrica interpretable, no distorsionable
    # por la eleccion del punto de referencia)
    f2_min_comun = int(max(puntos_milp[:, 1].min(), puntos_nsga2[:, 1].min()))
    f1_milp_en_f2min = puntos_milp[puntos_milp[:, 1] == puntos_milp[:, 1].min()][0, 0]
    f1_nsga2_en_f2min = puntos_nsga2[puntos_nsga2[:, 1] == puntos_nsga2[:, 1].min()][0, 0]
    gap_f1_pct = 100 * (f1_nsga2_en_f2min - f1_milp_en_f2min) / f1_milp_en_f2min
    print(f"Gap relativo en f1 (al f2 minimo de cada frente): {gap_f1_pct:+.2f}%")

    return {
        "instancia": f"{R}x{C}", "nV": P.nV, "nA": P.nA, "ref_point": ref,
        "milp": {"n_puntos": len(puntos_milp), "tiempo_s": tiempo_milp, "hv": hv_milp,
                 "todos_cerraron_optimalidad": todos_cerraron, "puntos": puntos_milp.tolist()},
        "nsga2": {"n_puntos": len(puntos_nsga2), "tiempo_s": dt_nsga2, "hv": hv_nsga2,
                  "puntos": puntos_nsga2.tolist()},
        "razon_hv": razon, "gap_f1_pct_en_f2_min": gap_f1_pct,
    }


if __name__ == "__main__":
    resumen = {}
    resumen["6x6"] = comparar((6, 6), "resultados/milp_6x6.json") if os.path.exists("resultados/milp_6x6.json") else None
    resumen["8x8"] = comparar((8, 8), "resultados/milp_8x8.json")
    resumen["10x10"] = comparar((10, 10), "resultados/milp_10x10.json")
    resumen["12x12"] = comparar((12, 12), "resultados/milp_12x12.json")
    with open("resultados/comparacion_hv_todos.json", "w") as fh:
        json.dump(resumen, fh, indent=2)
