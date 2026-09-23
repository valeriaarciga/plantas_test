"""Tarea 4: tiempo/HV de NSGA-II (ya corregido) a traves de la escalera de
tamanos, incluyendo 14x47 (donde el MILP ya no aplica por diseno)."""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from core import Problema
from nsga2_reforestacion import correr_nsga2
from pymoo.indicators.hv import HV

TAMANOS = [(6, 6), (8, 8), (10, 10), (12, 12), (14, 47)]
POP, GENS, SEED = 120, 200, 0

resultados = []
for R, C in TAMANOS:
    P = Problema(R, C, t=0.10)
    esc = P.escenario(seed=SEED)

    rng = np.random.default_rng(0)
    peor = [(P.f1(P.sembrar(esc, rng=rng, perturbar=200)),
             P.f2(P.sembrar(esc, rng=rng, perturbar=200))) for _ in range(10)]
    peor = np.array(peor)
    ref = (float(peor[:, 0].max()) * 1.2, float(peor[:, 1].max()) * 1.2 + 1)

    t0 = time.perf_counter()
    res, hv_hist, gen_conv = correr_nsga2(P, esc, pop=POP, gens=GENS, seed=SEED, ref_point=ref)
    dt = time.perf_counter() - t0

    fila = {
        "R": R, "C": C, "nV": P.nV, "nA": P.nA,
        "nsga2_tiempo_s": round(dt, 2), "nsga2_gen_convergencia": gen_conv,
        "nsga2_hv": round(hv_hist[-1], 4), "ref_point": ref,
        "nsga2_n_soluciones": len(res.F),
        "nsga2_f1_min": round(float(res.F[:, 0].min()), 4),
        "nsga2_f2_en_f1min": int(res.F[np.argmin(res.F[:, 0]), 1]),
    }
    resultados.append(fila)
    print(f"{R:2d}x{C:2d}  nV={P.nV:4d} nA={P.nA:5d}  tiempo={dt:6.2f}s  "
          f"gen_conv={gen_conv:3d}/{GENS}  HV={hv_hist[-1]:.2f}  "
          f"n_frente={len(res.F)}  f1_min={fila['nsga2_f1_min']:.2f} (f2={fila['nsga2_f2_en_f1min']})")

with open("resultados/nsga2_timing.json", "w") as fh:
    json.dump(resultados, fh, indent=2)
print("\nguardado en resultados/nsga2_timing.json")
