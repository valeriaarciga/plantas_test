import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from core import Problema
from milp_gurobi import frente_epsilon
from nsga2_reforestacion import correr_nsga2
from pymoo.indicators.hv import HV

P = Problema(6, 6, t=0.10)
esc = P.escenario(seed=0)

# --- MILP: frente exacto ya conocido (eps=1..5), lo recomputamos rapido para timing limpio ---
t0 = time.perf_counter()
frente_milp = frente_epsilon(P, esc, epsilons=range(0, 6), time_limit=60, mip_gap=0.001)
dt_milp = time.perf_counter() - t0
puntos_milp = np.array([[r["f1"], r["f2"]] for r in frente_milp if r["f1"] is not None])
puntos_milp = np.unique(puntos_milp, axis=0)

# --- NSGA-II: misma instancia/escenario/semilla, ya corregido (eliminate_duplicates=True) ---
t0 = time.perf_counter()
res, hv_hist, gen_conv = correr_nsga2(P, esc, pop=80, gens=100, seed=0, ref_point=(200.0, 40.0))
dt_nsga2 = time.perf_counter() - t0
puntos_nsga2 = np.unique(np.round(res.F, 6), axis=0)

# --- punto de referencia comun para HV: peor caso observado + margen ---
todos = np.vstack([puntos_milp, puntos_nsga2])
ref = (float(todos[:,0].max()) * 1.05, float(todos[:,1].max()) * 1.3 + 1)
hv_calc = HV(ref_point=np.array(ref))
hv_milp = hv_calc(puntos_milp)
hv_nsga2 = hv_calc(puntos_nsga2)

print(f"Instancia 6x6 (nV={P.nV}, nA={P.nA}), escenario seed=0, ref_point={ref}\n")
print(f"MILP exacto:  {len(puntos_milp)} puntos  tiempo={dt_milp:.1f}s  HV={hv_milp:.4f}")
for f1, f2 in puntos_milp[np.argsort(puntos_milp[:,1])]:
    print(f"    f1={f1:8.4f}  f2={f2:.0f}")
print(f"\nNSGA-II:      {len(puntos_nsga2)} puntos  tiempo={dt_nsga2:.1f}s  HV={hv_nsga2:.4f}")
for f1, f2 in puntos_nsga2[np.argsort(puntos_nsga2[:,1])]:
    print(f"    f1={f1:8.4f}  f2={f2:.0f}")

razon = hv_nsga2 / hv_milp
print(f"\nRazon HV(NSGA-II)/HV(MILP) = {razon:.4%}")
print(f"Tiempo MILP / Tiempo NSGA-II = {dt_milp/dt_nsga2:.1f}x")

resumen = {
    "instancia": "6x6", "nV": P.nV, "nA": P.nA, "ref_point": ref,
    "milp": {"n_puntos": len(puntos_milp), "tiempo_s": dt_milp, "hv": hv_milp,
             "puntos": puntos_milp.tolist()},
    "nsga2": {"n_puntos": len(puntos_nsga2), "tiempo_s": dt_nsga2, "hv": hv_nsga2,
              "puntos": puntos_nsga2.tolist()},
    "razon_hv": razon,
}
with open("resultados/comparacion_hv_6x6.json", "w") as fh:
    json.dump(resumen, fh, indent=2)
