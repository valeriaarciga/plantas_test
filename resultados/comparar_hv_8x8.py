import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from core import Problema
from nsga2_reforestacion import correr_nsga2
from pymoo.indicators.hv import HV

P = Problema(8, 8, t=0.10)
esc = P.escenario(seed=0)

with open("resultados/milp_8x8.json") as fh:
    frente_milp_raw = json.load(fh)
puntos_milp = np.array([[r["f1"], r["f2"]] for r in frente_milp_raw if r["f1"] is not None])
puntos_milp = np.unique(puntos_milp, axis=0)
tiempo_milp = sum(r["runtime_s"] for r in frente_milp_raw)
todos_cerraron = all(r["cerro_optimalidad"] for r in frente_milp_raw if r["f1"] is not None)

t0 = time.perf_counter()
res, hv_hist, gen_conv = correr_nsga2(P, esc, pop=120, gens=200, seed=0, ref_point=(200.0, 60.0))
dt_nsga2 = time.perf_counter() - t0
puntos_nsga2 = np.unique(np.round(res.F, 6), axis=0)

todos = np.vstack([puntos_milp, puntos_nsga2])
ref = (float(todos[:,0].max()) * 1.05, float(todos[:,1].max()) * 1.3 + 1)
hv_calc = HV(ref_point=np.array(ref))
hv_milp = hv_calc(puntos_milp)
hv_nsga2 = hv_calc(puntos_nsga2)

print(f"Instancia 8x8 (nV={P.nV}, nA={P.nA}), escenario seed=0, ref_point={ref}")
print(f"MILP: todos los puntos cerraron a optimalidad = {todos_cerraron}\n")
print(f"MILP (barrido eps, gaps reales por punto):  {len(puntos_milp)} puntos  tiempo_total={tiempo_milp:.1f}s  HV={hv_milp:.4f}")
for f1, f2 in puntos_milp[np.argsort(puntos_milp[:,1])]:
    print(f"    f1={f1:8.4f}  f2={f2:.0f}")
print(f"\nNSGA-II:  {len(puntos_nsga2)} puntos  tiempo={dt_nsga2:.1f}s  HV={hv_nsga2:.4f}")
for f1, f2 in puntos_nsga2[np.argsort(puntos_nsga2[:,1])]:
    print(f"    f1={f1:8.4f}  f2={f2:.0f}")

razon = hv_nsga2 / hv_milp
print(f"\nRazon HV(NSGA-II)/HV(MILP) = {razon:.4%}")
print(f"Tiempo MILP / Tiempo NSGA-II = {tiempo_milp/dt_nsga2:.1f}x")

resumen = {
    "instancia": "8x8", "nV": P.nV, "nA": P.nA, "ref_point": ref,
    "milp": {"n_puntos": len(puntos_milp), "tiempo_s": tiempo_milp, "hv": hv_milp,
             "todos_cerraron_optimalidad": todos_cerraron, "puntos": puntos_milp.tolist()},
    "nsga2": {"n_puntos": len(puntos_nsga2), "tiempo_s": dt_nsga2, "hv": hv_nsga2,
              "puntos": puntos_nsga2.tolist()},
    "razon_hv": razon,
}
with open("resultados/comparacion_hv_8x8.json", "w") as fh:
    json.dump(resumen, fh, indent=2)
