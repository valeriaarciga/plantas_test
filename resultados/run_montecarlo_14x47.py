"""Tarea 5 - Pipeline de Montecarlo completo a escala real (14x47).
Genera |Omega|=30 escenarios y corre NSGA-II (ya corregido) sobre cada uno,
reportando la distribucion de (f1*, f2*) y validando contra el piso teorico
E[f2_min] ~= 10.66 calculado analiticamente en core.py.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from core import Problema, P_OCU, PI
from nsga2_reforestacion import correr_nsga2

P = Problema(14, 47, t=0.10)
PISO_TEORICO = P.nA * P_OCU**2 * (PI**2).sum()
print(f"Instancia 14x47: nV={P.nV} nA={P.nA}")
print(f"Piso teorico E[f2_min] = {PISO_TEORICO:.4f}\n")

N_ESCENARIOS = 30
resultados = []
t0 = time.perf_counter()
for k in range(N_ESCENARIOS):
    esc = P.escenario(seed=k)
    piso_k = int(((esc[P.U] >= 0) & (esc[P.U] == esc[P.V])).sum())
    tk0 = time.perf_counter()
    res, hv_hist, gen_conv = correr_nsga2(P, esc, pop=100, gens=120, seed=k,
                                           ref_point=(1500.0, 500.0))
    dtk = time.perf_counter() - tk0
    f1_min = float(res.F[:, 0].min())
    f2_en_f1min = int(res.F[np.argmin(res.F[:, 0]), 1])
    f2_min_obs = int(res.F[:, 1].min())
    viola_piso = f2_min_obs < piso_k
    resultados.append({
        "escenario": k, "piso_teorico_escenario": piso_k,
        "f1_min": f1_min, "f2_en_f1min": f2_en_f1min, "f2_min_observado": f2_min_obs,
        "viola_piso": viola_piso, "tiempo_s": dtk, "gen_convergencia": gen_conv,
        "n_soluciones_frente": len(res.F),
    })
    print(f"escenario {k:2d}: piso={piso_k:3d}  f1_min={f1_min:8.2f} (f2={f2_en_f1min:3d})  "
          f"f2_min_obs={f2_min_obs:3d}  viola_piso={viola_piso}  t={dtk:.1f}s")

dt_total = time.perf_counter() - t0
f1s = np.array([r["f1_min"] for r in resultados])
f2s = np.array([r["f2_en_f1min"] for r in resultados])
violaciones = sum(r["viola_piso"] for r in resultados)

print(f"\n{'='*60}")
print(f"Tiempo total: {dt_total:.1f}s ({dt_total/N_ESCENARIOS:.1f}s/escenario)")
print(f"f1_min: media={f1s.mean():.2f}  std={f1s.std():.2f}  p5={np.percentile(f1s,5):.2f}  p95={np.percentile(f1s,95):.2f}")
print(f"f2 (en f1_min): media={f2s.mean():.2f}  std={f2s.std():.2f}  p5={np.percentile(f2s,5):.2f}  p95={np.percentile(f2s,95):.2f}")
print(f"Piso teorico E[f2_min] = {PISO_TEORICO:.4f}")
print(f"Violaciones del piso teorico: {violaciones}/{N_ESCENARIOS} (debe ser 0)")

with open("resultados/montecarlo_14x47.json", "w") as fh:
    json.dump({
        "piso_teorico": PISO_TEORICO, "n_escenarios": N_ESCENARIOS,
        "resultados": resultados,
        "resumen": {
            "f1_media": float(f1s.mean()), "f1_std": float(f1s.std()),
            "f1_p5": float(np.percentile(f1s,5)), "f1_p95": float(np.percentile(f1s,95)),
            "f2_media": float(f2s.mean()), "f2_std": float(f2s.std()),
            "f2_p5": float(np.percentile(f2s,5)), "f2_p95": float(np.percentile(f2s,95)),
            "violaciones_piso": int(violaciones),
        }
    }, fh, indent=2)
print("\nguardado en resultados/montecarlo_14x47.json")
