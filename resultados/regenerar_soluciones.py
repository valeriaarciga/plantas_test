"""Reconstruye los vectores de solucion (s) exactos usados en comparar_hv_todos.py
para poder visualizarlos con visualizar_bello.py. Mismo escenario/semilla/params
que la comparacion de HV ya reportada; el MILP se re-resuelve solo en el punto
eps objetivo (no el barrido completo) para ahorrar tiempo.

NOTA DE HONESTIDAD: los operadores de nsga2_reforestacion.py (SiembraColoracion,
MutacionIntercambio, CruceBolaHexagonal) usan np.random.default_rng() SIN semilla
dentro de _do() -> el parametro seed= de minimize() no los gobierna, asi que
NSGA-II no es bit-a-bit reproducible entre corridas aunque se pase el mismo seed.
Se re-corre con los mismos pop/gens/seed/escenario que comparar_hv_todos.py y se
toma, del frente resultante, el punto con f2 mas cercano al reportado en la tabla
-- los valores de f1 pueden diferir en la 2da-3ra cifra decimal respecto a la
tabla ya publicada, no es una corrida nueva con otra intencion.
"""
import sys, os, time, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from core import Problema
from milp_gurobi import frente_epsilon
from nsga2_reforestacion import correr_nsga2

TAMANOS = {
    "6x6":   dict(R=6, C=6,  eps_milp=1, time_limit=60,  mip_gap=0.001),
    "8x8":   dict(R=8, C=8,  eps_milp=1, time_limit=180, mip_gap=0.02),
    "10x10": dict(R=10, C=10, eps_milp=1, time_limit=180, mip_gap=0.03),
    "12x12": dict(R=12, C=12, eps_milp=0, time_limit=180, mip_gap=0.03),
}

resultados = {}
for nombre, cfg in TAMANOS.items():
    print(f"\n{'='*60}\n{nombre}\n{'='*60}")
    P = Problema(cfg["R"], cfg["C"], t=0.10)
    esc = P.escenario(seed=0)

    # --- MILP: un solo punto eps ---
    t0 = time.perf_counter()
    frente = frente_epsilon(P, esc, epsilons=[cfg["eps_milp"]],
                             time_limit=cfg["time_limit"], mip_gap=cfg["mip_gap"])
    dt = time.perf_counter() - t0
    r = frente[0]
    s_milp = r["s"]
    print(f"MILP  eps={cfg['eps_milp']}: f1={r['f1']:.4f} f2={r['f2']} "
          f"status={r['status']} gap={r['mip_gap']*100:.3f}%  [{dt:.1f}s]")

    # --- NSGA-II: mismo escenario/semilla/pop/gens que comparar_hv_todos.py ---
    t0 = time.perf_counter()
    res, hv_hist, gen_conv = correr_nsga2(P, esc, pop=120, gens=200, seed=0,
                                           ref_point=(1e9, 1e9))
    dt2 = time.perf_counter() - t0
    F, X = res.F, res.X.astype(int)
    idx_min_f2 = int(np.argmin(F[:, 1]))
    f2_min_nsga = int(F[idx_min_f2, 1])
    # si el f2 minimo del frente no es el objetivo, se usa igual el minimo
    # disponible (es precisamente el punto que ilustra el gap, ej. 12x12)
    s_nsga = X[idx_min_f2]
    print(f"NSGA-II: f2_min_en_frente={f2_min_nsga}  f1={F[idx_min_f2,0]:.4f}  [{dt2:.1f}s]")

    resultados[nombre] = {
        "P": P, "esc": esc,
        "s_milp": s_milp, "f1_milp": r["f1"], "f2_milp": r["f2"],
        "s_nsga": s_nsga, "f1_nsga": float(F[idx_min_f2, 0]), "f2_nsga": f2_min_nsga,
    }

with open("resultados/soluciones_mapas.pkl", "wb") as fh:
    pickle.dump(resultados, fh)
print("\nguardado en resultados/soluciones_mapas.pkl")
