import sys, os, time, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from core import Problema
from nsga2_reforestacion import correr_nsga2

P = Problema(14, 47, t=0.10)
esc = P.escenario(seed=0)

t0 = time.perf_counter()
res, hv_hist, gen_conv = correr_nsga2(P, esc, pop=120, gens=200, seed=0, ref_point=(1e9, 1e9))
dt = time.perf_counter() - t0
F, X = res.F, res.X.astype(int)
idx = int(np.argmin(F[:, 0]))
s = X[idx]
print(f"14x47: f1={F[idx,0]:.2f} f2={int(F[idx,1])}  [{dt:.1f}s]")

with open("resultados/solucion_14x47.pkl", "wb") as fh:
    pickle.dump({"P": P, "esc": esc, "s": s, "f1": float(F[idx,0]), "f2": int(F[idx,1])}, fh)
print("guardado en resultados/solucion_14x47.pkl")
