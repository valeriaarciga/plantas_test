import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import Problema
from milp_gurobi import frente_epsilon

P = Problema(6, 6, t=0.10)
esc = P.escenario(seed=0)
print(f"Instancia 6x6: nV={P.nV} nA={P.nA} cuota={list(P.quota)}", file=sys.stderr)

EPS = [0, 1, 2, 3, 4, 5]
t0 = time.perf_counter()
frente = frente_epsilon(P, esc, epsilons=EPS, time_limit=60, mip_gap=0.001)
dt = time.perf_counter() - t0
print(f"\nTiempo total barrido 6x6: {dt:.1f}s", file=sys.stderr)

out = []
for r in frente:
    row = {k: r[k] for k in ("epsilon","f1","f2","mip_gap","cota_dual","runtime_s","status","nodos_explorados","cerro_optimalidad")}
    out.append(row)
    if r["f1"] is not None:
        print(f"eps={r['epsilon']:2d}  f1*={r['f1']:8.4f}  f2={r['f2']:2d}  gap={r['mip_gap']*100:6.3f}%  status={r['status']:10s} t={r['runtime_s']:.2f}s", file=sys.stderr)
    else:
        print(f"eps={r['epsilon']:2d}  {r['status']} (sin incumbente) t={r['runtime_s']:.2f}s", file=sys.stderr)

with open("resultados/milp_6x6.json", "w") as fh:
    json.dump(out, fh, indent=2)
print("guardado en resultados/milp_6x6.json", file=sys.stderr)
