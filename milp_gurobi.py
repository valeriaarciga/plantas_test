"""
milp_gurobi.py — Bloque 3: modelo MILP exacto (Gurobi) para el problema de
reforestacion, con barrido de epsilon-restriccion para el frente de Pareto exacto.

Requiere: gurobipy con licencia (para 6x6/8x8 hace falta licencia completa;
la size-limited/trial de Gurobi tiene tope de ~2000 variables, y estas
instancias ya lo superan).

Uso:
    from core import Problema
    from milp_gurobi import resolver_milp, frente_epsilon

    P = Problema(6, 6)
    esc = P.escenario(seed=0)
    x_opt, f1, f2 = resolver_milp(P, esc)                 # un solo punto
    frente = frente_epsilon(P, esc, epsilons=range(0, 15)) # frente completo
"""
import gurobipy as gp
from gurobipy import GRB
import numpy as np


def construir_modelo(P, escenario, time_limit=600, mip_gap=0.01, threads=None):
    """Construye el modelo UNA sola vez (Paso 1-4). Se reutiliza en el
    barrido de epsilon cambiando solo el RHS de la restriccion f2<=eps,
    lo que permite reoptimizar en caliente (warm start) en vez de
    reconstruir el modelo en cada iteracion."""
    nV, nS = P.nV, 10
    U, V = P.U, P.V
    nA = P.nA

    m = gp.Model("reforestacion")
    if threads:
        m.Params.Threads = threads
    m.Params.TimeLimit = time_limit
    m.Params.MIPGap = mip_gap
    m.Params.MIPFocus = 1          # prioriza encontrar buenas soluciones factibles rapido

    # ---- Paso 2: variables ----
    # x[v,i]: binaria, especie i en el nodo v
    x = m.addVars(nV, nS, vtype=GRB.BINARY, name="x")
    # y[a,i,j]: CONTINUA (Frieze-Yadegar no requiere integralidad) -> resuelve mucho mas rapido
    y = m.addVars(nA, nS, nS, vtype=GRB.CONTINUOUS, lb=0.0, name="y")

    # ---- Paso 3: funcion objetivo (se fija por separado en cada llamada) ----
    Ct = P.Ct
    f1_expr = gp.quicksum(Ct[i, j] * y[a, i, j]
                          for a in range(nA) for i in range(nS) for j in range(nS)
                          if Ct[i, j] != 0.0)
    f2_expr = gp.quicksum(y[a, i, i] for a in range(nA) for i in range(nS))

    # ---- Paso 4: restricciones ----
    # (C1) asignacion unica
    m.addConstrs((x.sum(v, "*") == 1 for v in range(nV)), name="asignacion")

    # (C2) linealizacion Frieze-Yadegar: y liga x_u,i con x_v,j sin exigir binariedad
    for a in range(nA):
        u, v = int(U[a]), int(V[a])
        for i in range(nS):
            m.addConstr(gp.quicksum(y[a, i, j] for j in range(nS)) == x[u, i],
                        name=f"lin1_{a}_{i}")
        for j in range(nS):
            m.addConstr(gp.quicksum(y[a, i, j] for i in range(nS)) == x[v, j],
                        name=f"lin2_{a}_{j}")

    # (C3) cuotas con tolerancia t
    quota, t = P.quota, P.t
    for i in range(nS):
        lo = int(np.ceil((1 - t) * quota[i]))
        hi = int(np.floor((1 + t) * quota[i]))
        m.addConstr(x.sum("*", i) >= lo, name=f"cuota_min_{i}")
        m.addConstr(x.sum("*", i) <= hi, name=f"cuota_max_{i}")

    # (C4) nodos preexistentes fijos -> bounds, no restriccion (mas rapido: presolve los elimina)
    for v in range(nV):
        if escenario[v] >= 0:
            for i in range(nS):
                val = 1.0 if i == escenario[v] else 0.0
                x[v, i].lb = val
                x[v, i].ub = val

    m._x, m._y, m._f1, m._f2 = x, y, f1_expr, f2_expr
    m.update()
    return m


def resolver_milp(P, escenario, time_limit=600, mip_gap=0.01, lam1=1.0, lam2=1.0):
    """Resuelve un unico punto escalarizado: min lam1*f1 + lam2*f2.
       Regresa (s, f1, f2, info) donde info trae el MIPGap REAL de Gurobi
       (no una heuristica por tiempo), el status y el runtime."""
    m = construir_modelo(P, escenario, time_limit, mip_gap)
    m.setObjective(lam1 * m._f1 + lam2 * m._f2, GRB.MINIMIZE)
    m.optimize()
    s, f1, f2 = _extraer(m, P)
    info = _info_solver(m)
    return s, f1, f2, info


def _info_solver(m):
    """Extrae el estado real del solver: gap, cota dual, runtime, status.
       gap = None si no hay incumbente (infactible o sin solucion aun)."""
    status_nombre = {
        GRB.OPTIMAL: "OPTIMAL", GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE", GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.SUBOPTIMAL: "SUBOPTIMAL", GRB.INTERRUPTED: "INTERRUPTED",
    }.get(m.Status, f"STATUS_{m.Status}")
    tiene_sol = m.SolCount > 0
    return {
        "status": status_nombre,
        "status_code": m.Status,
        "mip_gap": float(m.MIPGap) if tiene_sol else None,   # gap REAL reportado por Gurobi
        "cota_dual": float(m.ObjBound) if hasattr(m, "ObjBound") else None,
        "incumbente": float(m.ObjVal) if tiene_sol else None,
        "runtime_s": float(m.Runtime),
        "nodos_explorados": int(m.NodeCount),
        "cerro_optimalidad": (m.Status == GRB.OPTIMAL),
    }


def frente_epsilon(P, escenario, epsilons, time_limit=600, mip_gap=0.01):
    """Frente de Pareto EXACTO por eps-restriccion:
         min f1  s.a.  f2 <= eps,   para cada eps en 'epsilons'.
       El modelo se construye UNA vez; solo se agrega/actualiza la
       restriccion f2<=eps y se reoptimiza -> Gurobi reutiliza la base
       anterior (warm start), mucho mas rapido que reconstruir cada vez."""
    m = construir_modelo(P, escenario, time_limit, mip_gap)
    m.setObjective(m._f1, GRB.MINIMIZE)
    restr_eps = m.addConstr(m._f2 <= epsilons[0], name="eps_restriccion")

    resultados = []
    for eps in epsilons:
        restr_eps.RHS = eps          # solo cambia el RHS -> reoptimiza en caliente
        m.update()
        m.optimize()
        info = _info_solver(m)
        if m.SolCount > 0:
            s, f1, f2 = _extraer(m, P)
        else:
            s, f1, f2 = None, None, None
        resultados.append({"epsilon": eps, "f1": f1, "f2": f2, "s": s, **info})
    return resultados


def _extraer(m, P):
    """Reconstruye el vector de asignacion s a partir de las x optimas."""
    if m.SolCount == 0:
        return None, None, None
    s = np.full(P.nV, -1, dtype=int)
    for v in range(P.nV):
        for i in range(10):
            if m._x[v, i].X > 0.5:
                s[v] = i
                break
    f1, f2 = P.f1(s), P.f2(s)   # revalidacion cruzada con core.py: deben coincidir con m._f1/._f2
    return s, f1, f2


if __name__ == "__main__":
    import sys, time
    sys.path.insert(0, "/home/claude")
    from core import Problema

    # Instancia diminuta 3x3 (960 variables) SOLO para probar la mecanica
    # del modelo bajo el limite de la licencia size-limited de este entorno.
    # En un Gurobi con licencia completa, usar Problema(6,6) o Problema(8,8).
    P = Problema(3, 3, t=0.10)
    esc = P.escenario(seed=0)
    print(f"Instancia de prueba: nV={P.nV} nA={P.nA} cuota={P.quota}")
    print(f"Escenario: {(esc>=0).sum()} nodos preexistentes fijos")

    t0 = time.perf_counter()
    s, f1, f2, info = resolver_milp(P, esc, time_limit=60)
    print(f"\nUn solo punto (lam1=lam2=1): f1={f1:.4f} f2={f2}  [{time.perf_counter()-t0:.1f}s]")
    print(f"Info del solver: status={info['status']}  MIPGap={info['mip_gap']*100:.4f}%  "
          f"cota_dual={info['cota_dual']:.4f}  incumbente={info['incumbente']:.4f}  "
          f"nodos={info['nodos_explorados']}  cerro={info['cerro_optimalidad']}")
    assert P.cuota_ok(s), "ERROR: solucion MILP viola la cuota"
    assert (s[esc >= 0] == esc[esc >= 0]).all(), "ERROR: MILP modifico nodo preexistente"
    print("Validacion cruzada con core.py: OK (cuota respetada, preexistentes intactos)")

    print("\nFrente de Pareto exacto por eps-restriccion:")
    frente = frente_epsilon(P, esc, epsilons=range(0, 8), time_limit=30)
    for r in frente:
        if r["f1"] is not None:
            print(f"  eps={r['epsilon']:2d}  f1*={r['f1']:7.4f}  f2={r['f2']:2d}  "
                  f"MIPGap={r['mip_gap']*100:6.3f}%  cota_dual={r['cota_dual']:7.4f}  "
                  f"status={r['status']:12s} t={r['runtime_s']:.2f}s")
        else:
            print(f"  eps={r['epsilon']:2d}  {r['status']} (sin incumbente)")
