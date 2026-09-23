"""
nsga2_reforestacion.py — Bloque 4: NSGA-II (pymoo) con operadores adaptados
a la geometria hexagonal, reutilizando exclusivamente las funciones de core.py.

Uso:
    from core import Problema
    from nsga2_reforestacion import correr_nsga2

    P = Problema(6, 6)
    esc = P.escenario(seed=0)
    resultado, historial_hv = correr_nsga2(P, esc, pop=100, gens=150, seed=0)
"""
import numpy as np
from pymoo.core.problem import Problem as PymooProblem
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV
from pymoo.core.callback import Callback


# ------------------------------------------------------------------
# 1. Definicion del problema: SOLO delega en core.py, no reimplementa nada
# ------------------------------------------------------------------
class ReforestacionProblem(PymooProblem):
    """n_var = nV enteros categoricos (especie por nodo). n_obj = 2 (f1, f2).
       Los nodos preexistentes (escenario >= 0) quedan fuera del dominio
       libre: se fijan en la representacion y los operadores nunca los tocan."""

    def __init__(self, P, escenario):
        self.P = P
        self.escenario = escenario
        self.libres = np.where(escenario < 0)[0]
        super().__init__(n_var=P.nV, n_obj=2, n_constr=0,
                          xl=0, xu=9, vtype=int)

    def _evaluate(self, X, out, *args, **kwargs):
        # X: matriz (n_individuos, nV). Cada fila ya trae los nodos
        # preexistentes con su valor fijo (los operadores no los alteran).
        F = np.zeros((X.shape[0], 2))
        for k in range(X.shape[0]):
            s = X[k].astype(int)
            F[k, 0] = self.P.f1(s)
            F[k, 1] = self.P.f2(s)
        out["F"] = F


# ------------------------------------------------------------------
# 2. Muestreo inicial: siembra por 3-coloracion (P.sembrar de core.py)
# ------------------------------------------------------------------
class SiembraColoracion(Sampling):
    def __init__(self, escenario, perturbar=15):
        super().__init__()
        self.escenario = escenario
        self.perturbar = perturbar

    def _do(self, problem, n_samples, **kwargs):
        P, rng = problem.P, np.random.default_rng()
        X = np.empty((n_samples, P.nV), dtype=int)
        for k in range(n_samples):
            X[k] = P.sembrar(self.escenario, rng=rng, perturbar=self.perturbar)
        return X


# ------------------------------------------------------------------
# 3. Mutacion: intercambio (swap) -> preserva la cuota EXACTAMENTE
#    (no hace falta reparar nada: es la propiedad clave del operador)
# ------------------------------------------------------------------
class MutacionIntercambio(Mutation):
    def __init__(self, prob=0.3):
        super().__init__()
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        libres, rng = problem.libres, np.random.default_rng()
        Xm = X.copy()
        for k in range(Xm.shape[0]):
            if rng.random() < self.prob:
                a, b = rng.choice(libres, 2, replace=False)
                Xm[k, a], Xm[k, b] = Xm[k, b], Xm[k, a]
        return Xm


# ------------------------------------------------------------------
# 4. Cruzamiento: bola hexagonal -> hereda un parche compacto del
#    grafo (no una particion arbitraria por filas), y repara por
#    conteo para restaurar la cuota tras heredar el parche.
# ------------------------------------------------------------------
class CruceBolaHexagonal(Crossover):
    def __init__(self, radio_max=3):
        super().__init__(2, 1)   # 2 padres -> 1 hijo
        self.radio_max = radio_max

    def _do(self, problem, X, **kwargs):
        P = problem.P
        _, n_matings, n_var = X.shape
        Y = np.empty((1, n_matings, n_var), dtype=int)
        rng = np.random.default_rng()
        for k in range(n_matings):
            padre1, padre2 = X[0, k].astype(int), X[1, k].astype(int)
            hijo = padre1.copy()
            centro = int(rng.choice(problem.libres))
            radio = int(rng.integers(1, self.radio_max + 1))
            # BFS acotado a 'radio' saltos sobre el grafo (bola hexagonal)
            parche = self._bola(P, centro, radio)
            parche = np.intersect1d(parche, problem.libres)
            hijo[parche] = padre2[parche]
            Y[0, k] = self._reparar_cuota(P, hijo, problem.libres, rng)
        return Y

    @staticmethod
    def _bola(P, centro, radio):
        visitados = {centro}
        frontera = {centro}
        for _ in range(radio):
            nueva = set()
            for v in frontera:
                nueva.update(P.vec[v].tolist())
            nueva -= visitados
            visitados |= nueva
            frontera = nueva
        return np.array(sorted(visitados))

    @staticmethod
    def _reparar_cuota(P, s, libres, rng):
        """Restaura el conteo exacto por especie tras el cruce, moviendo
           el minimo de nodos necesarios (los que sobran de una especie
           pasan a la que falta), preservando el patron espacial heredado
           en todo lo demas."""
        conteo = np.bincount(s[libres], minlength=10)
        objetivo = P.quota - np.bincount(P_ocupadas(s, libres), minlength=10)
        # nodos libres a favor (sobran) / en contra (faltan) por especie
        sobran = conteo - objetivo
        idx_libres = list(libres)
        rng.shuffle(idx_libres)
        for i in np.where(sobran > 0)[0]:
            for _ in range(sobran[i]):
                # busca un nodo libre de la especie i para reasignar
                cand = next((v for v in idx_libres if s[v] == i), None)
                if cand is None:
                    break
                j = int(np.argmin(sobran))   # especie que mas falta
                s[cand] = j
                sobran[i] -= 1
                sobran[j] += 1
        return s


def P_ocupadas(s, libres):
    """Auxiliar: nodos NO libres (preexistentes) no aportan al conteo ajustable."""
    mask = np.ones(len(s), dtype=bool)
    mask[libres] = False
    return s[mask & (s >= 0)]


# ------------------------------------------------------------------
# 5. Callback: historial de hipervolumen por generacion, para el
#    criterio de convergencia (sin mejora en 30 generaciones -> parar)
# ------------------------------------------------------------------
class HistorialHV(Callback):
    def __init__(self, ref_point):
        super().__init__()
        self.hv = HV(ref_point=ref_point)
        self.historial = []

    def notify(self, algorithm):
        F = algorithm.pop.get("F")
        self.historial.append(self.hv(F))


# ------------------------------------------------------------------
# 6. Orquestador
# ------------------------------------------------------------------
def correr_nsga2(P, escenario, pop=100, gens=150, seed=0,
                  ref_point=(1200.0, 400.0), paciencia=30):
    problem = ReforestacionProblem(P, escenario)
    algorithm = NSGA2(
        pop_size=pop,
        sampling=SiembraColoracion(escenario),
        crossover=CruceBolaHexagonal(radio_max=3),
        mutation=MutacionIntercambio(prob=0.3),
        # eliminate_duplicates=True compara por genotipo (X) exacto, que es lo
        # unico posible en el punto del ciclo donde pymoo lo invoca (mating,
        # antes de evaluar F en los hijos -> comparar en espacio objetivo ahi
        # no es viable). Es la causa raiz del frente colapsado: sin esto,
        # copias identicas dominan el conteo de "no dominados" y diluyen la
        # presion de seleccion. El costo (cdist sobre ~100-200 individuos)
        # es trivial incluso a 658 variables; el temor a que fuera "costoso"
        # no se sostuvo al medirlo.
        eliminate_duplicates=True,
    )
    callback = HistorialHV(ref_point=np.array(ref_point))
    res = minimize(problem, algorithm, ("n_gen", gens),
                    seed=seed, callback=callback, verbose=False)

    # criterio de convergencia: reporta en que generacion se estanco el HV
    hv_hist = callback.historial
    gen_convergencia = gens
    for g in range(paciencia, len(hv_hist)):
        if max(hv_hist[g - paciencia:g]) <= hv_hist[g - paciencia] + 1e-9:
            gen_convergencia = g
            break
    return res, hv_hist, gen_convergencia


if __name__ == "__main__":
    import sys, time
    sys.path.insert(0, "/home/claude")
    from core import Problema

    P = Problema(6, 6, t=0.10)
    esc = P.escenario(seed=0)
    print(f"Instancia: nV={P.nV}  nA={P.nA}  cuota={P.quota}")

    t0 = time.perf_counter()
    res, hv_hist, gen_conv = correr_nsga2(P, esc, pop=80, gens=100, seed=0)
    dt = time.perf_counter() - t0

    F = res.F
    print(f"\nFrente aproximado: {len(F)} soluciones no dominadas  [{dt:.1f}s]")
    print(f"Convergencia (HV) en la generacion {gen_conv} de {len(hv_hist)}")
    print(f"HV final = {hv_hist[-1]:.4f}")

    orden = np.argsort(F[:, 0])
    print("\n  f1*        f2")
    for f1, f2 in F[orden][:12]:
        print(f"  {f1:8.4f}  {f2:5.0f}")

    # validaciones cruzadas con core.py
    for x in res.X[:5].astype(int):
        assert P.cuota_ok(x), "ERROR: individuo del frente viola la cuota"
        assert (x[esc >= 0] == esc[esc >= 0]).all(), "ERROR: se modifico un preexistente"
    print("\nValidacion cruzada: cuota respetada y preexistentes intactos en toda la muestra revisada")
