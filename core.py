"""
core.py — Nucleo del modelo de optimizacion de reforestacion.
Bloque 1 del plan de experimentos. Sin dependencias mas alla de numpy/scipy.

Uso rapido:
    from core import *
    P = Problema(14, 47)
    esc = P.escenario(seed=0)
    s   = P.sembrar(esc, rng=np.random.default_rng(0))
    print(P.f1(s), P.f2(s))
"""
import numpy as np
from scipy.stats import binom

# ----------------------------------------------------------------------
# Datos de especies (validados en campo, salvo altura -> ver docstring)
# ----------------------------------------------------------------------
SP     = ["AL", "AS", "ASc", "ASt", "OC", "OE", "OR", "OS", "PL", "YF"]
NOMBRE = ["Agave lechuguilla", "Agave salmiana", "Agave scabra", "Agave striata",
          "Opuntia cantabrigiensis", "Opuntia engelmannii", "Opuntia robusta",
          "Opuntia streptacantha", "Prosopis laevigata", "Yucca filifera"]

R_HA = np.array([42, 196, 42, 42, 49, 38, 73, 64, 86, 26])          # cuota / ha
S_I  = np.array([.9265, .9082, .7453, .9138, .8287,
                 .8194, .9198, .8525, .8179, .7263])                 # supervivencia 2023
PI   = np.array([.06326106, .29513344, .06203063, .05956977, .07772922,
                 .06190335, .11650897, .09711910, .12804956, .03869490])
PI   = PI / PI.sum()                                                 # composicion preexistente
P_OCU = 0.19632575                                                   # prob. ocupacion por nodo
N_HA  = 658

# Rasgos funcionales. ATENCION: H_ADULTA es literatura general, NO dato del
# socio formador (ellos solo dieron tallas de vivero de 15-50 cm, inservibles
# a horizonte de 5 anos). Es el parametro mas debil -> analisis de sensibilidad.
H_ADULTA = np.array([0.4, 2.0, 1.0, 0.6, 1.5, 2.0, 2.0, 3.5, 8.0, 8.0])   # m
GREMIO   = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 3])   # 0 CAM-roseta 1 CAM-cladodio 2 C3-lenosa 3 CAM-arborescente
RAIZ     = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0.5])   # 0 somera, .5 media, 1 profunda


# ----------------------------------------------------------------------
# Matriz de competencia por rasgos
# ----------------------------------------------------------------------
def matriz_C(sigma=0.5, beta=0.20, w=(0.4, 0.4, 0.2)):
    """C_ij = exp(-d2/(2 sigma^2)) - beta * F_ij   con F = pares nodriza-suculenta."""
    wh, wr, wg = w
    hh = np.log(H_ADULTA)
    hh = (hh - hh.min()) / (hh.max() - hh.min())
    d2 = (wh * (hh[:, None] - hh[None, :]) ** 2
          + wr * (RAIZ[:, None] - RAIZ[None, :]) ** 2
          + wg * (GREMIO[:, None] != GREMIO[None, :]).astype(float))
    C = np.exp(-d2 / (2 * sigma ** 2))
    nodriza, suculenta = (GREMIO == 2), np.isin(GREMIO, [0, 1, 3])
    F = np.zeros((10, 10))
    F[np.ix_(nodriza, suculenta)] = 1.0
    F[np.ix_(suculenta, nodriza)] = 1.0
    C = C - beta * F
    np.fill_diagonal(C, 1.0)
    return C


def matriz_C_tilde(**kw):
    """Competencia esperada realizada: ambos individuos deben sobrevivir."""
    return S_I[:, None] * S_I[None, :] * matriz_C(**kw)


# ----------------------------------------------------------------------
# Cuotas reescaladas (mayores residuos) para instancias reducidas
# ----------------------------------------------------------------------
def cuotas(nV):
    """Reescala R_HA a nV nodos garantizando suma exacta = nV."""
    exacto = R_HA / N_HA * nV
    base   = np.floor(exacto).astype(int)
    resto  = nV - base.sum()
    if resto > 0:
        orden = np.argsort(-(exacto - base))
        base[orden[:resto]] += 1
    return base


def compra_minima(alpha=0.20, R=R_HA):
    """n_i minimo tal que P(Bin(n, S_i) >= R_i) >= 1 - alpha. Exacto, sin aprox. normal."""
    out = np.zeros(len(R), dtype=int)
    for i in range(len(R)):
        n = int(R[i])
        while binom.sf(R[i] - 1, n, S_I[i]) < 1 - alpha:
            n += 1
        out[i] = n
    return out


# ----------------------------------------------------------------------
# Problema: grafo, coloracion, escenarios, objetivos
# ----------------------------------------------------------------------
class Problema:
    def __init__(self, R=14, C=47, t=0.10, **kwC):
        self.R, self.C, self.nV, self.t = R, C, R * C, t
        self.U, self.V = self._aristas()
        self.nA   = len(self.U)
        self.vec  = self._vecindarios()
        self.col  = self._coloracion()
        self.Ct   = matriz_C_tilde(**kwC)
        self.quota = cuotas(self.nV)

    # ---- grafo tres bolillos (retisculo triangular) ----
    def _aristas(self):
        R, C = self.R, self.C
        idx = lambda r, c: r * C + c
        E = set()
        for r in range(R):
            for c in range(C):
                u = idx(r, c)
                if c + 1 < C:
                    E.add((u, idx(r, c + 1)))
                if r + 1 < R:
                    for cc in ((c - 1, c) if r % 2 == 0 else (c, c + 1)):
                        if 0 <= cc < C:
                            E.add((u, idx(r + 1, cc)))
        E = np.array(sorted(E))
        return E[:, 0].copy(), E[:, 1].copy()

    def _vecindarios(self):
        vec = [[] for _ in range(self.nV)]
        for u, v in zip(self.U, self.V):
            vec[u].append(v); vec[v].append(u)
        return [np.array(x) for x in vec]

    def _coloracion(self):
        """3-coloracion propia via coordenadas axiales. Verificada: 0 aristas monocromaticas."""
        col = np.zeros(self.nV, dtype=int)
        for r in range(self.R):
            for c in range(self.C):
                q = c - (r - (r & 1)) // 2
                col[r * self.C + c] = (q - r) % 3
        return col

    # ---- escenarios Montecarlo ----
    def escenario(self, seed=0):
        """Devuelve arreglo de nV enteros: -1 = nodo libre, 0..9 = especie preexistente fija."""
        rng = np.random.default_rng(seed)
        s = np.full(self.nV, -1, dtype=int)
        ocu = rng.random(self.nV) < P_OCU
        s[ocu] = rng.choice(10, size=int(ocu.sum()), p=PI)
        return s

    # ---- objetivos (vectorizados) ----
    def f1(self, s):
        """Competencia esperada realizada."""
        return float(self.Ct[s[self.U], s[self.V]].sum())

    def f2(self, s):
        """Aristas monoespecificas (monocultivo)."""
        return int((s[self.U] == s[self.V]).sum())

    def evaluar(self, s):
        return self.f1(s), self.f2(s)

    # ---- metricas reportadas (no optimizadas) ----
    def gadow(self, s):
        return float(np.mean([np.mean(s[self.vec[v]] != s[v]) for v in range(self.nV)]))

    def kl_cuota(self, s):
        p = np.bincount(s, minlength=10) / self.nV
        r = self.quota / self.nV
        m = p > 0
        return float(np.sum(p[m] * np.log(p[m] / r[m])))

    # ---- validacion de cuota ----
    def cuota_ok(self, s):
        n = np.bincount(s, minlength=10)
        return bool(np.all(n >= np.ceil((1 - self.t) * self.quota)) and
                    np.all(n <= np.floor((1 + self.t) * self.quota)))

    # ---- siembra por 3-coloracion (extremo del frente: f2 minimo) ----
    def sembrar(self, esc, rng=None, perturbar=0):
        """Aloja cada especie dentro de una clase de color -> minimiza f2.
           Respeta nodos preexistentes fijos y la cuota exacta."""
        rng = rng or np.random.default_rng(0)
        s = esc.copy()
        libres = np.where(s < 0)[0]
        self._libre = (esc < 0)          # nodos modificables (los fijos nunca se tocan)
        falta = self.quota - np.bincount(esc[esc >= 0], minlength=10)
        falta = np.maximum(falta, 0)
        # ajusta para que la suma coincida con el numero de nodos libres
        d = len(libres) - falta.sum()
        while d != 0:
            j = int(np.argmax(falta)) if d < 0 else int(np.argmin(falta))
            falta[j] += 1 if d > 0 else -1
            d = len(libres) - falta.sum()
        # capacidad libre por clase de color
        cap = {k: list(libres[self.col[libres] == k]) for k in range(3)}
        for k in cap:
            rng.shuffle(cap[k])
        # asigna especies enteras a la clase con mas espacio (greedy)
        for i in np.argsort(-falta):
            need = falta[i]
            for k in sorted(cap, key=lambda k: -len(cap[k])):
                take = min(need, len(cap[k]))
                for _ in range(take):
                    s[cap[k].pop()] = i
                need -= take
                if need == 0:
                    break
        # perturbacion: intercambios que preservan la cuota exactamente
        for _ in range(perturbar):
            a, b = rng.choice(libres, 2, replace=False)
            if s[a] != s[b]:
                s[a], s[b] = s[b], s[a]
        return s

    # ---- mutacion por intercambio: preserva la cuota por construccion ----
    def swap(self, s, rng):
        libres = np.where(s >= 0)[0]
        a, b = rng.choice(libres, 2, replace=False)
        s[a], s[b] = s[b], s[a]
        return a, b

    def delta_f1(self, s, a, b):
        """Cambio en f1 al intercambiar a<->b. O(12) en vez de O(|A|)."""
        Ct, na, nb = self.Ct, self.vec[a], self.vec[b]
        na = na[na != b]; nb = nb[nb != a]
        return float((Ct[s[b], s[na]] - Ct[s[a], s[na]]).sum()
                     + (Ct[s[a], s[nb]] - Ct[s[b], s[nb]]).sum())

    def delta_f2(self, s, a, b):
        """Cambio en f2 al intercambiar a<->b. O(12)."""
        na, nb = self.vec[a], self.vec[b]
        na = na[na != b]; nb = nb[nb != a]
        return int(((s[na] == s[b]).sum() - (s[na] == s[a]).sum())
                   + ((s[nb] == s[a]).sum() - (s[nb] == s[b]).sum()))

    # ---- busqueda local memetica (intercambios: cuota invariante) ----
    def busqueda_local(self, s, w1=1.0, w2=1.0, iters=40000, rng=None, cand=24):
        """Primer-mejora sobre swaps entre nodos libres.
           Sesga la eleccion hacia los nodos mas conflictivos.
           La cuota se preserva por construccion: no hay reparacion."""
        rng = rng or np.random.default_rng(0)
        s = s.copy()
        libres = np.where(np.asarray(self._libre))[0]
        if len(libres) < 2:
            return s
        for _ in range(iters):
            # nodo a: sesgado a los conflictivos; nodo b: candidatos al azar
            a = libres[rng.integers(len(libres))]
            bs = libres[rng.integers(0, len(libres), size=cand)]
            mejor, mejor_b = 0.0, -1
            for b in bs:
                if b == a or s[a] == s[b]:
                    continue
                d = w1 * self.delta_f1(s, a, b) + w2 * self.delta_f2(s, a, b)
                if d < mejor:
                    mejor, mejor_b = d, b
            if mejor_b >= 0:
                s[a], s[mejor_b] = s[mejor_b], s[a]
        return s


# ----------------------------------------------------------------------
# Pruebas de validacion (deben pasar todas antes de seguir al bloque 2)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 62)
    P = Problema(14, 47)
    print(f"|V| = {P.nV}   (esperado 658)")
    print(f"|A| = {P.nA}   (esperado 1853)")
    grados = np.array([len(v) for v in P.vec])
    print(f"grado max = {grados.max()} (esperado 6), medio = {grados.mean():.2f} (esperado 5.63)")

    mono = int((P.col[P.U] == P.col[P.V]).sum())
    print(f"aristas monocromaticas de la 3-coloracion = {mono} (esperado 0)")
    print(f"tamanos de clase = {np.bincount(P.col)} (esperado [224 217 217])")

    C = matriz_C()
    print(f"\nC simetrica: {np.allclose(C, C.T)};  diagonal = 1: {np.allclose(np.diag(C), 1)}")
    print(f"C[PL,AL] = {C[8,0]:+.3f} (negativo = facilitacion de la nodriza)")
    print(f"rango de C~ = [{P.Ct.min():.3f}, {P.Ct.max():.3f}]")

    # terreno limpio -> f2 debe ser 0 (Propiedad B)
    vacio = np.full(P.nV, -1)
    s0 = P.sembrar(vacio, np.random.default_rng(0))
    print(f"\nTerreno limpio -> f2 = {P.f2(s0)} (esperado 0);  cuota ok = {P.cuota_ok(s0)}")
    print(f"                  f1 = {P.f1(s0):.1f}   Gadow = {P.gadow(s0):.3f}   KL = {P.kl_cuota(s0):.5f}")

    # piso estructural impuesto por la vegetacion preexistente
    pisos = []
    for k in range(300):
        e = P.escenario(seed=k)
        m = (e[P.U] >= 0) & (e[P.U] == e[P.V])
        pisos.append(int(m.sum()))
    pisos = np.array(pisos)
    teorico = P.nA * P_OCU ** 2 * (PI ** 2).sum()
    print(f"\nPiso preexistente: simulado {pisos.mean():.2f} +/- {pisos.std():.2f}"
          f"   teorico {teorico:.2f}   (deben coincidir)")

    # terreno limpio + busqueda local -> f2 debe llegar a 0
    import time
    t0 = time.perf_counter()
    s0b = P.busqueda_local(s0, w1=1.0, w2=1.0, iters=20000, rng=np.random.default_rng(3))
    print(f"Terreno limpio + busqueda local -> f1 = {P.f1(s0b):.1f}, f2 = {P.f2(s0b)}, "
          f"cuota ok = {P.cuota_ok(s0b)}  [{time.perf_counter()-t0:.1f} s]")

    # con escenario: f2 sembrado debe estar cerca del piso, nunca por debajo
    e = P.escenario(seed=1)
    piso = int(((e[P.U] >= 0) & (e[P.U] == e[P.V])).sum())
    s1 = P.sembrar(e, np.random.default_rng(1))
    t0 = time.perf_counter()
    s1b = P.busqueda_local(s1, w1=1.0, w2=1.0, iters=40000, rng=np.random.default_rng(4))
    dt = time.perf_counter() - t0
    print(f"\nEscenario 1: piso teorico f2 = {piso}")
    print(f"   sembrado        : f1 = {P.f1(s1):7.1f}   f2 = {P.f2(s1):4d}")
    print(f"   + busqueda local: f1 = {P.f1(s1b):7.1f}   f2 = {P.f2(s1b):4d}   "
          f"cuota ok = {P.cuota_ok(s1b)}  [{dt:.1f} s]")
    assert P.f2(s1b) >= piso, "ERROR: f2 por debajo del piso teorico"
    assert (s1b[e >= 0] == e[e >= 0]).all(), "ERROR: se modifico un nodo preexistente"

    # delta incremental debe coincidir con la reevaluacion completa
    rng = np.random.default_rng(7)
    s = s1.copy(); antes = P.f1(s)
    a, b = P.swap(s, rng)
    s[a], s[b] = s[b], s[a]                 # deshacer para medir el delta
    d = P.delta_f1(s, a, b)
    s[a], s[b] = s[b], s[a]                 # rehacer
    print(f"\ndelta_f1 = {d:+.6f}   real = {P.f1(s)-antes:+.6f}   "
          f"coinciden: {abs(d-(P.f1(s)-antes)) < 1e-9}")

    # velocidad
    import time
    t0 = time.perf_counter()
    for _ in range(2000):
        P.f1(s); P.f2(s)
    print(f"\nEvaluacion completa (f1+f2): {(time.perf_counter()-t0)/2000*1e6:.1f} us")

    # tabla de compra
    print("\nCompra minima con garantia probabilistica:")
    print(f"{'Especie':26s} {'R_i':>4s}  " + "  ".join(f"a={a:<4.2f}" for a in (.30, .20, .10, .05)))
    tabla = {a: compra_minima(a) for a in (.30, .20, .10, .05)}
    for i in range(10):
        print(f"{NOMBRE[i]:26s} {R_HA[i]:4d}  " + "  ".join(f"{tabla[a][i]:6d}" for a in tabla))
    print(f"{'TOTAL':26s} {R_HA.sum():4d}  " + "  ".join(f"{tabla[a].sum():6d}" for a in tabla))

    # tamanos de instancia para la escalera del MILP
    print("\nEscalera de instancias para el MILP:")
    for (r, c) in [(4,4), (6,6), (8,8), (10,10), (12,12), (14,47)]:
        q = Problema(r, c)
        print(f"  {r:2d}x{c:2d}: |V|={q.nV:4d}  |A|={q.nA:5d}  "
              f"binarias={q.nV*10:6d}  continuas y={q.nA*100:7d}  cuota={cuotas(q.nV)}")
    print("=" * 62)
