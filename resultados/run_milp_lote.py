"""run_milp_lote.py -- resuelve a OPTIMALIDAD (mip_gap=0) el barrido
epsilon-restriccion de varias instancias en una sola corrida, guardando
todo en un unico JSON consolidado (en vez de un script + un JSON por tamano).

"A optimalidad" significa mip_gap=0.0: Gurobi debe CERRAR la brecha entre
la mejor solucion encontrada y la cota dual, no solo acercarse. Eso puede
tardar mucho o no cerrar nunca en instancias grandes -- por eso cada punto
del resultado trae "cerro_optimalidad": True/False, para que sepas cuales
si son un optimo certificado y cuales solo el mejor incumbente hallado
dentro del limite de tiempo.

Uso:
    python resultados/run_milp_lote.py
    python resultados/run_milp_lote.py --time-limit 900
    python resultados/run_milp_lote.py --instancias 6x6,8x8,10x10
"""
import sys, os, time, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import Problema
from milp_gurobi import frente_epsilon

# Escalera por defecto. OJO: a partir de aqui el numero de variables crece
# rapido (binarias = nV*10, continuas y = nA*100) y una licencia
# size-limited/trial de Gurobi (~2000 vars) ya no alcanza ni para plantear
# el modelo -- necesitas licencia completa para 8x8 en adelante.
INSTANCIAS_DEFAULT = [(6, 6), (8, 8), (10, 10), (12, 12)]
EPS_DEFAULT = [0, 1, 2, 3, 4, 6, 8, 10]


def parse_instancias(s):
    out = []
    for tok in s.split(","):
        r, c = tok.lower().split("x")
        out.append((int(r), int(c)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instancias", type=str, default=None,
                     help='ej. "6x6,8x8,10x10" (default: escalera 6x6..12x12)')
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time-limit", type=int, default=1800,
                     help="limite de tiempo POR PUNTO epsilon, en segundos")
    ap.add_argument("--epsilons", type=str, default=None,
                     help='ej. "0,1,2,3,4,6,8,10" (default: escalera estandar)')
    ap.add_argument("--salida", type=str, default="resultados/milp_lote.json")
    args = ap.parse_args()

    instancias = parse_instancias(args.instancias) if args.instancias else INSTANCIAS_DEFAULT
    epsilons = [int(x) for x in args.epsilons.split(",")] if args.epsilons else EPS_DEFAULT

    resultados_totales = []
    t0_global = time.perf_counter()

    for (r, c) in instancias:
        P = Problema(r, c, t=0.10)
        esc = P.escenario(seed=args.seed)
        n_bin = P.nV * 10
        n_cont = P.nA * 100
        print(f"\n=== Instancia {r}x{c}: nV={P.nV} nA={P.nA} "
              f"binarias={n_bin} continuas={n_cont} ===", file=sys.stderr)

        t0 = time.perf_counter()
        # mip_gap=0.0 -> exige cerrar la brecha exactamente (optimalidad certificada)
        frente = frente_epsilon(P, esc, epsilons=epsilons,
                                 time_limit=args.time_limit, mip_gap=0.0)
        dt = time.perf_counter() - t0

        n_cerrados = sum(1 for pt in frente if pt.get("cerro_optimalidad"))
        print(f"--- {r}x{c}: {n_cerrados}/{len(frente)} puntos con optimalidad "
              f"certificada, tiempo={dt:.1f}s ---", file=sys.stderr)

        for pt in frente:
            fila = {"instancia": f"{r}x{c}", "nV": P.nV, "nA": P.nA, "seed": args.seed}
            fila.update({k: pt[k] for k in
                         ("epsilon", "f1", "f2", "mip_gap", "cota_dual",
                          "runtime_s", "status", "nodos_explorados", "cerro_optimalidad")})
            resultados_totales.append(fila)
            if pt["f1"] is not None:
                print(f"  eps={pt['epsilon']:2d}  f1*={pt['f1']:8.4f}  f2={pt['f2']:2d}  "
                      f"gap={pt['mip_gap']*100:6.3f}%  optimo={pt['cerro_optimalidad']}  "
                      f"t={pt['runtime_s']:.1f}s", file=sys.stderr)
            else:
                print(f"  eps={pt['epsilon']:2d}  {pt['status']} (sin incumbente) "
                      f"t={pt['runtime_s']:.1f}s", file=sys.stderr)

    dt_total = time.perf_counter() - t0_global
    print(f"\n=== Tiempo total del lote: {dt_total:.1f}s "
          f"({dt_total/60:.1f} min) ===", file=sys.stderr)

    os.makedirs(os.path.dirname(args.salida), exist_ok=True)
    with open(args.salida, "w") as fh:
        json.dump(resultados_totales, fh, indent=2)
    print(f"guardado en {args.salida}", file=sys.stderr)

    # resumen: cuales instancias/puntos NO cerraron optimalidad (para que
    # sepas donde el numero reportado es solo el mejor incumbente, no un
    # optimo certificado)
    no_cerrados = [f for f in resultados_totales if not f.get("cerro_optimalidad")]
    if no_cerrados:
        print(f"\nAVISO: {len(no_cerrados)} puntos NO cerraron optimalidad "
              f"dentro del time-limit ({args.time_limit}s). Sube --time-limit "
              f"o interpreta esos valores como cota, no como optimo.", file=sys.stderr)


if __name__ == "__main__":
    main()
