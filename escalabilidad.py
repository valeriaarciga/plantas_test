"""
escalabilidad.py — Bloque 5: curva de escalabilidad MILP vs NSGA-II.

Corre el MILP en instancias crecientes con limite de tiempo, guardando el
MIPGap AUNQUE NO CIERRE (ese gap abierto es en si mismo el resultado que
demuestra la intratabilidad, no una corrida fallida). Corre NSGA-II en las
mismas instancias (donde el MILP si es tratable) y ademas en la escala real
14x47 (donde el MILP ya no aplica).

Requiere: core.py, milp_gurobi.py, nsga2_reforestacion.py en el mismo path.

Uso:
    python3 escalabilidad.py                      # corrida completa (tarda)
    python3 escalabilidad.py --smoke               # prueba rapida, tiempos cortos
"""
import sys, time, csv, argparse
import numpy as np

from core import Problema
from milp_gurobi import resolver_milp
from nsga2_reforestacion import correr_nsga2
from pymoo.indicators.hv import HV


# ------------------------------------------------------------------
# Configuracion de la escalera de instancias
# ------------------------------------------------------------------
# (R, C, correr_milp, time_limit_milp_s)
ESCALERA = [
    (6,  6,  True,   600),
    (8,  8,  True,  1800),
    (10, 10, True,  3600),
    (12, 12, True,  3600),
    (14, 47, False, None),   # escala real: solo NSGA-II, el MILP ya no aplica
]

# parametros NSGA-II (misma configuracion en todas las instancias, para
# que la comparacion de tiempos/HV sea homogenea)
POP, GENS, SEED = 120, 250, 0
REF_POINT = (1500.0, 500.0)   # punto de referencia COMUN para el hipervolumen
                               # (peor caso esperado en f1 y f2; ver nota abajo)


def punto_referencia_valido(P, esc):
    """El punto de referencia del HV debe dominar a TODAS las soluciones
       observadas. Se estima con una solucion deliberadamente mala
       (asignacion aleatoria sin optimizar) mas un margen de seguridad."""
    rng = np.random.default_rng(0)
    peor = []
    for _ in range(20):
        s = P.sembrar(esc, rng=rng, perturbar=200)
        peor.append((P.f1(s), P.f2(s)))
    peor = np.array(peor)
    return float(peor[:, 0].max() * 1.2), float(peor[:, 1].max() * 1.2)


def correr_bloque5(escalera=ESCALERA, pop=POP, gens=GENS, seed=SEED,
                    salida_csv="escalabilidad_resultados.csv"):
    filas = []
    for R, C, con_milp, tl in escalera:
        print(f"\n{'='*70}\nInstancia {R}x{C}\n{'='*70}")
        P = Problema(R, C, t=0.10)
        esc = P.escenario(seed=seed)
        ref = punto_referencia_valido(P, esc)
        hv_calc = HV(ref_point=np.array(ref))

        fila = {"R": R, "C": C, "nV": P.nV, "nA": P.nA}

        # ---------------- MILP ----------------
        if con_milp:
            t0 = time.perf_counter()
            s_m, f1_m, f2_m, info_m = resolver_milp(P, esc, time_limit=tl, mip_gap=0.01)
            dt_m = time.perf_counter() - t0
            if s_m is None:
                # Infactible o sin solucion en el tiempo dado. A escalas muy
                # chicas el redondeo de cuotas + nodos preexistentes fijos
                # puede dejar el modelo sin region factible: es una fragilidad
                # conocida (ver seccion 3.2 del documento) y NO un error del
                # script. Se registra como tal en vez de fallar la corrida.
                fila.update({"milp_f1": None, "milp_f2": None,
                             "milp_tiempo_s": round(dt_m, 1),
                             "milp_time_limit_s": tl,
                             "milp_gap_pct": None,
                             "milp_cota_dual": None,
                             "milp_nodos": None,
                             "milp_cerro": None,
                             "milp_status": info_m["status"]})
                print(f"MILP:   {info_m['status']} en {dt_m:.1f}s "
                      f"(probable degeneracion de cuota a esta escala)")
            else:
                gap_pct = info_m["mip_gap"] * 100
                fila.update({
                    "milp_f1": f1_m, "milp_f2": f2_m,
                    "milp_tiempo_s": round(dt_m, 1),
                    "milp_time_limit_s": tl,
                    "milp_gap_pct": round(gap_pct, 4),        # <-- GAP REAL de Gurobi (ObjBound vs incumbente)
                    "milp_cota_dual": round(info_m["cota_dual"], 4),
                    "milp_nodos": info_m["nodos_explorados"],
                    "milp_cerro": info_m["cerro_optimalidad"],  # <-- status real, no heuristica por tiempo
                    "milp_status": info_m["status"],
                })
                cerro_txt = "CERRO A OPTIMALIDAD" if info_m["cerro_optimalidad"] else \
                            f"NO CERRO (gap real {gap_pct:.3f}%, cota dual {info_m['cota_dual']:.4f})"
                print(f"MILP:   f1={f1_m:.4f}  f2={f2_m}  tiempo={dt_m:.1f}s "
                      f"(limite {tl}s, {info_m['nodos_explorados']} nodos B&B) -> {cerro_txt}")
        else:
            fila.update({"milp_f1": None, "milp_f2": None,
                         "milp_tiempo_s": None, "milp_time_limit_s": None,
                         "milp_gap_pct": None, "milp_cota_dual": None,
                         "milp_nodos": None, "milp_cerro": None,
                         "milp_status": "omitido"})
            print("MILP:   omitido (instancia fuera de rango tratable)")

        # ---------------- NSGA-II ----------------
        t0 = time.perf_counter()
        res, hv_hist, gen_conv = correr_nsga2(P, esc, pop=pop, gens=gens,
                                               seed=seed, ref_point=ref)
        dt_n = time.perf_counter() - t0
        hv_final = hv_hist[-1]
        f1_min = float(res.F[:, 0].min())
        f2_en_f1min = int(res.F[np.argmin(res.F[:, 0]), 1])

        fila.update({
            "nsga2_tiempo_s": round(dt_n, 1),
            "nsga2_gen_convergencia": gen_conv,
            "nsga2_gens_totales": gens,
            "nsga2_hv": round(hv_final, 2),
            "nsga2_f1_min": round(f1_min, 4),
            "nsga2_f2_en_f1min": f2_en_f1min,
            "nsga2_n_soluciones": len(res.F),
        })
        print(f"NSGA-II: f1_min={f1_min:.4f} (f2={f2_en_f1min})  "
              f"HV={hv_final:.2f}  tiempo={dt_n:.1f}s  "
              f"convergio en gen {gen_conv}/{gens}")

        # ---------------- comparacion directa (solo si hay MILP factible) ----
        if con_milp and fila["milp_f1"] is not None:
            gap_pct = 100 * (f1_min - fila["milp_f1"]) / fila["milp_f1"] if fila["milp_f1"] != 0 else float("nan")
            fila["nsga2_gap_vs_milp_pct"] = round(gap_pct, 2)
            print(f"Gap NSGA-II vs MILP (en f1, a f2 comparable): {gap_pct:+.2f}%")
        else:
            fila["nsga2_gap_vs_milp_pct"] = None

        filas.append(fila)

    # ---------------- tabla final ----------------
    print(f"\n{'='*70}\nTABLA RESUMEN\n{'='*70}")
    campos = list(filas[0].keys())
    ancho = max(len(c) for c in campos)
    for f in filas:
        print(f"\nInstancia {f['R']}x{f['C']}:")
        for c in campos[2:]:
            print(f"  {c:<28s} {f[c]}")

    with open(salida_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=campos)
        writer.writeheader()
        writer.writerows(filas)
    print(f"\nResultados guardados en {salida_csv}")
    return filas


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                     help="corrida rapida de prueba (instancias chicas, tiempos cortos)")
    args = ap.parse_args()

    if args.smoke:
        # SOLO para validar que el script corre de principio a fin;
        # tamanos elegidos para caber bajo licencias size-limited de Gurobi
        # Y para ser factibles (a escalas muy chicas la cuota redondeada
        # puede no tener region factible; ver nota en el cuerpo del script).
        escalera_prueba = [
            (3, 3, True, 20),
            (4, 3, False, None),   # >2000 vars: se omite el MILP en este smoke test
        ]
        correr_bloque5(escalera=escalera_prueba, pop=40, gens=40,
                       salida_csv="smoke_resultados.csv")
    else:
        correr_bloque5()
