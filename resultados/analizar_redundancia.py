"""Cuenta puntos eps redundantes (mismo f1* que el punto anterior, dentro de
tolerancia) en un barrido eps del MILP. Un punto redundante confirma que
relajar mas la cuota de monocultivo ya no ayuda a f1 -> refuerza el hallazgo
de 6x6 (f1 y f2 alineados porque C_ii=1 siempre es el maximo de competencia).
Uso: python analizar_redundancia.py resultados/milp_8x8.json
"""
import sys, json

def contar_redundantes(path, tol=1e-6):
    with open(path) as fh:
        frente = json.load(fh)
    validos = [r for r in frente if r["f1"] is not None]
    validos.sort(key=lambda r: r["epsilon"])
    redundantes = []
    prev_f1 = None
    for r in validos:
        if prev_f1 is not None and abs(r["f1"] - prev_f1) < tol:
            redundantes.append(r["epsilon"])
        prev_f1 = r["f1"]
    print(f"{path}: {len(validos)} puntos factibles, {len(redundantes)} redundantes "
          f"(mismo f1* que el eps anterior): eps={redundantes}")
    return validos, redundantes

if __name__ == "__main__":
    for path in sys.argv[1:]:
        contar_redundantes(path)
