"""
Caso 2 - solver.py
Resuelve el TSP exacto (deposito + 12 clientes) con la formulacion MTZ
(Miller-Tucker-Zemlin) usando PuLP + CBC (open source, sin licencia comercial).

Variables:
    x[i,j] in {0,1} -> 1 si la ruta va directo de i a j
    u[i]   >= 0      -> posicion de visita de i (solo para eliminar subciclos)

Restricciones:
    1) Salida unica de cada nodo
    2) Entrada unica a cada nodo
    3) MTZ: elimina subciclos que no incluyan al deposito
"""

import pulp

# ----- Matriz de distancias (13 puntos: deposito + 12 clientes) -----
NODES = [0, 3, 10, 21, 22, 35, 40, 41, 47, 65, 71, 76, 77]

_M = [
    [0, 29, 18, 16, 24, 32, 17, 28, 29, 27, 27, 33, 34],
    [29, 0, 38, 40, 29, 48, 15, 57, 27, 53, 55, 24, 44],
    [18, 38, 0, 31, 42, 49, 22, 27, 21, 19, 36, 50, 16],
    [16, 40, 31, 0, 21, 18, 32, 25, 45, 30, 16, 34, 47],
    [24, 29, 42, 21, 0, 20, 30, 46, 46, 48, 36, 14, 56],
    [32, 48, 49, 18, 20, 0, 45, 40, 60, 17, 25, 33, 65],
    [17, 15, 22, 32, 30, 45, 0, 44, 16, 39, 45, 32, 30],
    [28, 57, 27, 25, 46, 40, 44, 0, 48, 11, 17, 59, 41],
    [29, 27, 21, 45, 46, 60, 16, 48, 0, 40, 54, 48, 18],
    [27, 53, 19, 30, 48, 17, 39, 11, 40, 0, 26, 60, 30],
    [27, 55, 36, 16, 36, 25, 45, 17, 54, 26, 0, 50, 52],
    [33, 24, 50, 34, 14, 33, 32, 59, 48, 60, 50, 0, 62],
    [34, 44, 16, 47, 56, 65, 30, 41, 18, 30, 52, 62, 0],
]

DIST = {(NODES[i], NODES[j]): _M[i][j] for i in range(len(NODES)) for j in range(len(NODES))}
DEPOT = NODES[0]


def get_matrix():
    """Devuelve (lista_nodos, dict_distancias)."""
    return NODES, DIST


def solve_tsp(nodes=None, dist=None, depot=None, msg=False):
    """
    Resuelve el TSP exacto con formulacion MTZ.

    Parametros opcionales permiten reutilizar el solver con otra instancia
    (por ejemplo al agregar un cliente nuevo, pregunta 8 de la Parte III).

    Retorna un dict con:
        status      : estado del solver ("Optimal", etc.)
        total       : distancia total optima (km)
        sequence    : lista con el orden de visita, empieza y termina en el deposito
        arcs        : lista de tuplas (i, j, distancia) seleccionadas
        n_vars      : numero de variables del modelo
        n_constrs   : numero de restricciones del modelo
        elapsed     : tiempo de resolucion en segundos
    """
    import time

    nodes = nodes if nodes is not None else NODES
    dist = dist if dist is not None else DIST
    depot = depot if depot is not None else DEPOT
    n = len(nodes)

    prob = pulp.LpProblem("TSP_Caso2", pulp.LpMinimize)

    x = {
        (i, j): pulp.LpVariable(f"x_{i}_{j}", cat="Binary")
        for i in nodes
        for j in nodes
        if i != j
    }
    u = {i: pulp.LpVariable(f"u_{i}", lowBound=0, upBound=n) for i in nodes}

    # Funcion objetivo: minimizar distancia total
    prob += pulp.lpSum(dist[(i, j)] * x[(i, j)] for i in nodes for j in nodes if i != j)

    # Salida unica
    for i in nodes:
        prob += pulp.lpSum(x[(i, j)] for j in nodes if j != i) == 1, f"Salida_{i}"

    # Entrada unica
    for j in nodes:
        prob += pulp.lpSum(x[(i, j)] for i in nodes if i != j) == 1, f"Entrada_{j}"

    # MTZ: eliminacion de subciclos
    for i in nodes:
        if i == depot:
            continue
        for j in nodes:
            if j == depot or i == j:
                continue
            prob += u[i] - u[j] + n * x[(i, j)] <= n - 1, f"MTZ_{i}_{j}"

    prob += u[depot] == 0

    n_vars = len(x) + len(u)
    n_constrs = len(prob.constraints)

    start = time.time()
    prob.solve(pulp.PULP_CBC_CMD(msg=msg))
    elapsed = time.time() - start

    status = pulp.LpStatus[prob.status]
    result = {
        "status": status,
        "total": None,
        "sequence": None,
        "arcs": None,
        "n_vars": n_vars,
        "n_constrs": n_constrs,
        "elapsed": elapsed,
    }

    if status != "Optimal":
        return result

    total = pulp.value(prob.objective)
    arcs = [(i, j) for (i, j) in x if pulp.value(x[(i, j)]) > 0.5]
    nxt = {i: j for (i, j) in arcs}

    seq = [depot]
    cur = depot
    for _ in range(n):
        cur = nxt[cur]
        seq.append(cur)

    result["total"] = total
    result["sequence"] = seq
    result["arcs"] = [(i, j, dist[(i, j)]) for (i, j) in arcs]
    return result


if __name__ == "__main__":
    res = solve_tsp(msg=False)
    print("Estado:", res["status"])
    print("Distancia total optima:", res["total"], "km")
    print("Variables:", res["n_vars"], " Restricciones:", res["n_constrs"])
    print("Secuencia:", " -> ".join(str(s) for s in res["sequence"]))
    print("Tiempo:", round(res["elapsed"], 3), "s")
