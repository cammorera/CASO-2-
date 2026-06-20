"""
Caso 2 - TSP por etapas: Streamlit app
Resuelve el TSP exacto (deposito + 12 clientes) con la formulacion MTZ
implementada en solver.py (PuLP + CBC, sin licencia comercial).
"""

import pandas as pd
import streamlit as st

from solver import get_matrix, solve_tsp

st.set_page_config(page_title="Caso 2 · TSP por etapas", layout="wide")

st.title("Caso 2 · Enfoque por etapas — TSP exacto")
st.caption(
    "Resuelve el problema del agente viajero sobre la matriz Desde-Hasta ya reducida "
    "(depósito + 12 clientes), no sobre la red vial completa."
)

NODES, DIST = get_matrix()
DEPOT = NODES[0]
n = len(NODES)

with st.expander("📋 Matriz de distancias (km)", expanded=False):
    df = pd.DataFrame(
        [[DIST[(i, j)] for j in NODES] for i in NODES], index=NODES, columns=NODES
    )
    st.dataframe(df)

st.subheader("Modelo")
st.markdown(
    f"""
- **Nodos:** {n} (1 depósito + {n - 1} clientes)
- **Variables binarias** $x_{{ij}}$: {n * (n - 1)}
- **Variables continuas** $u_i$ (MTZ): {n}
- **Restricciones de salida/entrada única:** {2 * n}
- **Restricciones MTZ (anti-subciclo):** hasta {(n - 1) * (n - 2)}
- **Función objetivo:** minimizar la distancia total recorrida en el ciclo cerrado
"""
)

run = st.button("🚀 Resolver TSP", type="primary")

if run:
    with st.spinner("Resolviendo el modelo..."):
        res = solve_tsp(msg=False)

    if res["status"] != "Optimal":
        st.error(f"El solver no encontró una solución óptima. Estado: {res['status']}")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Distancia total óptima", f"{res['total']:.0f} km")
        col2.metric("Tiempo de resolución", f"{res['elapsed']:.2f} s")
        col3.metric("Estado", res["status"])

        st.subheader("🗺️ Secuencia de visita")
        st.success(" → ".join(str(s) for s in res["sequence"]))

        st.subheader("Arcos seleccionados")
        arcs_df = pd.DataFrame(
            res["arcs"], columns=["Desde", "Hasta", "Distancia (km)"]
        ).sort_values("Desde").reset_index(drop=True)
        st.dataframe(arcs_df, use_container_width=True)

        st.caption(
            f"Modelo MTZ: {res['n_vars']} variables, {res['n_constrs']} restricciones "
            "(ver detalle en `solver.py`)."
        )
else:
    st.info("Presiona **Resolver TSP** para correr el modelo exacto.")
