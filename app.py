import streamlit as st
import pandas as pd
import datetime
from collections import defaultdict
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpBinary, LpStatus, value
import networkx as nx


def parse_schedule(text_input):
    data = []
    lines = text_input.strip().split("\n")

    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 6:
            try:
                year = parts[0]
                day = parts[1]
                weekday = parts[2]
                resident = parts[3]
                intern = parts[4]
                date = datetime.datetime.strptime(f"{day}-{year}", "%d-%b-%Y")
                data.append({
                    "date": date,
                    "weekday": weekday,
                    "resident": resident,
                    "intern": intern
                })
            except Exception:
                continue  # ignora linha com erro

    df = pd.DataFrame(data)
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)
    return df


def get_unique_interns(df):
    return sorted(df["intern"].unique())


def build_ilp_model(df, interns):
    weeks = list(df.index)
    intern_indices = {intern: i for i, intern in enumerate(interns)}

    # Variáveis binárias: x[w][i] = 1 se intern i for alocado na semana w
    x = [[LpVariable(f"x_{w}_{i}", cat=LpBinary) for i in range(len(interns))] for w in weeks]

    model = LpProblem("Intern_Assignment", LpMinimize)

    # Objetivo: minimizar total de internos com 3 alocações
    total_assignments = [lpSum(x[w][i] for w in weeks) for i in range(len(interns))]
    model += lpSum([(assignments - 2)**2 for assignments in total_assignments])  # penaliza ter 3+

    # Cada semana deve ter exatamente 1 interno
    for w in weeks:
        model += lpSum(x[w][i] for i in range(len(interns))) == 1

    # Nenhum interno pode ser alocado 2 semanas consecutivas
    for i in range(len(interns)):
        for w in range(len(weeks) - 1):
            model += x[w][i] + x[w+1][i] <= 1

    # No máximo 3 alocações por interno
    for i in range(len(interns)):
        model += lpSum(x[w][i] for w in weeks) <= 3

    return model, x, intern_indices


def solve_ilp(df):
    interns = get_unique_interns(df)
    model, x, intern_indices = build_ilp_model(df, interns)
    model.solve()

    allocation = []
    for w in range(len(df)):
        for i, intern in enumerate(interns):
            if x[w][i].varValue == 1:
                allocation.append(intern)
                break
    df["ILP Assignment"] = allocation
    return df


def solve_min_cost_flow(df):
    interns = get_unique_interns(df)
    weeks = list(df.index)

    G = nx.DiGraph()

    for intern in interns:
        G.add_edge("source", intern, capacity=3, weight=0)

    for w in weeks:
        node = f"week_{w}"
        G.add_edge(node, "sink", capacity=1, weight=0)
        for intern in interns:
            prev_week = df.loc[w - 1, "intern"] if w > 0 else None
            if w > 0 and df["intern"].iloc[w - 1] == intern:
                continue
            G.add_edge(intern, node, capacity=1, weight=0)

    flow_dict = nx.max_flow_min_cost(G, "source", "sink")

    assignment = []
    for w in weeks:
        node = f"week_{w}"
        chosen = None
        for intern in interns:
            if intern in flow_dict and node in flow_dict[intern] and flow_dict[intern][node] == 1:
                chosen = intern
                break
        assignment.append(chosen if chosen else "N/A")

    df["Flow Assignment"] = assignment
    return df


# ===== Streamlit App =====
st.set_page_config(page_title="Intern Allocation Optimizer", layout="wide")
st.title("🧠 Intern Allocation Optimizer")
st.markdown("Cole abaixo a escala de semanas. O sistema oferece duas soluções: 0-1 ILP e Min-Cost Flow.")

input_text = st.text_area("📋 Schedule Input (cole os dados aqui)", height=400)

if input_text.strip():
    try:
        df = parse_schedule(input_text)

        tab1, tab2 = st.tabs(["🔢 0-1 ILP", "🔗 Min-Cost Flow"])

        with tab1:
            st.header("📌 Solução via Programação Inteira (ILP)")
            ilp_result = solve_ilp(df.copy())
            st.dataframe(ilp_result[["date", "weekday", "ILP Assignment"]], use_container_width=True)

        with tab2:
            st.header("📌 Solução via Min-Cost Flow")
            flow_result = solve_min_cost_flow(df.copy())
            st.dataframe(flow_result[["date", "weekday", "Flow Assignment"]], use_container_width=True)

    except Exception as e:
        st.error(f"Erro ao processar dados: {e}")
