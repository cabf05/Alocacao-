import streamlit as st
import pandas as pd
import datetime
from io import StringIO
import pulp
import networkx as nx

def parse_schedule(text_input):
    data = []
    for line in text_input.strip().split("\n"):
        parts = line.strip().split()
        if len(parts) >= 6:
            year = int(parts[0])
            day = parts[1]
            weekday = parts[2]
            resident = parts[3]
            intern = parts[4]
            try:
                date = datetime.datetime.strptime(f"{day}-{year}", "%d-%b-%Y")
                data.append({
                    "date": date,
                    "weekday": weekday,
                    "resident": resident,
                    "intern": intern
                })
            except:
                continue
    df = pd.DataFrame(data)
    df = df.sort_values("date").reset_index(drop=True)
    return df

def build_weeks(df):
    schedule = []
    for i in range(len(df)):
        row = df.iloc[i]
        if row["weekday"] == "Tue":
            mon_intern = None
            if i > 0 and df.iloc[i - 1]["weekday"] == "Mon":
                mon_intern = df.iloc[i - 1]["intern"]
            schedule.append({
                "week": row["date"].strftime("%Y-%m-%d"),
                "tue_date": row["date"],
                "resident": row["resident"],
                "intern_tue": row["intern"],
                "intern_mon": mon_intern
            })
    return pd.DataFrame(schedule)

def solve_ilp(schedule):
    interns = set(schedule["intern_tue"]).union(schedule["intern_mon"].dropna())
    intern_list = list(interns)
    week_list = list(schedule["week"])

    prob = pulp.LpProblem("Intern_Scheduling", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", ((w, i) for w in week_list for i in intern_list), cat="Binary")

    for w in week_list:
        prob += pulp.lpSum([x[(w, i)] for i in intern_list]) == 1

    for idx, row in schedule.iterrows():
        w = row["week"]
        allowed = {row["intern_tue"]}
        if row["intern_mon"]:
            allowed.add(row["intern_mon"])
        for i in intern_list:
            if i not in allowed:
                prob += x[(w, i)] == 0

    for i in intern_list:
        for w1, w2 in zip(week_list[:-1], week_list[1:]):
            prob += x[(w1, i)] + x[(w2, i)] <= 1

    for i in intern_list:
        prob += pulp.lpSum([x[(w, i)] for w in week_list]) <= 3

    overloads = {i: pulp.LpVariable(f"over_{i}", cat="Binary") for i in intern_list}
    for i in intern_list:
        prob += pulp.lpSum([x[(w, i)] for w in week_list]) - 2 <= overloads[i]
    prob += pulp.lpSum([overloads[i] for i in intern_list])

    prob.solve()

    result = []
    for idx, row in schedule.iterrows():
        for i in intern_list:
            if pulp.value(x[(row["week"], i)]) == 1:
                result.append({
                    "week": row["week"],
                    "tuesday": row["tue_date"].strftime("%d-%b-%Y"),
                    "resident": row["resident"],
                    "intern": i
                })
                break
    return pd.DataFrame(result)

def solve_min_cost_flow(schedule):
    G = nx.DiGraph()
    source = "S"
    sink = "T"
    interns = list(set(schedule["intern_tue"]).union(schedule["intern_mon"].dropna()))
    weeks = list(schedule["week"])

    for intern in interns:
        G.add_edge(source, intern, capacity=3, weight=0)

    for idx, row in schedule.iterrows():
        w = row["week"]
        allowed = {row["intern_tue"]}
        if row["intern_mon"]:
            allowed.add(row["intern_mon"])
        G.add_edge(w, sink, capacity=1, weight=0)
        for i in allowed:
            penalty = 1
            G.add_edge(i, w, capacity=1, weight=penalty)

    flowDict = nx.max_flow_min_cost(G, source, sink)
    result = []
    for idx, row in schedule.iterrows():
        for i in interns:
            if i in flowDict and row["week"] in flowDict[i] and flowDict[i][row["week"]] == 1:
                result.append({
                    "week": row["week"],
                    "tuesday": row["tue_date"].strftime("%d-%b-%Y"),
                    "resident": row["resident"],
                    "intern": i
                })
    return pd.DataFrame(result)

# App UI
st.set_page_config(layout="wide")
st.title("Intern On-Call Optimizer")

tab1, tab2 = st.tabs(["0-1 ILP Optimization", "Min-Cost Flow"])

with tab1:
    st.header("Otimizador 0-1 ILP")
    input_text = st.text_area("Cole aqui os dados da escala:", height=400)
    if st.button("Executar ILP"):
        df = parse_schedule(input_text)
        schedule = build_weeks(df)
        result_df = solve_ilp(schedule)
        st.success("Alocação gerada com sucesso!")
        st.dataframe(result_df)

with tab2:
    st.header("Otimizador Min-Cost Flow")
    input_text2 = st.text_area("Cole aqui os dados da escala:", key="flow", height=400)
    if st.button("Executar Min-Cost Flow"):
        df2 = parse_schedule(input_text2)
        schedule2 = build_weeks(df2)
        result_df2 = solve_min_cost_flow(schedule2)
        st.success("Alocação gerada com sucesso!")
        st.dataframe(result_df2)
