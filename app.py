import streamlit as st
import pandas as pd
from datetime import datetime
from ortools.sat.python import cp_model

# ----------------------------
# Utilitários com cache
# ----------------------------
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, parse_dates=["Date"])
    else:
        df = pd.read_excel(uploaded_file, engine="openpyxl", parse_dates=["Date"])
    df.columns = df.columns.str.strip()
    return df

@st.cache_data
def pick_slots(df):
    """Retorna índices de slots de JC/MKSAP (segunda) e MR (sexta) conforme regras."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    # Journal Club: segundas entre set/jun, toda 2ª semana
    start_jc = datetime(df.Date.dt.year.min(), 9, 1)
    end_jc   = datetime(df.Date.dt.year.min()+1, 6, 30)
    mondays = df[(df.Date.dt.weekday == 0) &
                 (df.Date >= start_jc) & (df.Date <= end_jc)].sort_values("Date")
    # pegar alternadas, máximo 9
    jc_idxs = mondays.iloc[::2].index.tolist()[:9]

    # Cardiology Report: sextas a partir de 15/08 até 30/06, toda 2ª semana
    start_mr = datetime(df.Date.dt.year.min(), 8, 15)
    end_mr   = end_jc
    fridays = df[(df.Date.dt.weekday == 4) &
                 (df.Date >= start_mr) & (df.Date <= end_mr)].sort_values("Date")
    mr_idxs = fridays.iloc[::2].index.tolist()[:18]

    return jc_idxs, mr_idxs

# ----------------------------
# Montagem e solução ILP (Pulp)
# ----------------------------
def solve_schedule(df, jc_idxs, mr_idxs):
    import pulp
    R = df["Name"].unique().tolist()
    S = jc_idxs + mr_idxs

    # disponibilidade simples: r disponível em todos os S (ajuste se quiser)
    avail = {(r, s): True for r in R for s in S}

    # modelo
    model = pulp.LpProblem("Schedule", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", (R, S), cat="Binary")

    # cada slot = 1 residente
    for s in S:
        model += pulp.lpSum(x[r][s] for r in R) == 1

    # só se disponível
    for r in R:
        for s in S:
            model += x[r][s] <= avail[r, s]

    # horas totais
    hours = {r: pulp.lpSum(x[r][s] * df.at[s, "Hours"] for s in S) for r in R}

    # força 3 com ambos JC e MR
    y = pulp.LpVariable.dicts("y", R, cat="Binary")
    for r in R:
        model += y[r] <= pulp.lpSum(x[r][s] for s in jc_idxs)
        model += y[r] <= pulp.lpSum(x[r][s] for s in mr_idxs)
        model += y[r] >= pulp.lpSum(x[r][s] for s in jc_idxs) + \
                        pulp.lpSum(x[r][s] for s in mr_idxs) - 1
    model += pulp.lpSum(y[r] for r in R) == 3

    # objetivo: minimizar diferença de carga
    z_max = pulp.LpVariable("z_max", lowBound=0)
    z_min = pulp.LpVariable("z_min", lowBound=0)
    for r in R:
        model += hours[r] <= z_max
        model += hours[r] >= z_min
    model += z_max - z_min

    # resolve
    model.solve(pulp.PULP_CBC_CMD(msg=False))

    # extrai
    out = []
    for s in S:
        for r in R:
            if x[r][s].value() == 1:
                row = df.loc[s]
                out.append({
                    "Date":       row["Date"].date(),
                    "Assignment": row["Assignment"],
                    "Resident":   r,
                    "JC?" :       s in jc_idxs,
                    "MR?":        s in mr_idxs,
                    "Hours":      row["Hours"]
                })
    return pd.DataFrame(out).sort_values("Date")

# ----------------------------
# App Streamlit
# ----------------------------
st.title("📅 Alocação Focada: JC & Cardiology Reports")

uploaded = st.file_uploader("Upload (.csv / .xlsx)", type=["csv","xlsx"])
if uploaded:
    df = load_data(uploaded)
    jc_idxs, mr_idxs = pick_slots(df)
    if st.button("Gerar Melhor Agenda"):
        with st.spinner("Calculando..."):
            schedule = solve_schedule(df, jc_idxs, mr_idxs)
        st.subheader("Agenda Ótima")
        st.dataframe(schedule)
        st.markdown(f"- **Total de slots:** {len(jc_idxs)} JC + {len(mr_idxs)} MR = {len(jc_idxs)+len(mr_idxs)}")
        st.markdown("- 3 residentes apresentarão ambos JC e MR.")
