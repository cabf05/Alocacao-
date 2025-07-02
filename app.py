import streamlit as st
import pandas as pd
from datetime import datetime
from ortools.sat.python import cp_model

# ----------------------------
# 1) Funções de carregamento e pré-processamento com cache
# ----------------------------
@st.cache_data(show_spinner=False)
def load_data(uploaded_file):
    """Lê CSV ou Excel e retorna DataFrame com coluna Date em datetime."""
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, parse_dates=["Date"])
    else:
        df = pd.read_excel(
            uploaded_file,
            engine="openpyxl",
            parse_dates=["Date"]
        )
    # normalizar nome de colunas (só por precaução)
    df.columns = df.columns.str.strip()
    return df

@st.cache_data(show_spinner=False)
def filter_date_range(df, start_date, end_date):
    """Aplica filtro de data."""
    if start_date:
        df = df[df.Date >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.Date <= pd.to_datetime(end_date)]
    return df.reset_index(drop=True)

@st.cache_data(show_spinner=False)
def prepare_availability(df):
    """
    Cria:
      - lista de residentes,
      - lista de índices de slots,
      - dicionário availability só com pares (residente,slot) disponíveis.
    """
    residentes = list(df["Name"].unique())
    slots = df.index.tolist()
    availability = {}
    # exemplo de regra: todo residente está disponível, mas aqui
    # você pode filtrar por Staff Type ou outras regras
    for r in residentes:
        for i in slots:
            availability[(r, i)] = True
    return residentes, slots, availability

# ----------------------------
# 2) Modelo CP-SAT esparso
# ----------------------------
def solve_model(df, residentes, slots, availability, objective):
    """
    Monta e resolve o modelo CP-SAT com variáveis esparsas.
    objective: "fairness", "min_hours", ou "elective_focus"
    """
    model = cp_model.CpModel()

    # 2.1) Variáveis binárias apenas onde availability == True
    x = {}
    for (r, i), avail in availability.items():
        if avail:
            x[(r, i)] = model.NewBoolVar(f"x_{r}_{i}")

    # 2.2) Cada slot é coberto exatamente uma vez
    for i in slots:
        vars_slot = [x[(r, i)] for r in residentes if (r, i) in x]
        model.Add(sum(vars_slot) == 1)

    # 2.3) Precomputar horas e bônus por (r,i)
    hours = { (r, i): int(df.at[i, "Hours"]) for (r, i) in x }
    elective_bonus = {
        (r, i): 1 if "Elective" in str(df.at[i, "Staff Type"]) else 0
        for (r, i) in x
    }

    # 2.4) Funções de custo / equilíbrio
    # total_hours[r] = soma de horas alocadas a r
    total_hours = {}
    for r in residentes:
        terms = []
        for i in slots:
            if (r, i) in x:
                terms.append(x[(r, i)] * hours[(r, i)])
        total_hours[r] = model.NewIntVar(0, sum(hours.values()), f"hrs_{r}")
        model.Add(total_hours[r] == sum(terms))

    if objective == "fairness":
        # minimize max(r) - min(r)
        z_max = model.NewIntVar(0, sum(hours.values()), "z_max")
        z_min = model.NewIntVar(0, sum(hours.values()), "z_min")
        for r in residentes:
            model.Add(total_hours[r] <= z_max)
            model.Add(total_hours[r] >= z_min)
        model.Minimize(z_max - z_min)

    elif objective == "min_hours":
        # minimize soma total de horas
        model.Minimize(sum(total_hours[r] for r in residentes))

    elif objective == "elective_focus":
        # maximize alocações em Elective → minimize -bonus
        term = []
        for (r, i), var in x.items():
            term.append(var * elective_bonus[(r, i)])
        model.Minimize(-sum(term))

    else:
        raise ValueError(f"Objective '{objective}' não reconhecido")

    # 2.5) Resolver com parâmetros para velocidade
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 20    # limite de tempo
    solver.parameters.num_search_workers = 8      # threads paralelas
    solver.parameters.maximize = False            # garante MIP
    status = solver.Solve(model)

    # 2.6) Extrair solução para DataFrame
    results = []
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for (r, i), var in x.items():
            if solver.Value(var) == 1:
                row = df.loc[i]
                results.append({
                    "Date":    row["Date"].date(),
                    "Assignment": row["Assignment"],
                    "Resident":  r,
                    "Hours":     row["Hours"],
                    "Staff Type": row["Staff Type"]
                })
    return pd.DataFrame(results)

# ----------------------------
# 3) Streamlit App
# ----------------------------
def main():
    st.title("🔄 Sugestões Ágeis de Alocação de Residentes")

    uploaded = st.file_uploader("Faça upload (.csv / .xlsx)", type=["csv","xlsx"])
    if not uploaded:
        st.info("Aguardando upload da planilha...")
        return

    df = load_data(uploaded)

    # filtro de datas
    min_date = df.Date.min().date()
    max_date = df.Date.max().date()
    start_date, end_date = st.date_input(
        "Período de interesse",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    df = filter_date_range(df, start_date, end_date)
    st.subheader("📋 Dados Filtrados")
    st.dataframe(df)

    # disponibilidade esparsa
    residentes, slots, availability = prepare_availability(df)

    # gerar cenários em cache separado
    if st.button("🔢 Gerar 3 Cenários"):
        with st.spinner("Resolvendo cenários..."):
            df1 = solve_model(df, residentes, slots, availability, "fairness")
            df2 = solve_model(df, residentes, slots, availability, "min_hours")
            df3 = solve_model(df, residentes, slots, availability, "elective_focus")

        st.subheader("Cenário 1 – Minimizar Diferença de Carga (Fairness)")
        st.dataframe(df1)

        st.subheader("Cenário 2 – Minimizar Soma Total de Horas")
        st.dataframe(df2)

        st.subheader("Cenário 3 – Priorizar 'Elective'")
        st.dataframe(df3)

if __name__ == "__main__":
    main()
