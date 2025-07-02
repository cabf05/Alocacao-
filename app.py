import streamlit as st
import pandas as pd
import pulp

def main():
    st.title("Sugestões de Alocação de Residentes")

    # Upload da planilha
    uploaded_file = st.file_uploader(
        "Faça upload da planilha (.csv ou .xlsx)", 
        type=["csv", "xlsx"]
    )
    if not uploaded_file:
        st.info("Aguardando upload da planilha...")
        return

    # Leitura do arquivo
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, parse_dates=["Date"])
    else:
        df = pd.read_excel(uploaded_file, parse_dates=["Date"])

    st.subheader("Dados Originais")
    st.dataframe(df)

    # Configuração básica
    residentes = df["Name"].unique()
    slots = df.index.tolist()

    # Para simplificar, consideramos todos disponíveis; depois você pode ajustar
    availability = {(r, i): 1 for r in residentes for i in slots}

    # Função genérica para resolver o modelo
    def solve_model(objective: str) -> pd.DataFrame:
        model = pulp.LpProblem("Alocacao", pulp.LpMinimize)
        # Variáveis binárias x[r][i]
        x = {
            (r, i): pulp.LpVariable(f"x_{r}_{i}", cat="Binary")
            for r in residentes for i in slots
        }

        # 1. Cada slot deve ter exatamente um residente
        for i in slots:
            model += (pulp.lpSum(x[r, i] for r in residentes) == 1)

        # 2. Só alocar se disponível
        for r in residentes:
            for i in slots:
                model += x[r, i] <= availability[(r, i)]

        # 3. Horas totais por residente
        hours = {
            r: pulp.lpSum(x[r, i] * df.loc[i, "Hours"] for i in slots)
            for r in residentes
        }

        # 4. Objetivo
        if objective == "fairness":
            # Minimizar diferença entre max e min de horas
            z_max = pulp.LpVariable("z_max", lowBound=0)
            z_min = pulp.LpVariable("z_min", lowBound=0)
            for r in residentes:
                model += hours[r] <= z_max
                model += hours[r] >= z_min
            model += z_max - z_min

        elif objective == "min_max":
            # Minimizar soma total de horas (exemplo alternativo)
            model += pulp.lpSum(hours[r] for r in residentes)

        elif objective == "elective_focus":
            # Priorizar slots de Elective
            bonus = {
                (r, i): 1 if "Elective" in str(df.loc[i, "Staff Type"]) else 0
                for r in residentes for i in slots
            }
            model += -pulp.lpSum(x[r, i] * bonus[r, i] for r in residentes for i in slots)

        # Resolver
        solver = pulp.PULP_CBC_CMD(msg=False)
        model.solve(solver)

        # Construir DataFrame de resultado
        result = []
        for i in slots:
            for r in residentes:
                if x[r, i].value() == 1:
                    result.append({
                        "Date": df.loc[i, "Date"].date(),
                        "Assignment": df.loc[i, "Assignment"],
                        "Resident": r,
                        "Hours": df.loc[i, "Hours"],
                        "Staff Type": df.loc[i, "Staff Type"]
                    })
        return pd.DataFrame(result)

    # Gera os 3 cenários
    st.subheader("Cenário 1: Minimizar Diferença de Carga (Fairness)")
    df1 = solve_model("fairness")
    st.dataframe(df1)

    st.subheader("Cenário 2: Minimizar Soma de Horas")
    df2 = solve_model("min_max")
    st.dataframe(df2)

    st.subheader("Cenário 3: Priorizar 'Elective'")
    df3 = solve_model("elective_focus")
    st.dataframe(df3)

if __name__ == "__main__":
    main()
