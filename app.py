# -------------------------------------------------------------
#  📊 DASHBOARD FINANCIERO AVANZADO
#      (ROIC & EVA alineados con GuruFocus/Finviz)
# -------------------------------------------------------------
import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta

# -------------------------------------------------------------
# ⚙️ Configuración global de la página
# -------------------------------------------------------------
st.set_page_config(
    page_title="📊 Dashboard Financiero Avanzado",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------
# Parámetros WACC por defecto (ajustables en el sidebar)
# -------------------------------------------------------------
Rf = 0.0435   # Tasa libre de riesgo
Rm = 0.085    # Retorno esperado del mercado
Tc = 0.21     # Tasa impositiva corporativa

# =============================================================
# 1. FUNCIONES AUXILIARES
# =============================================================
def safe_first(obj):
    """Devuelve el primer valor no nulo de una serie o el propio escalar."""
    if obj is None:
        return None
    if hasattr(obj, "dropna"):
        obj = obj.dropna()
        return obj.iloc[0] if not obj.empty else None
    return obj

def get_cash_equivalents(bs, info):
    """Localiza efectivo y equivalentes en el balance o en .info."""
    for k in [
        "Cash And Cash Equivalents",
        "Cash And Cash Equivalents At Carrying Value",
        "Cash Cash Equivalents And Short Term Investments",
    ]:
        if k in bs.index:
            return bs.loc[k]
    return pd.Series([info.get("totalCash")], index=bs.columns[:1])

def get_ebit(tkr):
    """Obtiene EBIT (o equivalente) desde diferentes estados."""
    keys = ["EBIT", "Operating Income", "Earnings Before Interest and Taxes"]
    for k in keys:
        if k in tkr.financials.index:
            return tkr.financials.loc[k]
        if k in tkr.income_stmt.index:
            return tkr.income_stmt.loc[k]
    # Último recurso: campo directo
    return pd.Series([tkr.info.get("ebit")], index=tkr.financials.columns[:1])

def invested_capital_avg(debt, equity, cash_eq):
    """Promedio de 2 años de (Deuda + Equity – Cash & Equivalents)."""
    def ic(i):
        return (debt.iloc[i] or 0) + (equity.iloc[i] or 0) - (cash_eq.iloc[i] or 0)
    current = ic(0)
    previous = ic(1) if len(debt) > 1 else current
    return (current + previous) / 2 or None

def calcular_wacc(info, total_debt):
    """WACC clásico usando CAPM + coste de deuda implícito."""
    beta  = info.get("beta", 1.0)
    price = info.get("currentPrice")
    shares = info.get("sharesOutstanding")
    market_cap = price * shares if price and shares else 0

    Re = Rf + beta * (Rm - Rf)
    Rd = 0.055 if total_debt else 0

    if market_cap + total_debt == 0:
        return None

    return (market_cap / (market_cap + total_debt)) * Re + \
           (total_debt / (market_cap + total_debt)) * Rd * (1 - Tc)

def calcular_crecimiento_historico(financials, metric):
    """CAGR a 4 periodos si hay datos suficientes."""
    if metric not in financials.index:
        return None
    datos = financials.loc[metric].dropna().iloc[:4]
    if len(datos) < 2 or datos.iloc[-1] == 0:
        return None
    años = len(datos) - 1
    return (datos.iloc[0] / datos.iloc[-1]) ** (1 / años) - 1

# =============================================================
# 2. OBTENCIÓN DE DATOS POR EMPRESA
# =============================================================
def obtener_datos_financieros(ticker):
    try:
        stock = yf.Ticker(ticker)
        info  = stock.info
        bs    = stock.balance_sheet

        # ---- Validaciones mínimas --------------------------------------
        if not info or bs.empty:
            raise ValueError("info o balance_sheet vacío")

        fin   = stock.financials
        cf    = stock.cashflow

        # ---- EBIT & NOPAT ---------------------------------------------
        ebit_series = get_ebit(stock)
        nopat = safe_first(ebit_series)
        if nopat is not None:
            nopat *= (1 - Tc)

        # ---- Capital invertido promedio (2 años, neto de efectivo) ----
        debt_series = (
            bs.loc["Total Debt"] if "Total Debt" in bs.index
            else bs.loc.get("Long Term Debt", 0) + bs.loc.get("Short Term Debt", 0)
        )
        equity_series = (
            bs.loc["Total Stockholder Equity"]
            if "Total Stockholder Equity" in bs.index
            else pd.Series([info.get("totalStockholderEquity")], index=bs.columns[:1])
        )
        cash_series = get_cash_equivalents(bs, info)
        invested_capital = invested_capital_avg(debt_series, equity_series, cash_series)

        # ---- ROIC ------------------------------------------------------
        roic = nopat / invested_capital if (nopat is not None and invested_capital) else None

        # ---- WACC ------------------------------------------------------
        total_debt_now = safe_first(debt_series) or info.get("totalDebt") or 0
        wacc = calcular_wacc(info, total_debt_now)

        # ---- EVA -------------------------------------------------------
        eva = (roic - wacc) * invested_capital if all(
            v is not None for v in [roic, wacc, invested_capital]
        ) else None

        # ---- Otros ratios / datos -------------------------------------
        price  = info.get("currentPrice")
        pfcf   = None
        fcf    = cf.loc["Free Cash Flow"].iloc[0] if "Free Cash Flow" in cf.index else None
        shares = info.get("sharesOutstanding")
        if fcf and shares:
            pfcf = price / (fcf / shares)

        revenue_growth = calcular_crecimiento_historico(fin, "Total Revenue")
        eps_growth     = calcular_crecimiento_historico(fin, "Net Income")
        fcf_growth     = calcular_crecimiento_historico(cf, "Free Cash Flow") or \
                         calcular_crecimiento_historico(cf, "Operating Cash Flow")

        cash_ratio = info.get("cashRatio")
        ocf = cf.loc["Operating Cash Flow"].iloc[0] if "Operating Cash Flow" in cf.index else None
        current_liab = bs.loc["Total Current Liabilities"].iloc[0] \
                       if "Total Current Liabilities" in bs.index else None
        cash_flow_ratio = (ocf / current_liab) if (ocf and current_liab) else None

        return {
            # --- básicos ---
            "Ticker": ticker,
            "Nombre": info.get("longName", ticker),
            "Sector": info.get("sector", "N/D"),
            "País":   info.get("country", "N/D"),
            "Industria": info.get("industry", "N/D"),
            "Precio": price,

            # --- valoración y dividendo ---
            "P/E": info.get("trailingPE"),
            "P/B": info.get("priceToBook"),
            "P/FCF": pfcf,
            "Dividend Year": info.get("dividendRate"),
            "Dividend Yield %": info.get("dividendYield"),
            "Payout Ratio": info.get("payoutRatio"),

            # --- rentabilidad ---
            "ROA": info.get("returnOnAssets"),
            "ROE": info.get("returnOnEquity"),

            # --- liquidez y deuda ---
            "Current Ratio": info.get("currentRatio"),
            "Quick Ratio":   info.get("quickRatio"),
            "LtDebt/Eq": info.get("longTermDebtToEquity"),
            "Debt/Eq":  info.get("debtToEquity"),

            # --- márgenes ---
            "Oper Margin":   info.get("operatingMargins"),
            "Profit Margin": info.get("profitMargins"),

            # --- métricas avanzadas ---
            "WACC": wacc,
            "ROIC": roic,
            "EVA":  eva,

            # --- crecimientos ---
            "Revenue Growth": revenue_growth,
            "EPS Growth":     eps_growth,
            "FCF Growth":     fcf_growth,

            # --- otros ---
            "Cash Ratio": cash_ratio,
            "Cash Flow Ratio": cash_flow_ratio,
            "Operating Cash Flow": ocf,
            "Current Liabilities": current_liab,
        }
    except Exception as e:
        return {"Ticker": ticker, "Error": str(e)}

# =============================================================
# 3. INTERFAZ PRINCIPAL
# =============================================================
def main():
    st.title("📊 Dashboard de Análisis Financiero Avanzado")

    # -------------- Sidebar -----------------------------------
    with st.sidebar:
        st.header("⚙️ Configuración")
        tickers_input = st.text_area("🔎 Ingresa tickers (coma)", "HRL, AAPL, MSFT")
        max_tickers   = st.slider("Número máximo de tickers", 1, 50, 20)

        st.markdown("---")
        global Rf, Rm, Tc
        Rf = st.number_input("Tasa libre de riesgo (%)", 0.0, 20.0, 4.35) / 100
        Rm = st.number_input("Retorno esperado del mercado (%)", 0.0, 30.0, 8.5) / 100
        Tc = st.number_input("Tasa impositiva corporativa (%)", 0.0, 50.0, 21.0) / 100

    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()][:max_tickers]

    # -------------- Botón de ejecución -------------------------
    if st.button("🔍 Analizar Acciones", type="primary"):
        if not tickers:
            st.warning("Por favor ingresa al menos un ticker")
            return

        resultados, errores = {}, {}
        barra = st.progress(0)
        for i, tk in enumerate(tickers, 1):
            data = obtener_datos_financieros(tk)
            if "Error" in data:
                errores[tk] = data["Error"]
            else:
                resultados[tk] = data
            barra.progress(i / len(tickers))
            time.sleep(1)
        barra.empty()

        if not resultados:
            st.error("No se pudo obtener datos válidos para ningún ticker")
            if errores:
                st.subheader("Errores detectados")
                st.table(pd.DataFrame([{"Ticker": k, "Error": v} for k, v in errores.items()]))
            return

        df = pd.DataFrame(resultados.values())

        # formateo %
        for col in [
            "Dividend Yield %", "Payout Ratio", "ROA", "ROE",
            "Oper Margin", "Profit Margin", "WACC", "ROIC"
        ]:
            df[col] = df[col].apply(lambda x: f"{x*100:,.2f}%" if pd.notnull(x) else "N/D")

        # -------------- Sección 1: Resumen ----------------------
        st.header("📋 Resumen General")
        columnas_mostrar = [
            "Ticker", "Nombre", "Sector", "Precio", "P/E", "P/B", "P/FCF",
            "Dividend Yield %", "Payout Ratio", "ROA", "ROE", "Current Ratio",
            "Debt/Eq", "Oper Margin", "Profit Margin", "WACC", "ROIC", "EVA"
        ]
        st.dataframe(df[columnas_mostrar].dropna(how='all', axis=1),
                     use_container_width=True, height=400)

        # Mostrar errores si existieran
        if errores:
            st.subheader("🚫 Tickers con error")
            st.table(pd.DataFrame([{"Ticker": k, "Error": v} for k, v in errores.items()]))

        # -------------- Sección 2: Valoración -------------------
        st.header("💰 Análisis de Valoración")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Ratios de Valoración")
            fig, ax = plt.subplots(figsize=(10, 4))
            df[["Ticker", "P/E", "P/B", "P/FCF"]].set_index("Ticker")\
              .apply(pd.to_numeric, errors='coerce').plot(kind="bar", ax=ax, rot=45)
            ax.set_ylabel("Ratio")
            st.pyplot(fig); plt.close()

        with col2:
            st.subheader("Dividend Yield (%)")
            fig, ax = plt.subplots(figsize=(10, 4))
            dy = df[["Ticker", "Dividend Yield %"]].replace("N/D", 0)
            dy["Dividend Yield %"] = dy["Dividend Yield %"].str.rstrip("%").astype(float)
            dy.set_index("Ticker").plot(kind="bar", ax=ax, rot=45)
            ax.set_ylabel("%")
            st.pyplot(fig); plt.close()

        # -------------- Sección 3: Rentabilidad -----------------
        st.header("📈 Rentabilidad y Eficiencia")
        tabs = st.tabs(["ROE vs ROA", "Márgenes", "WACC vs ROIC"])

        with tabs[0]:
            fig, ax = plt.subplots(figsize=(10, 5))
            rr = df[["Ticker", "ROE", "ROA"]].replace("N/D", 0)
            rr["ROE"] = rr["ROE"].str.rstrip("%").astype(float)
            rr["ROA"] = rr["ROA"].str.rstrip("%").astype(float)
            rr.set_index("Ticker").plot(kind="bar", ax=ax, rot=45)
            ax.set_ylabel("%")
            st.pyplot(fig); plt.close()

        with tabs[1]:
            fig, ax = plt.subplots(figsize=(10, 5))
            mm = df[["Ticker", "Oper Margin", "Profit Margin"]].replace("N/D", 0)
            mm["Oper Margin"]   = mm["Oper Margin"].str.rstrip("%").astype(float)
            mm["Profit Margin"] = mm["Profit Margin"].str.rstrip("%").astype(float)
            mm.set_index("Ticker").plot(kind="bar", ax=ax, rot=45)
            ax.set_ylabel("%")
            st.pyplot(fig); plt.close()

        with tabs[2]:
            fig, ax = plt.subplots(figsize=(10, 5))
            for _, r in df.iterrows():
                w = float(r["WACC"].rstrip("%")) if r["WACC"] != "N/D" else None
                rt= float(r["ROIC"].rstrip("%")) if r["ROIC"] != "N/D" else None
                if w is not None and rt is not None:
                    col = "green" if rt > w else "red"
                    ax.bar(r["Ticker"], rt, color=col, alpha=0.6)
                    ax.bar(r["Ticker"], w, color="gray", alpha=0.3)
            ax.set_ylabel("%")
            ax.set_title("ROIC vs WACC")
            st.pyplot(fig); plt.close()

        # -------------- Sección 4: Deuda & Liquidez -------------
        st.header("🏦 Deuda y Liquidez")
        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Apalancamiento")
            fig, ax = plt.subplots(figsize=(9, 4))
            df[["Ticker","Debt/Eq","LtDebt/Eq"]].set_index("Ticker")\
              .apply(pd.to_numeric, errors="coerce").plot(kind="bar", stacked=True, ax=ax, rot=45)
            ax.axhline(1, color="red", linestyle="--")
            st.pyplot(fig); plt.close()

        with col4:
            st.subheader("Liquidez")
            fig, ax = plt.subplots(figsize=(9, 4))
            df[["Ticker","Current Ratio","Quick Ratio"]].set_index("Ticker")\
              .apply(pd.to_numeric, errors="coerce").plot(kind="bar", ax=ax, rot=45)
            ax.axhline(1, color="green", linestyle="--")
            st.pyplot(fig); plt.close()

        # -------------- Sección 5: Crecimiento ------------------
        st.header("🚀 Crecimiento (CAGR 3-4 años)")
        growth_df = df.set_index("Ticker")[["Revenue Growth","EPS Growth","FCF Growth"]] * 100
        fig, ax = plt.subplots(figsize=(12, 6))
        growth_df.plot(kind="bar", ax=ax, rot=45)
        ax.axhline(0, color="black"); ax.set_ylabel("%")
        st.pyplot(fig); plt.close()

        # -------------- Sección 6: Análisis individual ----------
        st.header("🔍 Análisis Individual")
        pick = st.selectbox("Selecciona empresa", df["Ticker"].unique())
        det = df[df["Ticker"] == pick].iloc[0]

        cA,cB,cC = st.columns(3)
        with cA:
            st.metric("Precio", f"${det['Precio']:,.2f}" if det['Precio'] else "N/D")
            st.metric("P/E", det["P/E"]); st.metric("P/B", det["P/B"])
        with cB:
            st.metric("ROIC", det["ROIC"]); st.metric("WACC", det["WACC"])
            st.metric("EVA", f"{det['EVA']:,.0f}" if pd.notnull(det['EVA']) else "N/D")
        with cC:
            st.metric("ROE", det["ROE"]); st.metric("Dividend Yield", det["Dividend Yield %"])
            st.metric("Debt/Eq", det["Debt/Eq"])

        st.subheader("ROIC vs WACC")
        if det["ROIC"] != "N/D" and det["WACC"] != "N/D":
            r_val = float(det["ROIC"].rstrip("%"))
            w_val = float(det["WACC"].rstrip("%"))
            fig, ax = plt.subplots(figsize=(4,3))
            ax.bar(["ROIC","WACC"], [r_val, w_val],
                   color=["green" if r_val > w_val else "red", "gray"])
            ax.set_ylabel("%")
            st.pyplot(fig)
            st.success("✅ Crea valor" if r_val > w_val else "❌ Destruye valor")
        else:
            st.info("Datos insuficientes para comparar ROIC/WACC")

# -------------------------------------------------------------
# Punto de entrada
# -------------------------------------------------------------
if __name__ == "__main__":
    main()
