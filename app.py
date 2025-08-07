# -------------------------------------------------------------
#  📊 DASHBOARD FINANCIERO AVANZADO
#      ROIC y WACC con Kd y tasa efectiva por empresa
# -------------------------------------------------------------
import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import time

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
# Parámetros CAPM por defecto (editables en el sidebar)
# -------------------------------------------------------------
Rf  = 0.0435   # tasa libre de riesgo
Rm  = 0.085    # retorno esperado del mercado (= Rf + prima)
Tc0 = 0.21     # tasa impositiva por defecto

# =============================================================
# 1. FUNCIONES AUXILIARES
# =============================================================
def safe_first(obj):
    if obj is None: 
        return None
    if hasattr(obj, "dropna"):
        obj = obj.dropna()
    return obj.iloc[0] if hasattr(obj, "iloc") and not obj.empty else obj

def seek_row(df, keys):
    """Devuelve la primera fila que coincida con alguna clave; serie cero si no existe."""
    for k in keys:
        if k in df.index:
            return df.loc[k]
    return pd.Series([0], index=df.columns[:1])

def calc_ke(beta):
    """Costo del equity vía CAPM."""
    return Rf + beta * (Rm - Rf)

def calc_kd(interest, debt):
    """Costo de la deuda según gastos por intereses."""
    return interest / debt if debt else 0

def calc_wacc(mcap, debt, ke, kd, tax_rate):
    total = mcap + debt
    if total == 0:
        return None
    return (mcap / total) * ke + (debt / total) * kd * (1 - tax_rate)

def cagr4(fin, metric):
    """CAGR de 3-4 años si hay datos suficientes."""
    if metric not in fin.index:
        return None
    vals = fin.loc[metric].dropna().iloc[:4]
    if len(vals) < 2 or vals.iloc[-1] == 0:
        return None
    return (vals.iloc[0] / vals.iloc[-1]) ** (1 / (len(vals) - 1)) - 1

# =============================================================
# 2. OBTENCIÓN DE DATOS POR EMPRESA
# =============================================================
def obtener_datos_financieros(ticker, Tc_def):
    tkr  = yf.Ticker(ticker)
    info = tkr.info
    bs   = tkr.balance_sheet
    fin  = tkr.financials
    cf   = tkr.cashflow

    if not info or bs.empty:
        raise ValueError("Información o balance_sheet vacío")

    # --------- Elementos de capital -----------------------------------
    beta  = info.get("beta", 1)
    ke    = calc_ke(beta)

    debt_series = seek_row(bs, ["Total Debt", "Long Term Debt"])
    debt_now    = safe_first(debt_series) or info.get("totalDebt") or 0

    cash_series   = seek_row(bs, ["Cash And Cash Equivalents",
                                  "Cash And Cash Equivalents At Carrying Value",
                                  "Cash Cash Equivalents And Short Term Investments"])
    equity_series = seek_row(bs, ["Common Stock Equity", "Total Stockholder Equity"])

    # --------- Estado de resultados -----------------------------------
    interest_exp = safe_first(seek_row(fin, ["Interest Expense"]))
    ebt          = safe_first(seek_row(fin, ["Ebt", "EBT"]))
    tax_exp      = safe_first(seek_row(fin, ["Income Tax Expense"]))
    ebit         = safe_first(seek_row(fin, ["EBIT", "Operating Income",
                                             "Earnings Before Interest and Taxes"]))

    kd        = calc_kd(interest_exp, debt_now)
    tax_rate  = tax_exp / ebt if ebt else Tc_def
    mcap      = info.get("marketCap", 0)
    wacc      = calc_wacc(mcap, debt_now, ke, kd, tax_rate)

    # --------- ROIC & EVA ---------------------------------------------
    nopat = ebit * (1 - tax_rate) if ebit is not None else None
    invested_cap = safe_first(equity_series) + (debt_now - safe_first(cash_series))
    roic = nopat / invested_cap if (nopat is not None and invested_cap) else None
    eva  = (roic - wacc) * invested_cap if all(v is not None for v in (roic, wacc, invested_cap)) else None

    # --------- Otros ratios -------------------------------------------
    price = info.get("currentPrice")
    fcf   = safe_first(seek_row(cf, ["Free Cash Flow"]))
    shares = info.get("sharesOutstanding")
    pfcf  = price / (fcf / shares) if (fcf and shares) else None

    return {
        "Ticker": ticker,
        "Sector": info.get("sector"),
        "Precio": price,
        "P/E": info.get("trailingPE"),
        "P/B": info.get("priceToBook"),
        "P/FCF": pfcf,
        "Dividend Yield %": info.get("dividendYield"),
        "Payout Ratio": info.get("payoutRatio"),
        "ROA": info.get("returnOnAssets"),
        "ROE": info.get("returnOnEquity"),
        "Current Ratio": info.get("currentRatio"),
        "Quick Ratio": info.get("quickRatio"),
        "Debt/Eq": info.get("debtToEquity"),
        "LtDebt/Eq": info.get("longTermDebtToEquity"),
        "Oper Margin": info.get("operatingMargins"),
        "Profit Margin": info.get("profitMargins"),
        "WACC": wacc,
        "ROIC": roic,
        "EVA": eva,
        "Revenue Growth": cagr4(fin, "Total Revenue"),
        "EPS Growth":     cagr4(fin, "Net Income"),
        "FCF Growth":     cagr4(cf, "Free Cash Flow") or cagr4(cf, "Operating Cash Flow"),
    }

# =============================================================
# 3. INTERFAZ PRINCIPAL
# =============================================================
def main():
    st.title("📊 Dashboard de Análisis Financiero Avanzado")

    # ---------- Sidebar ---------------------------------------
    with st.sidebar:
        st.header("⚙️ Configuración")
        tk_in = st.text_area("Tickers (coma)", "HRL, AAPL, MSFT")
        max_t = st.slider("Máx tickers", 1, 50, 20)
        st.markdown("---")
        global Rf, Rm, Tc0
        Rf  = st.number_input("Risk-free (%)", 0.0, 20.0, 4.35) / 100
        Rm  = st.number_input("Market return (%)", 0.0, 30.0, 8.5) / 100
        Tc0 = st.number_input("Tax rate default (%)", 0.0, 50.0, 21.0) / 100

    tickers = [t.strip().upper() for t in tk_in.split(",") if t.strip()][:max_t]

    # ---------- Ejecución ------------------------------------
    if st.button("🔍 Analizar", type="primary"):
        if not tickers:
            st.warning("Ingresa al menos un ticker")
            return

        data_list, errs, bar = [], [], st.progress(0)
        for i, tk in enumerate(tickers, 1):
            try:
                data_list.append(obtener_datos_financieros(tk, Tc0))
            except Exception as e:
                errs.append({"Ticker": tk, "Error": str(e)})
            bar.progress(i / len(tickers))
            time.sleep(1)
        bar.empty()

        if not data_list:
            st.error("Sin datos válidos.")
            if errs:
                st.table(pd.DataFrame(errs))
            return

        df = pd.DataFrame(data_list)

        # Formateo porcentual
        pct_cols = ["Dividend Yield %", "Payout Ratio", "ROA", "ROE",
                    "Oper Margin", "Profit Margin", "WACC", "ROIC"]
        for col in pct_cols:
            df[col] = df[col].apply(lambda x: f"{x*100:,.2f}%" if pd.notnull(x) else "N/D")

        # --------- Sección 1: Resumen ---------------------------
        st.header("📋 Resumen General")
        resumen_cols = ["Ticker", "Sector", "Precio", "P/E", "P/B", "P/FCF",
                        "Dividend Yield %", "Payout Ratio", "ROA", "ROE",
                        "Current Ratio", "Debt/Eq", "Oper Margin", "Profit Margin",
                        "WACC", "ROIC", "EVA"]
        st.dataframe(df[resumen_cols].dropna(how='all', axis=1),
                     use_container_width=True, height=400)

        if errs:
            st.subheader("🚫 Errores")
            st.table(pd.DataFrame(errs))

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
