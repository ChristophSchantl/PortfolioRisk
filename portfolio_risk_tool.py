# portfolio_risk_tool_manual.py
# Streamlit-Tool für Portfolio-Bewertung & manuelle Gewichtung
# - Close-Prices
# - Lookback in Monaten (bis 36)
# - Manuelle Gewichte (Standard: Equal Weight)
# - Scatter: Rendite vs. Volatilität je Aktie
# - Szenario: zusätzliche Aktien (Before/After-Strukturvergleich)
# - Download der Ticker als CSV (EU-Format)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

TRADING_DAYS = 252

st.set_page_config(
    page_title="Portfolio-Bewertung (manuelle Gewichtung)",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Portfolio-Bewertung & Risiko – manuelle Gewichtung")

# ─────────────────────────────────────────────
# Sidebar – Universe, Lookback, Benchmarks
# ─────────────────────────────────────────────
st.sidebar.header("Universe & Einstellungen")

default_base = "REGN,LULU,VOW3.DE,REI,DDL,NOV,SRPT,CAG,CMCSA"

base_input = st.sidebar.text_input(
    "Basis-Portfolio (Ticker, kommasepariert)",
    value=default_base,
)

base_tickers = [t.strip() for t in base_input.split(",") if t.strip()]

extra_input = st.sidebar.text_input(
    "Zusätzliche Aktien für Szenario (optional)",
    value=""
)
extra_tickers = [t.strip() for t in extra_input.split(",") if t.strip()]

# Lookback in Monaten
months_back = st.sidebar.slider(
    "Lookback (Monate)",
    min_value=1,
    max_value=36,
    value=12
)

end_date = datetime.today()
start_date = end_date - timedelta(days=months_back * 30)  # grobe Approximation

risk_free_annual = st.sidebar.number_input(
    "Risikofreier Zins p.a.",
    min_value=-0.05,
    max_value=0.10,
    value=0.02,
    step=0.005,
    format="%.3f"
)

benchmark_ticker_spx = st.sidebar.text_input(
    "Benchmark 1 (S&P 500)",
    value="^GSPC"
)

benchmark_ticker_dax = st.sidebar.text_input(
    "Benchmark 2 (DAX)",
    value="^GDAXI"
)

# ─────────────────────────────────────────────
# Helper-Funktionen
# ─────────────────────────────────────────────
def load_prices(tickers, start, end):
    """
    Pro Ticker Close-Kurse laden.
    Gibt prices, ok_tickers, missing_tickers zurück.
    """
    series_list = []
    ok = []
    missing = []

    for t in tickers:
        try:
            data = yf.download(
                t,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False
            )
        except Exception:
            missing.append(t)
            continue

        if data is None or data.empty or "Close" not in data.columns:
            missing.append(t)
            continue

        s = data["Close"].dropna()
        if s.empty:
            missing.append(t)
            continue

        s.name = t
        series_list.append(s)
        ok.append(t)

    if not series_list:
        return pd.DataFrame(), ok, missing

    prices = pd.concat(series_list, axis=1).sort_index()
    return prices, ok, missing


def load_benchmark_series(ticker, start, end):
    try:
        data = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False
        )
    except Exception:
        return None

    if data is None or data.empty or "Close" not in data.columns:
        return None

    s = data["Close"].dropna()
    if s.empty:
        return None
    return s


def annualized_return(returns, freq=TRADING_DAYS):
    r = returns.dropna()
    if r.empty:
        return np.nan
    cumulative = (1 + r).prod()
    n = len(r)
    return cumulative ** (freq / n) - 1.0


def annualized_vol(returns, freq=TRADING_DAYS):
    r = returns.dropna()
    if r.empty:
        return np.nan
    return r.std(ddof=1) * np.sqrt(freq)


def downside_deviation(returns, risk_free_annual, freq=TRADING_DAYS):
    r = returns.dropna()
    if r.empty:
        return np.nan
    daily_rf = risk_free_annual / freq
    diff = r - daily_rf
    downside = np.where(diff < 0, diff, 0.0)
    if (downside == 0).all():
        return 0.0
    downside_var = (downside ** 2).mean()
    return np.sqrt(downside_var) * np.sqrt(freq)


def sharpe_ratio(returns, risk_free_annual, freq=TRADING_DAYS):
    ann_ret = annualized_return(returns, freq)
    ann_vol = annualized_vol(returns, freq)
    if ann_vol == 0 or np.isnan(ann_vol):
        return np.nan
    return (ann_ret - risk_free_annual) / ann_vol


def sortino_ratio(returns, risk_free_annual, freq=TRADING_DAYS):
    ann_ret = annualized_return(returns, freq)
    dd = downside_deviation(returns, risk_free_annual, freq)
    if dd == 0 or np.isnan(dd):
        return np.nan
    return (ann_ret - risk_free_annual) / dd


def max_drawdown(returns):
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    dd = cum / running_max - 1.0
    return dd.min()


def hit_ratio(returns):
    r = returns.dropna()
    if r.empty:
        return np.nan
    return (r > 0).mean()


def beta_alpha(asset_ret, bench_ret, risk_free_annual, freq=TRADING_DAYS):
    df = pd.concat([asset_ret, bench_ret], axis=1).dropna()
    if df.shape[0] < 10:
        return np.nan, np.nan
    rp, rb = df.iloc[:, 0], df.iloc[:, 1]
    cov = np.cov(rp, rb)[0, 1]
    var_b = np.var(rb)
    if var_b == 0:
        return np.nan, np.nan
    beta = cov / var_b
    ann_rp = annualized_return(rp, freq)
    ann_rb = annualized_return(rb, freq)
    alpha = (ann_rp - risk_free_annual) - beta * (ann_rb - risk_free_annual)
    return beta, alpha


def portfolio_series(returns_df, weights):
    w = np.array(weights).reshape(-1, 1)
    return (returns_df @ w).squeeze()


def portfolio_stats(returns_df, weights,
                    risk_free_annual,
                    bench_spx=None,
                    bench_dax=None):
    port_ret = portfolio_series(returns_df, weights)

    ann_ret = annualized_return(port_ret)
    ann_vol = annualized_vol(port_ret)
    sr = sharpe_ratio(port_ret, risk_free_annual)
    so = sortino_ratio(port_ret, risk_free_annual)
    mdd = max_drawdown(port_ret)
    hr = hit_ratio(port_ret)

    beta_spx, alpha_spx = (np.nan, np.nan)
    beta_dax, alpha_dax = (np.nan, np.nan)

    if bench_spx is not None:
        beta_spx, alpha_spx = beta_alpha(port_ret, bench_spx, risk_free_annual)
    if bench_dax is not None:
        beta_dax, alpha_dax = beta_alpha(port_ret, bench_dax, risk_free_annual)

    return {
        "Ann. Return": ann_ret,
        "Ann. Vol": ann_vol,
        "Sharpe": sr,
        "Sortino": so,
        "Max Drawdown": mdd,
        "Hit Ratio": hr,
        "Beta_SPX": beta_spx,
        "Alpha_SPX": alpha_spx,
        "Beta_DAX": beta_dax,
        "Alpha_DAX": alpha_dax,
    }


def fmt_pct(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x*100:,.2f}%"


def fmt_ratio(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.3f}"


# ─────────────────────────────────────────────
# Daten laden
# ─────────────────────────────────────────────
all_input_tickers = base_tickers + extra_tickers
all_input_tickers = [t for t in all_input_tickers if t]  # cleanup

if not all_input_tickers:
    st.error("Bitte mindestens einen Ticker im Basis-Portfolio eingeben.")
    st.stop()

with st.spinner("Lade Kursdaten..."):
    prices_all, ok_tickers, missing_tickers = load_prices(all_input_tickers, start_date, end_date)

if prices_all.empty:
    st.error("Keine Kursdaten gefunden. Prüfe Ticker, Internet-Verbindung oder Zeitraum.")
    if missing_tickers:
        st.warning("Keine Daten für: " + ", ".join(missing_tickers))
    st.stop()

if missing_tickers:
    st.warning("Folgende Ticker konnten nicht geladen werden und wurden ignoriert: "
               + ", ".join(missing_tickers))

returns_all = prices_all.pct_change().dropna()

# Benchmarks
bench_spx = load_benchmark_series(benchmark_ticker_spx, start_date, end_date)
bench_dax = load_benchmark_series(benchmark_ticker_dax, start_date, end_date)

bench_ret_spx = bench_spx.pct_change().dropna() if bench_spx is not None else None
bench_ret_dax = bench_dax.pct_change().dropna() if bench_dax is not None else None

if bench_ret_spx is None:
    st.warning("Benchmark S&P500 konnte nicht geladen werden.")
if bench_ret_dax is None:
    st.warning("Benchmark DAX konnte nicht geladen werden.")

st.subheader("Basisdaten")
st.write(f"Zeitraum: {start_date.date()} bis {end_date.date()}  •  {returns_all.shape[0]} Handelstage")
st.write(f"Aktive Titel: {', '.join(returns_all.columns)}")

# Base- & After-Universen auf tatsächlich verfügbare Ticker einschränken
base_universe = [t for t in base_tickers if t in returns_all.columns]
after_universe = [t for t in (base_tickers + extra_tickers) if t in returns_all.columns]

returns_base = returns_all[base_universe]
returns_after = returns_all[after_universe]

# ─────────────────────────────────────────────
# Einzelwert-Kennzahlen & Rendite/Vol-Scatter
# ─────────────────────────────────────────────
rows = []
for col in returns_all.columns:
    r = returns_all[col]
    beta_spx, alpha_spx = (np.nan, np.nan)
    beta_dax, alpha_dax = (np.nan, np.nan)

    if bench_ret_spx is not None:
        beta_spx, alpha_spx = beta_alpha(r, bench_ret_spx, risk_free_annual)
    if bench_ret_dax is not None:
        beta_dax, alpha_dax = beta_alpha(r, bench_ret_dax, risk_free_annual)

    rows.append({
        "Ticker": col,
        "Ann. Return": annualized_return(r),
        "Ann. Vol": annualized_vol(r),
        "Sharpe": sharpe_ratio(r, risk_free_annual),
        "Sortino": sortino_ratio(r, risk_free_annual),
        "Max Drawdown": max_drawdown(r),
        "Hit Ratio": hit_ratio(r),
        "Beta_SPX": beta_spx,
        "Alpha_SPX": alpha_spx,
        "Beta_DAX": beta_dax,
        "Alpha_DAX": alpha_dax,
    })

asset_df = pd.DataFrame(rows).set_index("Ticker")

display_df = asset_df.copy()
for c in ["Ann. Return", "Ann. Vol", "Max Drawdown", "Hit Ratio", "Alpha_SPX", "Alpha_DAX"]:
    display_df[c] = display_df[c].apply(fmt_pct)

for c in ["Sharpe", "Sortino", "Beta_SPX", "Beta_DAX"]:
    display_df[c] = display_df[c].apply(fmt_ratio)

st.subheader("Einzelwert-Kennzahlen")
st.dataframe(display_df, use_container_width=True)

# Scatter: Rendite vs. Volatilität (Einzeltitel) – sehr kleine Grafik, Achsen vertauscht
st.subheader("Rendite vs. Volatilität (Einzeltitel)")

# deutlich kleinere Figure
fig_scatter, ax_scatter = plt.subplots(figsize=(2.5, 2.0))  # Breite, Höhe in Inch

for ticker in asset_df.index:
    ret = asset_df.loc[ticker, "Ann. Return"]
    vol = asset_df.loc[ticker, "Ann. Vol"]
    ax_scatter.scatter(ret, vol, s=8)  # s = Markergröße
    ax_scatter.annotate(
        ticker,
        (ret, vol),
        textcoords="offset points",
        xytext=(3, 2),
        fontsize=6   # kleinere Schrift
    )

# vertikale Nulllinie für Rendite
ax_scatter.axvline(0, linestyle="--", linewidth=0.6)

ax_scatter.set_xlabel("Ann. Rendite", fontsize=8)
ax_scatter.set_ylabel("Ann. Volatilität", fontsize=8)

ax_scatter.tick_params(axis="both", labelsize=7)

fig_scatter.tight_layout(pad=0.5)

st.pyplot(fig_scatter, use_container_width=False)



# ─────────────────────────────────────────────
# Manuelle Gewichte – Basisportfolio
# ─────────────────────────────────────────────
st.subheader("Portfolio-Gewichte – Basisportfolio")

if not base_universe:
    st.warning("Keine der Basis-Ticker konnten geladen werden.")
else:
    n_base = len(base_universe)
    base_default_w = np.ones(n_base) / n_base * 100  # Prozent

    base_weights_df = pd.DataFrame({
        "Ticker": base_universe,
        "Weight_%": np.round(base_default_w, 2)
    })

    edited_base = st.data_editor(
        base_weights_df,
        key="base_weights_editor",
        use_container_width=True,
        num_rows="fixed"
    )

    w_base_raw = edited_base["Weight_%"].values.astype(float)
    if w_base_raw.sum() == 0:
        w_base = np.ones_like(w_base_raw) / len(w_base_raw)
    else:
        w_base = w_base_raw / w_base_raw.sum()

    st.caption(f"Gewichte werden intern auf Summe 100% normiert (aktuell: {w_base_raw.sum():.2f}%).")

    # Kennzahlen Basis-Portfolio
    base_stats = portfolio_stats(
        returns_base, w_base, risk_free_annual, bench_ret_spx, bench_ret_dax
    )

    colA, colB, colC = st.columns(3)
    colD, colE, colF = st.columns(3)

    colA.metric("Ann. Return (Basis)", fmt_pct(base_stats["Ann. Return"]))
    colB.metric("Ann. Vol (Basis)", fmt_pct(base_stats["Ann. Vol"]))
    colC.metric("Max Drawdown (Basis)", fmt_pct(base_stats["Max Drawdown"]))
    colD.metric("Sharpe (Basis)", fmt_ratio(base_stats["Sharpe"]))
    colE.metric("Sortino (Basis)", fmt_ratio(base_stats["Sortino"]))
    colF.metric("Hit Ratio (Basis)", fmt_pct(base_stats["Hit Ratio"]))

# ─────────────────────────────────────────────
# Szenario „zusätzliche Aktien“ – Before/After-Struktur
# ─────────────────────────────────────────────
st.subheader("Szenario: zusätzliche Aktien & Vergleich der Portfoliostruktur")

if not after_universe or after_universe == base_universe:
    st.info("Gib im Sidebar-Feld 'Zusätzliche Aktien' Ticker ein, um ein After-Szenario zu sehen.")
else:
    n_after = len(after_universe)
    after_default_w = np.ones(n_after) / n_after * 100

    after_weights_df = pd.DataFrame({
        "Ticker": after_universe,
        "Weight_%": np.round(after_default_w, 2)
    })

    edited_after = st.data_editor(
        after_weights_df,
        key="after_weights_editor",
        use_container_width=True,
        num_rows="fixed"
    )

    w_after_raw = edited_after["Weight_%"].values.astype(float)
    if w_after_raw.sum() == 0:
        w_after = np.ones_like(w_after_raw) / len(w_after_raw)
    else:
        w_after = w_after_raw / w_after_raw.sum()

    st.caption(f"(After) Gewichte werden intern auf Summe 100% normiert "
               f"(aktuell: {w_after_raw.sum():.2f}%).")

    returns_after_used = returns_all[after_universe]
    after_stats = portfolio_stats(
        returns_after_used, w_after, risk_free_annual, bench_ret_spx, bench_ret_dax
    )

    col1, col2 = st.columns(2)

    with col1:
        st.write("Basis-Portfolio – Gewichte:")
        st.bar_chart(
            pd.DataFrame({"Weight_%": w_base * 100}, index=base_universe)
        )

    with col2:
        st.write("After-Portfolio – Gewichte:")
        st.bar_chart(
            pd.DataFrame({"Weight_%": w_after * 100}, index=after_universe)
        )

    st.write("Portfolio-Kennzahlen: Basis vs. After")

    compare_df = pd.DataFrame(
        {
            "Basis": {
                "Ann. Return": base_stats["Ann. Return"],
                "Ann. Vol": base_stats["Ann. Vol"],
                "Sharpe": base_stats["Sharpe"],
                "Sortino": base_stats["Sortino"],
                "Max Drawdown": base_stats["Max Drawdown"],
                "Hit Ratio": base_stats["Hit Ratio"],
                "Beta_SPX": base_stats["Beta_SPX"],
                "Alpha_SPX": base_stats["Alpha_SPX"],
                "Beta_DAX": base_stats["Beta_DAX"],
                "Alpha_DAX": base_stats["Alpha_DAX"],
            },
            "After": {
                "Ann. Return": after_stats["Ann. Return"],
                "Ann. Vol": after_stats["Ann. Vol"],
                "Sharpe": after_stats["Sharpe"],
                "Sortino": after_stats["Sortino"],
                "Max Drawdown": after_stats["Max Drawdown"],
                "Hit Ratio": after_stats["Hit Ratio"],
                "Beta_SPX": after_stats["Beta_SPX"],
                "Alpha_SPX": after_stats["Alpha_SPX"],
                "Beta_DAX": after_stats["Beta_DAX"],
                "Alpha_DAX": after_stats["Alpha_DAX"],
            },
        }
    )

    # Formatierung für Anzeige
    disp_compare = compare_df.copy()
    for row in ["Ann. Return", "Ann. Vol", "Max Drawdown", "Hit Ratio", "Alpha_SPX", "Alpha_DAX"]:
        disp_compare.loc[row, :] = disp_compare.loc[row, :].apply(fmt_pct)
    for row in ["Sharpe", "Sortino", "Beta_SPX", "Beta_DAX"]:
        disp_compare.loc[row, :] = disp_compare.loc[row, :].apply(fmt_ratio)

    st.dataframe(disp_compare, use_container_width=True)

# ─────────────────────────────────────────────
# Download der Ticker als CSV (EU-Format)
# ─────────────────────────────────────────────
st.subheader("Export: Ticker-Liste als CSV (EU-Format)")

export_universe = sorted(set(all_input_tickers))
export_df = pd.DataFrame({"Ticker": export_universe})

csv_bytes = export_df.to_csv(
    sep=";", decimal=",", index=False
).encode("utf-8")

st.download_button(
    label="⬇️ Ticker-Liste als CSV speichern",
    data=csv_bytes,
    file_name="portfolio_ticker_eu.csv",
    mime="text/csv"
)

st.caption(
    "Alle Kennzahlen beruhen auf historischen Daten und stellen keine Garantie für "
    "zukünftige Entwicklungen dar. Nur für professionelle/qualifizierte Investoren."
)
