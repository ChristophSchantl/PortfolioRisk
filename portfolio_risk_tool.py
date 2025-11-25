# portfolio_risk_tool_institutional.py
# Streamlit-Tool für Portfolio-Bewertung & Risikoanalyse mit manuellen Gewichten
# Fokus: institutionelle Kennzahlen und Präsentation
#
# Features:
# - Close-Prices via yfinance
# - Lookback in Monaten (1–36)
# - Manuelle Gewichte (Standard = Equal Weight) für Basis-Portfolio
# - Optionales After-Portfolio mit zusätzlichen Aktien
# - Kennzahlen: Return, Vol, Sharpe, Sortino, MaxDD, Hit Ratio
#   + Beta/Alpha vs. S&P500 & DAX
#   + Tracking Error & Information Ratio
#   + Upside/Downside Capture
#   + Skew, Kurtosis, Tail Ratio
#   + VaR/ES (95/99)
#   + Drawdown-Statistiken
# - Grafiken:
#   - Rendite vs. Volatilität (Einzeltitel, klein)
#   - Kumulierte Performance (Basis/After)
#   - Drawdown-Kurve
#   - Rolling Vol & Rolling Sharpe
#   - Histogramm der Tagesrenditen mit Normal-Overlay
#   - Risk Contributions (CTR/MCTR) nach Volatilität
#   - Monats-Return-Matrix & Worst-Months
# - Export der Tickerliste im EU-CSV-Format

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from math import sqrt, pi, exp

TRADING_DAYS = 252

st.set_page_config(
    page_title="Institutionelles Portfolio-Risiko-Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Institutionelles Portfolio-Risiko-Dashboard (manuelle Gewichtung)")

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
    "Zusätzliche Aktien für After-Szenario (optional)",
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
# Helper-Funktionen: Daten
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

# ─────────────────────────────────────────────
# Helper-Funktionen: Kennzahlen
# ─────────────────────────────────────────────
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


def drawdown_series(returns):
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    dd = cum / running_max - 1.0
    return dd


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


def tracking_error_and_ir(port_ret, bench_ret, freq=TRADING_DAYS):
    df = pd.concat([port_ret, bench_ret], axis=1).dropna()
    if df.shape[0] < 10:
        return np.nan, np.nan
    rp, rb = df.iloc[:, 0], df.iloc[:, 1]
    diff = rp - rb
    te = diff.std(ddof=1) * np.sqrt(freq)
    ann_rp = annualized_return(rp, freq)
    ann_rb = annualized_return(rb, freq)
    if te == 0 or np.isnan(te):
        ir = np.nan
    else:
        ir = (ann_rp - ann_rb) / te
    return te, ir


def capture_ratios(port_ret, bench_ret, freq=TRADING_DAYS):
    df = pd.concat([port_ret, bench_ret], axis=1).dropna()
    if df.shape[0] < 10:
        return np.nan, np.nan
    rp, rb = df.iloc[:, 0], df.iloc[:, 1]

    mask_up = rb > 0
    mask_down = rb < 0

    if mask_up.sum() > 0:
        up_port = annualized_return(rp[mask_up], freq)
        up_bench = annualized_return(rb[mask_up], freq)
        up_cap = up_port / up_bench if up_bench != 0 else np.nan
    else:
        up_cap = np.nan

    if mask_down.sum() > 0:
        down_port = annualized_return(rp[mask_down], freq)
        down_bench = annualized_return(rb[mask_down], freq)
        down_cap = down_port / down_bench if down_bench != 0 else np.nan
    else:
        down_cap = np.nan

    return up_cap, down_cap


def tail_metrics(returns):
    r = returns.dropna()
    if r.empty:
        return (np.nan,) * 7
    skew = r.skew()
    kurt = r.kurtosis()
    q05 = np.percentile(r, 5)
    q95 = np.percentile(r, 95)
    tail_ratio = q95 / abs(q05) if q05 != 0 else np.nan
    var95 = q05
    var99 = np.percentile(r, 1)
    es95 = r[r <= q05].mean() if (r <= q05).any() else np.nan
    es99 = r[r <= var99].mean() if (r <= var99).any() else np.nan
    return skew, kurt, tail_ratio, var95, es95, var99, es99


def portfolio_series(returns_df, weights):
    w = np.array(weights).reshape(-1, 1)
    return (returns_df @ w).squeeze()


def rolling_sharpe(port_ret, risk_free_annual, window, freq=TRADING_DAYS):
    if window <= 1:
        return port_ret * np.nan
    daily_rf = risk_free_annual / freq
    ex = port_ret - daily_rf
    roll_mean = ex.rolling(window).mean()
    roll_std = ex.rolling(window).std(ddof=1)
    rs = roll_mean / roll_std * np.sqrt(freq)
    return rs


def risk_contributions(returns_df, weights, freq=TRADING_DAYS):
    """
    Risk Contributions auf Basis der Kovarianzmatrix (annualisiert).
    """
    cov = returns_df.cov()
    cov_annual = cov * freq
    w = np.array(weights)
    mat = cov_annual.values
    port_var = float(w.T @ mat @ w)
    port_vol = np.sqrt(port_var) if port_var >= 0 else np.nan
    if port_vol == 0 or np.isnan(port_vol):
        mctr = np.zeros_like(w)
        ctr = np.zeros_like(w)
        ctr_pct = np.zeros_like(w)
    else:
        mctr = (mat @ w) / port_vol
        ctr = w * mctr
        ctr_pct = ctr / port_vol
    return port_vol, mctr, ctr, ctr_pct


def monthly_return_matrix(port_ret):
    """
    Monatsreturns (resample) und Matrix Year x Month.
    Korrigierte Version: (1 + r) wird nur EINMAL verwendet.
    """
    if not isinstance(port_ret.index, pd.DatetimeIndex):
        return None, None

    # Korrektur: (1 + port_ret) resamplen und die Produkte bilden
    monthly = (1 + port_ret).resample('M').prod() - 1

    if monthly.empty:
        return None, None

    df = monthly.to_frame(name="Return")
    df["Year"] = df.index.year
    df["Month"] = df.index.strftime("%b")

    pivot = df.pivot(index="Year", columns="Month", values="Return")

    # Monate sauber sortieren
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    cols = [m for m in month_order if m in pivot.columns]
    pivot = pivot[cols]

    return pivot, monthly



def portfolio_stats(returns_df, weights,
                    risk_free_annual,
                    bench_spx=None,
                    bench_dax=None):
    """
    Portfolio-Kennzahlen + Benchmark-abhängige Metriken.
    """
    port_ret = portfolio_series(returns_df, weights)

    ann_ret = annualized_return(port_ret)
    ann_vol = annualized_vol(port_ret)
    sr = sharpe_ratio(port_ret, risk_free_annual)
    so = sortino_ratio(port_ret, risk_free_annual)

    dd = drawdown_series(port_ret)
    max_dd = dd.min()
    if (dd < 0).any():
        avg_dd = dd[dd < 0].mean()
        med_dd = dd[dd < 0].median()
        time_under_water = (dd < 0).mean()
    else:
        avg_dd = 0.0
        med_dd = 0.0
        time_under_water = 0.0

    hr = hit_ratio(port_ret)

    beta_spx, alpha_spx = (np.nan, np.nan)
    beta_dax, alpha_dax = (np.nan, np.nan)
    te_spx, ir_spx = (np.nan, np.nan)
    te_dax, ir_dax = (np.nan, np.nan)
    up_spx, down_spx = (np.nan, np.nan)
    up_dax, down_dax = (np.nan, np.nan)

    if bench_spx is not None:
        beta_spx, alpha_spx = beta_alpha(port_ret, bench_spx, risk_free_annual)
        te_spx, ir_spx = tracking_error_and_ir(port_ret, bench_spx)
        up_spx, down_spx = capture_ratios(port_ret, bench_spx)

    if bench_dax is not None:
        beta_dax, alpha_dax = beta_alpha(port_ret, bench_dax, risk_free_annual)
        te_dax, ir_dax = tracking_error_and_ir(port_ret, bench_dax)
        up_dax, down_dax = capture_ratios(port_ret, bench_dax)

    skew, kurt, tail_ratio, var95, es95, var99, es99 = tail_metrics(port_ret)

    return {
        "Ann. Return": ann_ret,
        "Ann. Vol": ann_vol,
        "Sharpe": sr,
        "Sortino": so,
        "Max Drawdown": max_dd,
        "Avg Drawdown": avg_dd,
        "Med Drawdown": med_dd,
        "Time Under Water": time_under_water,
        "Hit Ratio": hr,
        "Beta_SPX": beta_spx,
        "Alpha_SPX": alpha_spx,
        "Beta_DAX": beta_dax,
        "Alpha_DAX": alpha_dax,
        "TE_SPX": te_spx,
        "IR_SPX": ir_spx,
        "TE_DAX": te_dax,
        "IR_DAX": ir_dax,
        "Upside_SPX": up_spx,
        "Downside_SPX": down_spx,
        "Upside_DAX": up_dax,
        "Downside_DAX": down_dax,
        "Skew": skew,
        "Kurtosis": kurt,
        "TailRatio": tail_ratio,
        "VaR95": var95,
        "ES95": es95,
        "VaR99": var99,
        "ES99": es99,
        "Port_Returns": port_ret,  # für weitere Analysen
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

returns_base = returns_all[base_universe] if base_universe else pd.DataFrame()
returns_after = returns_all[after_universe] if after_universe else pd.DataFrame()

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
        "Max Drawdown": drawdown_series(r).min(),
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

# Scatter: Rendite vs. Volatilität – klein, Achsen vertauscht
st.subheader("Rendite vs. Volatilität (Einzeltitel)")

fig_scatter, ax_scatter = plt.subplots(figsize=(2.5, 2.0))

for ticker in asset_df.index:
    ret = asset_df.loc[ticker, "Ann. Return"]
    vol = asset_df.loc[ticker, "Ann. Vol"]
    ax_scatter.scatter(ret, vol, s=8)
    ax_scatter.annotate(
        ticker,
        (ret, vol),
        textcoords="offset points",
        xytext=(3, 2),
        fontsize=6
    )

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

base_stats = None
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

    st.caption(f"Basis: Gewichte werden intern auf Summe 100% normiert "
               f"(aktuell: {w_base_raw.sum():.2f}%).")

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
# Szenario: zusätzliche Aktien (After)
# ─────────────────────────────────────────────
st.subheader("Szenario: zusätzliche Aktien & Vergleich Basis vs. After")

after_stats = None
if not after_universe or after_universe == base_universe or base_stats is None:
    st.info("Gib im Sidebar-Feld 'Zusätzliche Aktien' Ticker ein, "
            "um ein After-Szenario inkl. Vergleich zu sehen.")
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

    st.caption(f"After: Gewichte werden intern auf Summe 100% normiert "
               f"(aktuell: {w_after_raw.sum():.2f}%).")

    returns_after_used = returns_all[after_universe]
    after_stats = portfolio_stats(
        returns_after_used, w_after, risk_free_annual, bench_ret_spx, bench_ret_dax
    )

    col1, col2 = st.columns(2)

    with col1:
        if base_universe:
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
                "TE_SPX": base_stats["TE_SPX"],
                "IR_SPX": base_stats["IR_SPX"],
                "TE_DAX": base_stats["TE_DAX"],
                "IR_DAX": base_stats["IR_DAX"],
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
                "TE_SPX": after_stats["TE_SPX"],
                "IR_SPX": after_stats["IR_SPX"],
                "TE_DAX": after_stats["TE_DAX"],
                "IR_DAX": after_stats["IR_DAX"],
            },
        }
    )

    disp_compare = compare_df.copy()
    for row in ["Ann. Return", "Ann. Vol", "Max Drawdown", "Hit Ratio",
                "Alpha_SPX", "Alpha_DAX", "TE_SPX", "TE_DAX"]:
        disp_compare.loc[row, :] = disp_compare.loc[row, :].apply(fmt_pct)
    for row in ["Sharpe", "Sortino", "Beta_SPX", "Beta_DAX", "IR_SPX", "IR_DAX"]:
        disp_compare.loc[row, :] = disp_compare.loc[row, :].apply(fmt_ratio)

    st.dataframe(disp_compare, use_container_width=True)

# ─────────────────────────────────────────────
# Risiko-Analyse: Drawdown, Rolling, Histogramm, Tail
# ─────────────────────────────────────────────
st.subheader("Risiko-Analyse & Tail-Metriken")

# Auswahl, welches Portfolio für tiefere Risikoanalyse verwendet wird
portfolio_choices = {}
if base_stats is not None:
    portfolio_choices["Basis"] = {
        "returns_df": returns_base,
        "weights": w_base,
        "stats": base_stats
    }
if after_stats is not None:
    portfolio_choices["After"] = {
        "returns_df": returns_after_used,
        "weights": w_after,
        "stats": after_stats
    }

if not portfolio_choices:
    st.info("Kein Portfolio mit definierten Gewichten verfügbar für Risikoanalyse.")
else:
    selected_name = st.selectbox("Portfolio für Risikoanalyse auswählen",
                                 list(portfolio_choices.keys()))
    sel = portfolio_choices[selected_name]
    sel_ret_df = sel["returns_df"]
    sel_w = sel["weights"]
    sel_stats = sel["stats"]
    port_ret = sel_stats["Port_Returns"]

    # Drawdown-Kurve
    dd = drawdown_series(port_ret)

    col_r1, col_r2 = st.columns([2, 1])

    with col_r1:
        st.write("Drawdown-Kurve")
        fig_dd, ax_dd = plt.subplots(figsize=(4, 2.5))
        dd.plot(ax=ax_dd)
        ax_dd.set_title(f"Drawdown {selected_name}")
        ax_dd.set_ylabel("Drawdown")
        ax_dd.set_xlabel("")
        ax_dd.axhline(0, color="black", linewidth=0.5)
        ax_dd.tick_params(labelsize=7)
        fig_dd.tight_layout(pad=0.5)
        st.pyplot(fig_dd, use_container_width=False)

    with col_r2:
        st.write("Tail- & Downside-Metriken")
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("VaR 95% (daily)", fmt_pct(sel_stats["VaR95"]))
        col_m2.metric("ES 95% (daily)", fmt_pct(sel_stats["ES95"]))
        col_m1.metric("VaR 99% (daily)", fmt_pct(sel_stats["VaR99"]))
        col_m2.metric("ES 99% (daily)", fmt_pct(sel_stats["ES99"]))
        col_m1.metric("Skew", fmt_ratio(sel_stats["Skew"]))
        col_m2.metric("Kurtosis", fmt_ratio(sel_stats["Kurtosis"]))
        col_m1.metric("Time Under Water", fmt_pct(sel_stats["Time Under Water"]))
        col_m2.metric("Avg Drawdown", fmt_pct(sel_stats["Avg Drawdown"]))

    # Rolling Vol & Rolling Sharpe
    st.write("Rolling Volatilität & Rolling Sharpe (6M-Fenster)")

    window = min(126, max(20, len(port_ret) // 3))  # dynamischer, aber meist ~6M
    roll_vol = port_ret.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS)
    roll_sharpe = rolling_sharpe(port_ret, risk_free_annual, window)

    fig_roll, ax_roll = plt.subplots(figsize=(4.5, 2.5))
    roll_vol.plot(ax=ax_roll, label="Rolling Vol (ann.)")
    ax_roll.set_ylabel("Volatilität", fontsize=8)
    ax_roll.tick_params(axis="y", labelsize=7)
    ax_roll.tick_params(axis="x", labelsize=7)

    ax2 = ax_roll.twinx()
    roll_sharpe.plot(ax=ax2, color="tab:red", label="Rolling Sharpe")
    ax2.set_ylabel("Sharpe", fontsize=8)
    ax2.tick_params(axis="y", labelsize=7)

    ax_roll.set_title(f"Rolling Risk Metrics ({selected_name})")
    lines, labels = ax_roll.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax_roll.legend(lines + lines2, labels + labels2, fontsize=7, loc="upper left")
    fig_roll.tight_layout(pad=0.5)
    st.pyplot(fig_roll, use_container_width=False)

    # Histogramm + Normal-Overlay
    st.write("Verteilung der Tagesrenditen (Histogramm + Normal-Overlay)")

    r = port_ret.dropna()
    if not r.empty:
        mu = r.mean()
        sigma = r.std(ddof=1)

        fig_hist, ax_hist = plt.subplots(figsize=(3.0, 2.2))
        ax_hist.hist(r, bins=40, density=True, alpha=0.6)
        if sigma > 0:
            x_vals = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
            pdf = 1.0 / (sigma * sqrt(2 * pi)) * np.exp(-0.5 * ((x_vals - mu) / sigma) ** 2)
            ax_hist.plot(x_vals, pdf, linewidth=1.0)
        ax_hist.set_title(f"Return-Distribution {selected_name}", fontsize=8)
        ax_hist.tick_params(axis="both", labelsize=7)
        fig_hist.tight_layout(pad=0.5)
        st.pyplot(fig_hist, use_container_width=False)

    # Tracking Error, IR, Capture Ratios
    st.write("Benchmark-bezogene Kennzahlen")

    bench_table_rows = []
    for label, br, te_key, ir_key, up_key, down_key in [
        ("S&P500", bench_ret_spx, "TE_SPX", "IR_SPX", "Upside_SPX", "Downside_SPX"),
        ("DAX", bench_ret_dax, "TE_DAX", "IR_DAX", "Upside_DAX", "Downside_DAX"),
    ]:
        if br is None:
            continue
        bench_table_rows.append({
            "Benchmark": label,
            "Tracking Error": sel_stats[te_key],
            "Information Ratio": sel_stats[ir_key],
            "Upside Capture": sel_stats[up_key],
            "Downside Capture": sel_stats[down_key],
        })

    if bench_table_rows:
        bench_df = pd.DataFrame(bench_table_rows).set_index("Benchmark")
        display_bench = bench_df.copy()
        display_bench["Tracking Error"] = display_bench["Tracking Error"].apply(fmt_pct)
        display_bench["Information Ratio"] = display_bench["Information Ratio"].apply(fmt_ratio)
        display_bench["Upside Capture"] = display_bench["Upside Capture"].apply(
            lambda x: "n/a" if np.isnan(x) else f"{x*100:,.1f}%"
        )
        display_bench["Downside Capture"] = display_bench["Downside Capture"].apply(
            lambda x: "n/a" if np.isnan(x) else f"{x*100:,.1f}%"
        )
        st.dataframe(display_bench, use_container_width=True)

    # Monatsmatrix & Worst-Months
    st.write("Monatsreturns & Stress-Monate")

    pivot_months, monthly_series = monthly_return_matrix(port_ret)
    if pivot_months is not None:
        fmt_pivot = pivot_months.applymap(lambda x: fmt_pct(x) if pd.notna(x) else "")
        st.dataframe(fmt_pivot, use_container_width=True)
        worst_months = monthly_series.nsmallest(5)
        worst_df = worst_months.to_frame(name="Return")
        worst_df["Year"] = worst_df.index.year
        worst_df["Month"] = worst_df.index.strftime("%b")
        worst_df["Return_fmt"] = worst_df["Return"].apply(fmt_pct)
        st.write("Schlechteste 5 Monate im Sample:")
        st.dataframe(
            worst_df[["Year", "Month", "Return_fmt"]],
            use_container_width=True
        )

# ─────────────────────────────────────────────
# Risk Contributions (MCTR / CTR)
# ─────────────────────────────────────────────
st.subheader("Risk Contributions (Volatilitätsbeitrag je Titel)")

if not portfolio_choices:
    st.info("Für Risk Contributions muss mindestens ein Portfolio mit Gewichten vorliegen.")
else:
    rc_choice = st.selectbox("Portfolio für Risk Contributions auswählen",
                             list(portfolio_choices.keys()),
                             key="rc_choice")
    rc_sel = portfolio_choices[rc_choice]
    rc_ret_df = rc_sel["returns_df"]
    rc_w = rc_sel["weights"]

    port_vol, mctr, ctr, ctr_pct = risk_contributions(rc_ret_df, rc_w)
    tickers_rc = rc_ret_df.columns

    rc_df = pd.DataFrame({
        "Ticker": tickers_rc,
        "Weight_%": rc_w * 100,
        "MCTR": mctr,
        "CTR": ctr,
        "CTR_% of Risk": ctr_pct * 100
    }).set_index("Ticker")

    rc_display = rc_df.copy()
    rc_display["Weight_%"] = rc_display["Weight_%"].apply(lambda x: f"{x:,.2f}%")
    rc_display["CTR_% of Risk"] = rc_display["CTR_% of Risk"].apply(lambda x: f"{x:,.2f}%")
    rc_display["MCTR"] = rc_display["MCTR"].apply(lambda x: f"{x:,.4f}")
    rc_display["CTR"] = rc_display["CTR"].apply(lambda x: f"{x:,.4f}")

    st.write(f"Portfolio-Volatilität (ann.): {fmt_pct(port_vol)}")
    st.dataframe(rc_display, use_container_width=True)

    st.write("Gewicht vs. Risikobeitrag (% vom Gesamt-Risiko)")

    fig_rc, ax_rc = plt.subplots(figsize=(4.0, 2.4))
    ax_rc.bar(tickers_rc, rc_df["CTR_% of Risk"])
    ax_rc.set_ylabel("% of Total Volatilität", fontsize=8)
    ax_rc.set_xticklabels(tickers_rc, rotation=45, ha="right", fontsize=7)
    ax_rc.tick_params(axis="y", labelsize=7)
    fig_rc.tight_layout(pad=0.5)
    st.pyplot(fig_rc, use_container_width=False)

# ─────────────────────────────────────────────
# Export: Ticker-Liste als CSV (EU-Format)
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
