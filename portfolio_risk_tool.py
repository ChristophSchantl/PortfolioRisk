# portfolio_risk_tool.py
# Streamlit-Tool für Portfolio-Bewertung & Sortino-Optimierung
# - verwendet ausschließlich den Close-Preis
# - min. Gewicht je Aktie: 5% (falls machbar)
# - Beta/Alpha vs S&P500 (^GSPC) und DAX (^GDAXI)
# - Kennzahlen für Equal-Weight- und Optimized-Portfolio

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.optimize import minimize

TRADING_DAYS = 252
MIN_WEIGHT = 0.05  # 5% Mindestgewicht pro Aktie

st.set_page_config(
    page_title="Portfolio-Bewertung & Sortino-Optimierung",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Portfolio-Bewertung & Sortino-Optimierung")

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
default_tickers = "REGN,LULU,VOW3.DE,REI,DDL,NOV,SRPT,CAG,CMCSA"

st.sidebar.header("Universe & Einstellungen")

tickers_input = st.sidebar.text_input(
    "Ticker (kommagetrennt)",
    value=default_tickers
)

tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]
if not tickers:
    st.error("Bitte mindestens einen Ticker eingeben.")
    st.stop()

years_back = st.sidebar.slider(
    "Lookback (Jahre)",
    min_value=1,
    max_value=10,
    value=3
)

end_date = datetime.today()
start_date = end_date - timedelta(days=365 * years_back)

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

optimize_objective = st.sidebar.selectbox(
    "Optimierungsziel",
    ["Sortino Ratio (maximieren)", "Sharpe Ratio (maximieren)"],
    index=0
)

# ─────────────────────────────────────────────
# Helper-Funktionen
# ─────────────────────────────────────────────
def load_prices(tickers, start, end):
    """
    Robustes Laden: pro Ticker, verwendet ausschließlich 'Close'.
    Gibt prices, OK-Ticker, fehlende Ticker zurück.
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

        if data is None or data.empty:
            missing.append(t)
            continue

        # NUR Close verwenden
        if "Close" not in data.columns:
            missing.append(t)
            continue

        s = data["Close"].copy().dropna()
        if s.empty:
            missing.append(t)
            continue

        s.name = t  # Name setzen statt rename()
        series_list.append(s)
        ok.append(t)

    if not series_list:
        return pd.DataFrame(), ok, missing

    prices = pd.concat(series_list, axis=1).sort_index()
    return prices, ok, missing


def load_benchmark_series(ticker, start, end):
    """
    Benchmark-Serie laden, ausschließlich 'Close' verwenden.
    """
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


def beta_alpha(port_ret, bench_ret, risk_free_annual, freq=TRADING_DAYS):
    """
    Beta & Alpha nach CAPM gegen eine Benchmark.
    """
    df = pd.concat([port_ret, bench_ret], axis=1).dropna()
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


def portfolio_stats(weights,
                    returns_df,
                    risk_free_annual,
                    bench_ret_spx=None,
                    bench_ret_dax=None):
    """
    Kennzahlen für ein Portfolio mit gegebenen Gewichten.
    Zusätzlich Beta/Alpha vs. S&P500 und DAX.
    """
    w = np.array(weights).reshape(-1, 1)
    port_ret = (returns_df @ w).squeeze()

    ann_ret = annualized_return(port_ret)
    ann_vol = annualized_vol(port_ret)
    sr = sharpe_ratio(port_ret, risk_free_annual)
    so = sortino_ratio(port_ret, risk_free_annual)
    mdd = max_drawdown(port_ret)
    hr = hit_ratio(port_ret)

    beta_spx, alpha_spx = (np.nan, np.nan)
    beta_dax, alpha_dax = (np.nan, np.nan)

    if bench_ret_spx is not None:
        beta_spx, alpha_spx = beta_alpha(port_ret, bench_ret_spx, risk_free_annual)
    if bench_ret_dax is not None:
        beta_dax, alpha_dax = beta_alpha(port_ret, bench_ret_dax, risk_free_annual)

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


def average_correlation(corr_matrix):
    n = corr_matrix.shape[0]
    if n <= 1:
        return np.nan
    vals = corr_matrix.values
    upper = vals[np.triu_indices(n, k=1)]
    return upper.mean()


def optimize_weights(returns_df,
                     risk_free_annual,
                     objective="sortino",
                     long_only=True,
                     min_weight=MIN_WEIGHT):
    """
    Optimiert Gewichte bzgl. Sortino oder Sharpe.
    Long-only, Summe=1.
    Mindestgewicht pro Asset: min_weight (sofern machbar).
    """
    n = returns_df.shape[1]
    w0 = np.ones(n) / n

    eff_min = min_weight
    if long_only and n * eff_min > 1.0:
        # 5% Mindestgewicht ist bei zu vielen Titeln nicht machbar
        st.warning(
            f"Minimales Gewicht von {min_weight:.0%} ist mit {n} Titeln nicht machbar. "
            "Setze Untergrenze auf 0%."
        )
        eff_min = 0.0

    bounds = None
    if long_only:
        bounds = [(eff_min, 1.0)] * n

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    def port_ret_from_w(w):
        return returns_df @ np.array(w)

    def neg_sortino(w):
        r = port_ret_from_w(w)
        val = sortino_ratio(r, risk_free_annual)
        return -val if not np.isnan(val) else 1e6

    def neg_sharpe(w):
        r = port_ret_from_w(w)
        val = sharpe_ratio(r, risk_free_annual)
        return -val if not np.isnan(val) else 1e6

    fun = neg_sortino if objective.lower().startswith("sortino") else neg_sharpe

    res = minimize(
        fun,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "disp": False},
    )

    if not res.success:
        st.warning(f"Optimierung nicht konvergent: {res.message}")
        return w0

    w_opt = res.x
    # numerische Artefakte bereinigen
    w_opt[w_opt < 0] = 0.0
    s = w_opt.sum()
    if s != 0:
        w_opt /= s
    return w_opt


def fmt_pct(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x*100:,.2f}%"


def fmt_ratio(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.3f}"


# ─────────────────────────────────────────────
# Daten laden
# ─────────────────────────────────────────────
with st.spinner("Lade Kursdaten..."):
    prices, ok_tickers, missing_tickers = load_prices(tickers, start_date, end_date)

if prices.empty:
    st.error("Keine Kursdaten gefunden. Prüfe Ticker, Internet-Verbindung oder Zeitraum.")
    if missing_tickers:
        st.warning("Keine Daten für: " + ", ".join(missing_tickers))
    st.stop()

if missing_tickers:
    st.warning(
        "Folgende Ticker konnten nicht geladen werden und wurden ignoriert: "
        + ", ".join(missing_tickers)
    )

returns = prices.pct_change().dropna()

# Benchmarks (Close)
bench_spx = load_benchmark_series(benchmark_ticker_spx, start_date, end_date)
bench_dax = load_benchmark_series(benchmark_ticker_dax, start_date, end_date)

bench_returns_spx = bench_spx.pct_change().dropna() if bench_spx is not None else None
bench_returns_dax = bench_dax.pct_change().dropna() if bench_dax is not None else None

if bench_returns_spx is None:
    st.warning("Benchmark S&P500 konnte nicht geladen werden.")
if bench_returns_dax is None:
    st.warning("Benchmark DAX konnte nicht geladen werden.")

st.subheader("Basisdaten")
st.write(f"Zeitraum: {start_date.date()} bis {end_date.date()}  •  {returns.shape[0]} Handelstage")
st.write(f"Aktive Titel im Universe: {', '.join(returns.columns)}")

# ─────────────────────────────────────────────
# Einzelwert-Kennzahlen inkl. Beta/Alpha
# ─────────────────────────────────────────────
rows = []
for col in returns.columns:
    r = returns[col]
    beta_spx, alpha_spx = (np.nan, np.nan)
    beta_dax, alpha_dax = (np.nan, np.nan)

    if bench_returns_spx is not None:
        beta_spx, alpha_spx = beta_alpha(r, bench_returns_spx, risk_free_annual)
    if bench_returns_dax is not None:
        beta_dax, alpha_dax = beta_alpha(r, bench_returns_dax, risk_free_annual)

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

st.subheader("Einzelwert-Kennzahlen (inkl. Beta/Alpha vs. S&P500 & DAX)")
st.dataframe(display_df, use_container_width=True)

# ─────────────────────────────────────────────
# Portfolio: Equal vs. Optimized
# ─────────────────────────────────────────────
n_assets = returns.shape[1]
w_equal = np.ones(n_assets) / n_assets

objective_label = "Sortino" if optimize_objective.startswith("Sortino") else "Sharpe"

st.subheader(f"Portfolio-Optimierung ({objective_label}-maximierend, min. Gewicht 5%)")

w_opt = optimize_weights(
    returns,
    risk_free_annual,
    objective="sortino" if objective_label == "Sortino" else "sharpe",
    long_only=True,
    min_weight=MIN_WEIGHT
)

weights_df = pd.DataFrame({
    "Ticker": returns.columns,
    "Equal Weight": w_equal,
    "Optimized Weight": w_opt
}).set_index("Ticker")

for c in ["Equal Weight", "Optimized Weight"]:
    weights_df[c] = weights_df[c].apply(lambda x: f"{100*x:,.2f}%")

st.write("Gewichte (gleichgewichtet vs. optimiert):")
st.dataframe(weights_df, use_container_width=True)

# Kennzahlen für beide Portfolios
port_equal_stats = portfolio_stats(
    w_equal, returns, risk_free_annual, bench_returns_spx, bench_returns_dax
)
port_opt_stats = portfolio_stats(
    w_opt, returns, risk_free_annual, bench_returns_spx, bench_returns_dax
)

def stats_to_df(label, stats_dict):
    row = stats_dict.copy()
    row["Portfolio"] = label
    return row

stats_df = pd.DataFrame([
    stats_to_df("Equal Weight", port_equal_stats),
    stats_to_df("Optimized", port_opt_stats),
]).set_index("Portfolio")

# Formatierung
for c in ["Ann. Return", "Ann. Vol", "Max Drawdown", "Hit Ratio", "Alpha_SPX", "Alpha_DAX"]:
    stats_df[c] = stats_df[c].apply(fmt_pct)

for c in ["Sharpe", "Sortino", "Beta_SPX", "Beta_DAX"]:
    stats_df[c] = stats_df[c].apply(fmt_ratio)

st.write("Portfolio-Kennzahlen (Equal vs. Optimized):")
st.dataframe(stats_df, use_container_width=True)

# ─────────────────────────────────────────────
# Risikokennzahlen der Optimierung (Explizit)
# ─────────────────────────────────────────────
st.subheader("Risikokennzahlen des optimierten Portfolios")

col_a, col_b, col_c = st.columns(3)
col_d, col_e, col_f = st.columns(3)

col_a.metric(
    "Ann. Return (Optimized)",
    fmt_pct(port_opt_stats["Ann. Return"]),
    delta=fmt_pct(port_opt_stats["Ann. Return"] - port_equal_stats["Ann. Return"])
)

col_b.metric(
    "Ann. Vol (Optimized)",
    fmt_pct(port_opt_stats["Ann. Vol"]),
    delta=fmt_pct(port_opt_stats["Ann. Vol"] - port_equal_stats["Ann. Vol"])
)

col_c.metric(
    "Max Drawdown (Optimized)",
    fmt_pct(port_opt_stats["Max Drawdown"]),
    delta=fmt_pct(port_opt_stats["Max Drawdown"] - port_equal_stats["Max Drawdown"])
)

col_d.metric(
    "Sharpe (Optimized)",
    fmt_ratio(port_opt_stats["Sharpe"]),
    delta=fmt_ratio(port_opt_stats["Sharpe"] - port_equal_stats["Sharpe"])
)

col_e.metric(
    "Sortino (Optimized)",
    fmt_ratio(port_opt_stats["Sortino"]),
    delta=fmt_ratio(port_opt_stats["Sortino"] - port_equal_stats["Sortino"])
)

col_f.metric(
    "Hit Ratio (Optimized)",
    fmt_pct(port_opt_stats["Hit Ratio"]),
    delta=fmt_pct(port_opt_stats["Hit Ratio"] - port_equal_stats["Hit Ratio"])
)

# ─────────────────────────────────────────────
# Korrelation & Performance
# ─────────────────────────────────────────────
st.subheader("Korrelation & Risiko-Struktur")

corr = returns.corr()
avg_corr = average_correlation(corr)

col1, col2 = st.columns([2, 1])

with col1:
    st.write("Korrelationsmatrix (Tagesrenditen):")
    st.dataframe(corr.style.format("{:.2f}"), use_container_width=True)

with col2:
    st.metric("Ø Paar-Korrelation im Portfolio", f"{avg_corr:,.2f}")

    port_equal_ret = (returns @ w_equal).dropna()
    port_opt_ret = (returns @ w_opt).dropna()

    cum_equal = (1 + port_equal_ret).cumprod()
    cum_opt = (1 + port_opt_ret).cumprod()

    fig, ax = plt.subplots()
    cum_equal.plot(ax=ax, label="Equal Weight")
    cum_opt.plot(ax=ax, label="Optimized")
    ax.set_title("Kumulierte Performance")
    ax.set_ylabel("Indexiert (Start = 1.0)")
    ax.legend()
    st.pyplot(fig)

st.caption(
    "Alle Kennzahlen rein historisch, keine Garantie für zukünftige Entwicklungen. "
    "Nur für professionelle/qualifizierte Investoren."
)
