# portfolio_risk_tool.py
# Streamlit-Tool für Portfolio-Bewertung & Sortino-Optimierung

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.optimize import minimize

# ---------------------------------
# Globale Parameter
# ---------------------------------
TRADING_DAYS = 252

st.set_page_config(
    page_title="Portfolio Risk & Sortino Optimizer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Portfolio-Bewertung & Sortino-Optimierung")

# ---------------------------------
# Sidebar: Inputs
# ---------------------------------
default_tickers = "REGN,LULU,VOW3.DE,REI,DDL,NOV,SRPT,CAG,CMCSA"

st.sidebar.header("Universe & Einstellungen")

tickers_input = st.sidebar.text_input(
    "Ticker (kommagetrennt)",
    value=default_tickers
)

tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]
if len(tickers) == 0:
    st.error("Bitte mindestens einen Ticker eingeben.")
    st.stop()

years_back = st.sidebar.slider(
    "Lookback (Jahre)",
    min_value=1,
    max_value=10,
    value=3,
    help="Zeitraum für die Historie"
)

end_date = datetime.today()
start_date = end_date - timedelta(days=365 * years_back)

risk_free_annual = st.sidebar.number_input(
    "Risikofreier Zins p.a.",
    min_value=-0.05,
    max_value=0.10,
    value=0.02,
    step=0.005,
    format="%.3f",
    help="Als Dezimalzahl, z.B. 0.02 = 2% p.a."
)

benchmark_ticker = st.sidebar.text_input(
    "Benchmark-Ticker (für Beta/Alpha)",
    value="^GSPC"
)

optimize_objective = st.sidebar.selectbox(
    "Optimierungsziel",
    ["Sortino Ratio (maximieren)", "Sharpe Ratio (maximieren)"],
    index=0
)

# ---------------------------------
# Helper-Funktionen
# ---------------------------------
def load_prices(tickers, start, end):
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )
    # yfinance liefert bei mehreren Tickers MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"].copy()
    else:
        prices = data["Close"].to_frame()
    prices = prices.dropna(how="all")
    return prices


def annualized_return(returns, freq=TRADING_DAYS):
    """Geometrische Jahresrendite einer Serie von Tagesrenditen."""
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    cumulative = (1.0 + returns).prod()
    n = len(returns)
    return cumulative ** (freq / n) - 1.0


def annualized_vol(returns, freq=TRADING_DAYS):
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    return returns.std(ddof=1) * np.sqrt(freq)


def downside_deviation(returns, risk_free_annual, freq=TRADING_DAYS):
    """Annualisierte Downside-Volatilität (nur negative Abweichungen)."""
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    # einfacher Ansatz: risikofreier Zins gleichmäßig auf Tage verteilt
    daily_rf = risk_free_annual / freq
    diff = returns - daily_rf
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
    excess = ann_ret - risk_free_annual
    return excess / ann_vol


def sortino_ratio(returns, risk_free_annual, freq=TRADING_DAYS):
    ann_ret = annualized_return(returns, freq)
    dd = downside_deviation(returns, risk_free_annual, freq)
    if dd == 0 or np.isnan(dd):
        return np.nan
    excess = ann_ret - risk_free_annual
    return excess / dd


def max_drawdown(returns):
    """Maximaler Drawdown aus Tagesrenditen."""
    # Equity-Kurve
    cum = (1.0 + returns).cumprod()
    running_max = cum.cummax()
    drawdown = cum / running_max - 1.0
    return drawdown.min()


def hit_ratio(returns):
    """Anteil positiver Tage."""
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    return (returns > 0).mean()


def beta_alpha(port_ret, bench_ret, risk_free_annual, freq=TRADING_DAYS):
    """Beta & Alpha (CAPM) gegen Benchmark."""
    df = pd.concat([port_ret, bench_ret], axis=1).dropna()
    if df.shape[0] < 10:
        return np.nan, np.nan
    rp = df.iloc[:, 0]
    rb = df.iloc[:, 1]

    # Kovarianz / Varianz für Beta
    cov = np.cov(rp, rb)[0, 1]
    var_b = np.var(rb)
    if var_b == 0:
        return np.nan, np.nan
    beta = cov / var_b

    # Jahresrenditen
    ann_rp = annualized_return(rp, freq)
    ann_rb = annualized_return(rb, freq)
    alpha = (ann_rp - risk_free_annual) - beta * (ann_rb - risk_free_annual)
    return beta, alpha


def portfolio_stats(weights, returns_df, risk_free_annual, bench_ret=None):
    """Kennzahlen für ein Portfolio mit gegebenen Gewichten."""
    w = np.array(weights).reshape(-1, 1)
    port_ret = returns_df @ w
    port_ret = port_ret.squeeze()

    ann_ret = annualized_return(port_ret)
    ann_vol = annualized_vol(port_ret)
    sr = sharpe_ratio(port_ret, risk_free_annual)
    so = sortino_ratio(port_ret, risk_free_annual)
    mdd = max_drawdown(port_ret)
    hr = hit_ratio(port_ret)

    beta, alpha = (np.nan, np.nan)
    if bench_ret is not None:
        beta, alpha = beta_alpha(port_ret, bench_ret, risk_free_annual)

    return {
        "Ann. Return": ann_ret,
        "Ann. Vol": ann_vol,
        "Sharpe": sr,
        "Sortino": so,
        "Max Drawdown": mdd,
        "Hit Ratio": hr,
        "Beta": beta,
        "Alpha": alpha,
    }


def average_correlation(corr_matrix):
    """Durchschnittliche Paar-Korrelation im Portfolio."""
    n = corr_matrix.shape[0]
    if n <= 1:
        return np.nan
    vals = corr_matrix.values
    # obere Dreiecks-Matrix ohne Diagonale
    upper = vals[np.triu_indices(n, k=1)]
    return upper.mean()


def optimize_weights(
    returns_df,
    risk_free_annual,
    objective="sortino",
    long_only=True
):
    """Optimiert Gewichte bzgl. Sortino oder Sharpe."""
    n = returns_df.shape[1]
    w0 = np.ones(n) / n

    bounds = None
    if long_only:
        bounds = [(0.0, 1.0)] * n

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    ]

    def port_ret_from_w(w):
        return returns_df @ np.array(w)

    def neg_sortino(w):
        r = port_ret_from_w(w)
        val = sortino_ratio(r, risk_free_annual)
        # für Minimizer: negative Sortino
        return -val if not np.isnan(val) else 1e6

    def neg_sharpe(w):
        r = port_ret_from_w(w)
        val = sharpe_ratio(r, risk_free_annual)
        return -val if not np.isnan(val) else 1e6

    if objective.lower().startswith("sortino"):
        fun = neg_sortino
    else:
        fun = neg_sharpe

    res = minimize(
        fun,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "disp": False}
    )

    if not res.success:
        st.warning(f"Optimierung nicht konvergent: {res.message}")
        return w0

    w_opt = res.x
    # numerische Rundungsfehler korrigieren
    w_opt = np.clip(w_opt, 0, 1)
    w_opt = w_opt / w_opt.sum()
    return w_opt


# ---------------------------------
# Daten laden
# ---------------------------------
with st.spinner("Lade Kursdaten..."):
    prices = load_prices(tickers, start_date, end_date)

if prices.empty:
    st.error("Keine Kursdaten gefunden. Prüfe Ticker oder Zeitraum.")
    st.stop()

returns = prices.pct_change().dropna()

# Benchmark
bench_returns = None
try:
    bench_prices = yf.download(
        benchmark_ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )["Close"].dropna()
    bench_returns = bench_prices.pct_change().dropna()
except Exception:
    st.warning("Benchmark-Daten konnten nicht geladen werden.")

st.subheader("Basisdaten")
st.write(f"Zeitraum: {start_date.date()} bis {end_date.date()}  •  {returns.shape[0]} Handelstage")
st.write(f"Anzahl Titel: {len(tickers)}")

# ---------------------------------
# Einzelwert-Kennzahlen
# ---------------------------------
asset_rows = []
for col in returns.columns:
    r = returns[col]
    metrics = {
        "Ticker": col,
        "Ann. Return": annualized_return(r),
        "Ann. Vol": annualized_vol(r),
        "Sharpe": sharpe_ratio(r, risk_free_annual),
        "Sortino": sortino_ratio(r, risk_free_annual),
        "Max Drawdown": max_drawdown(r),
        "Hit Ratio": hit_ratio(r),
    }
    asset_rows.append(metrics)

asset_df = pd.DataFrame(asset_rows).set_index("Ticker")

# Prozentformat für Renditen
display_df = asset_df.copy()
for col in ["Ann. Return", "Ann. Vol", "Max Drawdown", "Hit Ratio"]:
    display_df[col] = display_df[col].apply(lambda x: f"{100*x:,.2f}%" if pd.notna(x) else "")

for col in ["Sharpe", "Sortino"]:
    display_df[col] = display_df[col].apply(lambda x: f"{x:,.3f}" if pd.notna(x) else "")

st.subheader("Einzelwert-Kennzahlen")
st.dataframe(display_df, use_container_width=True)

# ---------------------------------
# Portfolio: Gleichgewichtung & Optimierung
# ---------------------------------
n_assets = returns.shape[1]
w_equal = np.ones(n_assets) / n_assets

objective_label = "Sortino" if optimize_objective.startswith("Sortino") else "Sharpe"

st.subheader(f"Portfolio-Optimierung ({objective_label}-maximierend)")

w_opt = optimize_weights(
    returns,
    risk_free_annual,
    objective="sortino" if objective_label == "Sortino" else "sharpe",
    long_only=True
)

weights_df = pd.DataFrame({
    "Ticker": returns.columns,
    "Equal Weight": w_equal,
    "Optimized Weight": w_opt
}).set_index("Ticker")

for col in ["Equal Weight", "Optimized Weight"]:
    weights_df[col] = weights_df[col].apply(lambda x: f"{100*x:,.2f}%")

st.write("Gewichte (gleichgewichtet vs. optimiert):")
st.dataframe(weights_df, use_container_width=True)

# Stats für beide Portfolios
port_equal_stats = portfolio_stats(w_equal, returns, risk_free_annual, bench_returns)
port_opt_stats = portfolio_stats(w_opt, returns, risk_free_annual, bench_returns)

def stats_to_df(label, stats_dict):
    row = stats_dict.copy()
    row["Portfolio"] = label
    return row

stats_df = pd.DataFrame([
    stats_to_df("Equal Weight", port_equal_stats),
    stats_to_df("Optimized", port_opt_stats),
]).set_index("Portfolio")

# Prozentformat für Renditen / Vol / MDD / Hit
for col in ["Ann. Return", "Ann. Vol", "Max Drawdown", "Hit Ratio", "Alpha"]:
    stats_df[col] = stats_df[col].apply(lambda x: f"{100*x:,.2f}%" if pd.notna(x) else "")

for col in ["Sharpe", "Sortino", "Beta"]:
    stats_df[col] = stats_df[col].apply(lambda x: f"{x:,.3f}" if pd.notna(x) else "")

st.write("Portfolio-Kennzahlen (gleichgewichtet vs. optimiert):")
st.dataframe(stats_df, use_container_width=True)

# ---------------------------------
# Korrelation & Risiko-Struktur
# ---------------------------------
st.subheader("Korrelation & Risiko-Struktur")

corr = returns.corr()
avg_corr = average_correlation(corr)

col1, col2 = st.columns([2, 1])

with col1:
    st.write("Korrelationsmatrix (Tagesrenditen):")
    st.dataframe(corr.style.format("{:.2f}"), use_container_width=True)

with col2:
    st.metric("Ø Paar-Korrelation im Portfolio", f"{avg_corr:,.2f}")

    # einfache Grafik: kumulierte Performance equal vs. optimiert
    port_equal_ret = (returns @ w_equal).dropna()
    port_opt_ret = (returns @ w_opt).dropna()

    cum_equal = (1 + port_equal_ret).cumprod()
    cum_opt = (1 + port_opt_ret).cumprod()

    fig, ax = plt.subplots()
    cum_equal.plot(ax=ax, label="Equal Weight")
    cum_opt.plot(ax=ax, label="Optimized")
    ax.set_title("Kumulierte Performance")
    ax.set_ylabel("Wert (Indexiert = 1.0)")
    ax.legend()
    st.pyplot(fig)

st.caption(
    "Hinweis: Alle Berechnungen basieren auf historischen Daten und sind keine Garantie "
    "für zukünftige Entwicklungen. Für professionelle/qualifizierte Investoren."
)
