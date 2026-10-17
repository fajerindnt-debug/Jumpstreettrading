# ============================================================
# JUMP STREET TRADING — Quant 3D / Risk-Reward / Running P&L
# Streamlit-ready starter build. Signal-only; no broker orders.
# Demo mode uses simulated data and simulated positions.
# Optional CSV format: time, open, high, low, close, volume
# Optional Twelve Data API: set TWELVE_DATA_API_KEY in Streamlit Secrets.
# ============================================================

import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="JUMP STREET TRADING",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------- THEMING ---------------------------
DEFAULTS = {
    "bg": "#08111F",
    "panel": "#101C2C",
    "text": "#E8EDF3",
    "muted": "#9AAAC0",
    "bull": "#2589FF",
    "bear": "#E8EDF3",
    "tp": "#16C784",
    "sl": "#EA3943",
    "surface": "#2878D0",
    "accent": "#D4AF37",
    "grid": "#26364B",
}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(f"color_{key}", value)

def color(key):
    return st.session_state.get(f"color_{key}", DEFAULTS[key])

def apply_css():
    st.markdown(
        f"""
        <style>
        .stApp {{ background: {color('bg')}; color: {color('text')}; }}
        [data-testid="stSidebar"] {{ background: {color('panel')}; }}
        [data-testid="stMetric"] {{
            background: {color('panel')}; border: 1px solid {color('grid')};
            padding: 14px; border-radius: 12px;
        }}
        .block-container {{ padding-top: 1.2rem; max-width: 1800px; }}
        h1,h2,h3 {{ color: {color('text')}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

apply_css()

# --------------------------- DATA ---------------------------
def synthetic_ohlcv(n=220, seed=19, start=2350.0, step_vol=0.42, freq="5min"):
    """Generate clearly synthetic OHLCV with scale adjusted for the selected instrument."""
    rng = np.random.default_rng(seed)
    times = pd.date_range(end=pd.Timestamp.now(tz="UTC").floor("min"), periods=n, freq=freq)
    drift = 0.04 * step_vol + 0.5 * step_vol * np.sin(np.arange(n) / 15)
    returns = drift + rng.normal(0, step_vol, n)
    close = start + np.cumsum(returns)
    open_ = np.r_[close[0] - returns[0], close[:-1]]
    wick = rng.uniform(0.08, 0.55, n)
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - wick
    volume = rng.lognormal(5.0, 0.3, n)
    return pd.DataFrame({"time": times, "open": open_, "high": high, "low": low,
                         "close": close, "volume": volume})

def normalize_csv(uploaded):
    df = pd.read_csv(uploaded)
    df.columns = [str(c).strip().lower() for c in df.columns]
    aliases = {"datetime": "time", "timestamp": "time", "date": "time",
               "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns})
    if "close" not in df.columns:
        if "price" in df.columns:
            df["close"] = df["price"]
        else:
            raise ValueError("CSV needs a 'close' or 'price' column.")
    for col in ("open", "high", "low"):
        if col not in df.columns:
            df[col] = df["close"]
    if "volume" not in df.columns:
        df["volume"] = 0.0
    if "time" not in df.columns:
        df["time"] = np.arange(len(df))
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    if len(df) < 20:
        raise ValueError("CSV must contain at least 20 valid rows.")
    return df

def try_twelve_data(symbol, interval, outputsize=250):
    # API key is read from Streamlit Secrets. If absent/failing, caller falls back.
    try:
        api_key = st.secrets.get("TWELVE_DATA_API_KEY", "")
    except Exception:
        api_key = ""
    if not api_key:
        return None, "No TWELVE_DATA_API_KEY secret; using demo data."
    import requests
    params = {"symbol": symbol, "interval": interval, "outputsize": outputsize,
              "apikey": api_key, "format": "JSON", "timezone": "UTC"}
    try:
        response = requests.get("https://api.twelvedata.com/time_series", params=params, timeout=12)
        response.raise_for_status()
        payload = response.json()
        if "values" not in payload:
            return None, str(payload.get("message", payload.get("status", "API returned no values")))
        df = pd.DataFrame(payload["values"]).rename(columns={
            "datetime": "time", "open": "open", "high": "high",
            "low": "low", "close": "close", "volume": "volume"
        })
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        for col in ("open", "high", "low", "close", "volume"):
            if col not in df:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["time", "open", "high", "low", "close"])
        df = df.sort_values("time").reset_index(drop=True)
        if len(df) < 20:
            return None, "API returned fewer than 20 usable candles."
        return df, f"Twelve Data: {len(df)} candles received."
    except Exception as exc:
        return None, f"Twelve Data unavailable ({exc}); using demo data."

# --------------------------- SIDEBAR ---------------------------
st.sidebar.title("JUMP STREET TRADING")
st.sidebar.caption("Multi-Asset Quant Terminal • Quant Math 3D • Signal-only")
mode = st.sidebar.selectbox("Data source", ["Demo data (safe default)", "Upload OHLCV CSV", "Twelve Data API"])

# Display symbol -> API symbol and synthetic demo price scale. API symbol support varies by plan/provider.
INSTRUMENTS = {
    "METALS": {
        "XAU/USD — Gold": ("XAU/USD", 2350.0, 0.42),
        "XAG/USD — Silver": ("XAG/USD", 31.0, 0.10),
        "XPT/USD — Platinum": ("XPT/USD", 1000.0, 2.0),
        "XPD/USD — Palladium": ("XPD/USD", 1000.0, 3.0),
    },
    "ENERGY": {
        "WTI — US Oil": ("WTI", 75.0, 0.35),
        "BRENT — Brent Oil": ("BRENT", 79.0, 0.35),
        "NATGAS — Natural Gas": ("NATGAS", 2.5, 0.035),
    },
    "CRYPTO": {
        "BTC/USD — Bitcoin": ("BTC/USD", 65000.0, 220.0),
        "ETH/USD — Ethereum": ("ETH/USD", 3200.0, 18.0),
        "SOL/USD — Solana": ("SOL/USD", 150.0, 2.0),
        "XRP/USD — XRP": ("XRP/USD", 0.55, 0.012),
    },
    "FOREX MAJOR": {
        "EUR/USD": ("EUR/USD", 1.08, 0.0007),
        "GBP/USD": ("GBP/USD", 1.29, 0.0009),
        "USD/JPY": ("USD/JPY", 150.0, 0.10),
        "USD/CHF": ("USD/CHF", 0.88, 0.0006),
        "AUD/USD": ("AUD/USD", 0.66, 0.0006),
        "NZD/USD": ("NZD/USD", 0.61, 0.0006),
        "USD/CAD": ("USD/CAD", 1.36, 0.0007),
    },
    "INDICES": {
        "US100 — Nasdaq 100": ("NDX", 20000.0, 35.0),
        "US500 — S&P 500": ("SPX", 5500.0, 9.0),
        "US30 — Dow Jones": ("DJI", 40000.0, 70.0),
        "GER40 — DAX": ("DAX", 18500.0, 35.0),
    },
}
label_to_config = {label: config for group in INSTRUMENTS.values() for label, config in group.items()}
selected_label = st.sidebar.selectbox(
    "Instrument", list(label_to_config.keys()), index=0,
    format_func=lambda label: label,
)
symbol, demo_start, demo_step_vol = label_to_config[selected_label]
st.sidebar.caption("Watchlist symbols are available in demo mode. Live API support depends on provider, symbol naming, and subscription.")
interval_map = {"1 min": "1min", "5 min": "5min", "15 min": "15min", "1 hour": "1h"}
interval_label = st.sidebar.selectbox("Data timeframe", list(interval_map.keys()), index=1)
uploaded = None
if mode == "Upload OHLCV CSV":
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if mode == "Twelve Data API":
    st.sidebar.info("Add TWELVE_DATA_API_KEY under Streamlit app Secrets. Not every listed symbol is guaranteed to be supported by your Twelve Data plan.")
refresh_seconds = st.sidebar.slider("Refresh interval (seconds)", 5, 60, 15, 5)
auto_refresh = st.sidebar.toggle("Auto-refresh dashboard", value=False)
st.sidebar.caption("Auto-refresh reruns the app. It does not create a broker connection or execute orders.")

with st.sidebar.expander("Global color customization", expanded=False):
    for key, label in [
        ("bg", "Background"), ("panel", "Panels"), ("text", "Text"),
        ("muted", "Muted text"), ("bull", "Bullish candle"), ("bear", "Bearish candle"),
        ("tp", "Take profit"), ("sl", "Stop loss"), ("surface", "3D surface"),
        ("accent", "Accent"), ("grid", "Grid"),
    ]:
        st.color_picker(label, key=f"color_{key}")

# Load data
data_status = ""
if mode == "Upload OHLCV CSV" and uploaded is not None:
    try:
        market = normalize_csv(uploaded)
        data_status = f"Uploaded CSV: {len(market)} rows."
    except Exception as exc:
        st.error(f"CSV error: {exc}")
        market = synthetic_ohlcv(freq=interval_map[interval_label])
        data_status = "Fell back to synthetic demo data because CSV could not be read."
elif mode == "Twelve Data API":
    market, data_status = try_twelve_data(symbol, interval_map[interval_label])
    if market is None:
        market = synthetic_ohlcv(start=demo_start, step_vol=demo_step_vol, freq=interval_map[interval_label])
        data_status += " | Displaying synthetic demo fallback, NOT live data."
else:
    market = synthetic_ohlcv(start=demo_start, step_vol=demo_step_vol, freq=interval_map[interval_label])
    data_status = f"Synthetic demo data for {selected_label}; not live market data."

market["time"] = pd.to_datetime(market["time"], utc=True, errors="coerce")
market = market.dropna(subset=["time", "open", "high", "low", "close"]).reset_index(drop=True)
market["return"] = market["close"].pct_change().fillna(0.0)
market["momentum"] = market["close"].diff(5).fillna(0.0)
market["volatility"] = market["return"].rolling(14, min_periods=2).std().fillna(0.0)
market["ema20"] = market["close"].ewm(span=20, adjust=False).mean()
market["ema50"] = market["close"].ewm(span=50, adjust=False).mean()

# --------------------------- HEADER ---------------------------
st.title("JUMP STREET TRADING")
st.caption(f"{selected_label} • Multi-Asset Quantitative Research Terminal • Signal-only")
c1, c2, c3, c4 = st.columns(4)
latest = float(market["close"].iloc[-1])
first = float(market["close"].iloc[0])
change_pct = (latest / first - 1) * 100 if first else 0
annual_vol = float(market["return"].std() * np.sqrt(252 * 24 * 12) * 100)
drawdown = market["close"] / market["close"].cummax() - 1
c1.metric("Latest price", f"{latest:,.2f}")
c2.metric("Dataset change", f"{change_pct:+.2f}%")
c3.metric("Annualized volatility proxy", f"{annual_vol:.2f}%")
c4.metric("Max price drawdown", f"{drawdown.min()*100:.2f}%")
st.caption(f"{data_status} | Data mode: {mode} | Last candle timestamp: {market['time'].iloc[-1]}")

# --------------------------- TABS ---------------------------
tabs = st.tabs(["Market Chart + Risk/Reward", "Quant Math 3D", "Running P&L", "Risk & Validation", "Journal / Settings", "Machine Learning Lab", "Advanced Quant 3D", "Signal Engine"])

# --------------------------- RISK REWARD + CANDLESTICK ---------------------------
with tabs[0]:
    left, right = st.columns([3.2, 1.15], gap="large")
    with right:
        st.subheader("Risk / Reward Tool")
        direction = st.radio("Direction", ["LONG", "SHORT"], horizontal=True)
        entry_default = round(latest, 2)
        entry = st.number_input("Entry price", value=entry_default, step=0.1, format="%.2f")
        sl_distance = st.number_input("Stop distance (price units)", min_value=0.01, value=10.0, step=0.1)
        ratio_choice = st.selectbox("Risk : Reward ratio", ["1:1", "1:1.5", "1:2", "1:3", "1:5", "Custom"], index=2)
        if ratio_choice == "Custom":
            rr = st.number_input("Reward multiple", min_value=0.1, max_value=20.0, value=2.0, step=0.1)
        else:
            rr = float(ratio_choice.split(":")[1])
        custom_sl = st.number_input("Custom SL price (optional; 0 = use stop distance)", min_value=0.0, value=0.0, step=0.1)
        custom_tp = st.number_input("Custom TP price (optional; 0 = derive from ratio)", min_value=0.0, value=0.0, step=0.1)
        account_balance = st.number_input("Account balance (USD)", min_value=1.0, value=1000.0, step=50.0)
        risk_pct = st.number_input("Risk per trade (%)", min_value=0.01, max_value=10.0, value=0.5, step=0.1)
        contract_size = st.number_input("Contract size (units per lot)", min_value=0.01, value=100.0, step=1.0,
                                        help="Confirm your broker's contract specification before using live sizing.")
        lot = st.number_input("Lot size (simulation)", min_value=0.001, value=0.01, step=0.001, format="%.3f")
        show_labels = st.checkbox("Show price / percentage labels", True)
        show_box = st.checkbox("Show risk/reward boxes", True)
        apply_custom = st.button("Recalculate setup", type="primary", use_container_width=True)

    sl = float(custom_sl) if custom_sl > 0 else (entry - sl_distance if direction == "LONG" else entry + sl_distance)
    risk_distance = abs(entry - sl)
    if custom_tp > 0:
        tp = float(custom_tp)
    else:
        tp = entry + rr * risk_distance if direction == "LONG" else entry - rr * risk_distance
    reward_distance = abs(tp - entry)
    actual_rr = reward_distance / risk_distance if risk_distance > 0 else 0.0
    # A quote-currency-aware contract conversion can vary by instrument/broker; this is a stated estimate.
    risk_usd_est = risk_distance * contract_size * lot
    reward_usd_est = reward_distance * contract_size * lot
    risk_budget = account_balance * risk_pct / 100
    breakeven = (risk_distance / (risk_distance + reward_distance) * 100) if (risk_distance + reward_distance) > 0 else 0
    rr_valid = (direction == "LONG" and sl < entry < tp) or (direction == "SHORT" and tp < entry < sl)
    if not rr_valid:
        st.warning("SL/TP placement conflicts with the selected direction. Review the setup before using it.")

    with left:
        st.subheader(f"{symbol} • {interval_label} • Market Chart")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=market["time"], open=market["open"], high=market["high"], low=market["low"], close=market["close"],
            increasing_line_color=color("bull"), increasing_fillcolor=color("bull"),
            decreasing_line_color=color("bear"), decreasing_fillcolor=color("bear"),
            name="Price"))
        fig.add_trace(go.Scatter(x=market["time"], y=market["ema20"], name="EMA 20",
                                 line=dict(color=color("surface"), width=1.5)))
        fig.add_trace(go.Scatter(x=market["time"], y=market["ema50"], name="EMA 50",
                                 line=dict(color=color("accent"), width=1.5)))
        x0 = market["time"].iloc[max(0, len(market)-45)]
        x1 = market["time"].iloc[-1] + pd.Timedelta(interval_map[interval_label])
        if show_box:
            top_profit, bottom_profit = max(entry, tp), min(entry, tp)
            top_loss, bottom_loss = max(entry, sl), min(entry, sl)
            fig.add_shape(type="rect", x0=x0, x1=x1, y0=top_profit, y1=bottom_profit,
                          fillcolor=color("tp"), opacity=0.17, line=dict(color=color("tp"), width=1))
            fig.add_shape(type="rect", x0=x0, x1=x1, y0=top_loss, y1=bottom_loss,
                          fillcolor=color("sl"), opacity=0.17, line=dict(color=color("sl"), width=1))
        for y, label, col in [(entry, "ENTRY", color("surface")), (sl, "STOP LOSS", color("sl")), (tp, "TAKE PROFIT", color("tp"))]:
            fig.add_hline(y=y, line_dash="dash", line_color=col, annotation_text=f"{label} {y:,.2f}" if show_labels else label,
                          annotation_font_color=col)
        fig.update_layout(
            height=660, template="plotly_dark", paper_bgcolor=color("bg"), plot_bgcolor=color("bg"),
            font=dict(color=color("text")), margin=dict(l=10, r=10, t=30, b=10),
            xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.02),
        )
        fig.update_xaxes(gridcolor=color("grid"))
        fig.update_yaxes(gridcolor=color("grid"), title="Price")
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displaylogo": False})

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Risk distance", f"{risk_distance:.2f}")
    m2.metric("Reward distance", f"{reward_distance:.2f}")
    m3.metric("Actual R:R", f"1 : {actual_rr:.2f}")
    m4.metric("Break-even win rate", f"{breakeven:.2f}%")
    r1, r2, r3 = st.columns(3)
    r1.metric("Estimated risk", f"${risk_usd_est:,.2f}", help="Estimate based on configured contract size × lot; confirm broker specs.")
    r2.metric("Estimated reward", f"${reward_usd_est:,.2f}")
    r3.metric("Risk budget", f"${risk_budget:,.2f}")
    if risk_usd_est > risk_budget:
        st.warning("Estimated setup risk exceeds the selected account risk budget. Reduce lot size or stop distance.")
    st.caption("This is a planning/simulation tool. It does not submit orders. Broker contract size, tick value, spread and commissions can differ.")

# --------------------------- QUANT MATH 3D ---------------------------
with tabs[1]:
    st.subheader("Quant Math 3D Lab")
    st.caption("The surface below is a data-derived visualization of normalized price, volatility and momentum—not a trained predictive model.")
    f1, f2 = st.columns([1, 3])
    with f1:
        factor_range = st.slider("Factor axis range", 0.5, 4.0, 2.0, 0.5)
        surface_opacity = st.slider("Surface opacity", 0.1, 1.0, 0.82, 0.05)
        point_count = st.slider("Factor resolution", 15, 55, 35, 5)
        use_custom_surface = st.checkbox("Use selected 3D surface color", True)
    px = market["close"].to_numpy()
    vol = market["volatility"].to_numpy()
    mom = market["momentum"].to_numpy()
    idx = np.arange(len(market))
    def safe_zscore(a):
        a = np.nan_to_num(np.asarray(a, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        std = np.std(a)
        return (a - np.mean(a)) / (std if std > 1e-12 else 1.0)
    pz, vz, mz = safe_zscore(px), safe_zscore(vol), safe_zscore(mom)
    factor_axis = np.linspace(-factor_range, factor_range, point_count)
    X, Y = np.meshgrid(idx, factor_axis)
    Z = np.array([pz + 0.22*f*vz + 0.12*(f*f-1)*mz for f in factor_axis])
    path_y = np.clip(0.45*vz + 0.25*mz, -2, 2)
    path_z = pz + 0.22*path_y*vz + 0.12*(path_y**2-1)*mz + 0.35
    fig3 = go.Figure()
    colorscale = [[0.0, color("bg")], [0.35, color("surface")], [0.7, color("accent")], [1.0, color("text")]] if use_custom_surface else "Viridis"
    fig3.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale=colorscale, opacity=surface_opacity,
                              showscale=True, colorbar=dict(title="Response")))
    fig3.add_trace(go.Scatter3d(x=idx, y=path_y, z=path_z, mode="lines",
                                line=dict(color=color("tp"), width=5), name="Data trajectory"))
    fig3.update_layout(height=720, template="plotly_dark", paper_bgcolor=color("bg"),
                       font=dict(color=color("text")), margin=dict(l=0,r=0,t=35,b=0),
                       scene=dict(bgcolor=color("bg"),
                                  xaxis=dict(title="Observation", backgroundcolor=color("bg"), gridcolor=color("grid")),
                                  yaxis=dict(title="Factor / volatility state", backgroundcolor=color("bg"), gridcolor=color("grid")),
                                  zaxis=dict(title="Normalized response", backgroundcolor=color("bg"), gridcolor=color("grid")),
                                  camera=dict(eye=dict(x=1.55,y=1.5,z=0.9))))
    st.plotly_chart(fig3, use_container_width=True, config={"scrollZoom": True, "displaylogo": False})
    # supporting charts
    col_a, col_b = st.columns(2)
    with col_a:
        figv = go.Figure()
        figv.add_trace(go.Scatter(x=market["time"], y=market["volatility"], name="Return volatility",
                                  line=dict(color=color("accent"), width=2)))
        figv.update_layout(title="Volatility proxy", height=330, template="plotly_dark",
                          paper_bgcolor=color("bg"), plot_bgcolor=color("bg"), font=dict(color=color("text")))
        figv.update_xaxes(gridcolor=color("grid")); figv.update_yaxes(gridcolor=color("grid"))
        st.plotly_chart(figv, use_container_width=True)
    with col_b:
        figm = go.Figure()
        figm.add_trace(go.Bar(x=market["time"], y=market["momentum"], marker_color=color("surface"), name="5-bar momentum"))
        figm.update_layout(title="Momentum proxy", height=330, template="plotly_dark",
                          paper_bgcolor=color("bg"), plot_bgcolor=color("bg"), font=dict(color=color("text")))
        figm.update_xaxes(gridcolor=color("grid")); figm.update_yaxes(gridcolor=color("grid"))
        st.plotly_chart(figm, use_container_width=True)

    # Additional quantitative modules: each chart is calculated from the loaded series.
    st.divider()
    st.subheader("Quant Math & ML Research Modules")
    st.caption("These are descriptive research tools, not validated trading signals. Synthetic/demo data remains simulated.")
    qm_tabs = st.tabs([
        "Probability", "Stochastic Paths", "Regime Proxy", "Factor Analysis",
        "Monte Carlo", "Optimization", "Econometrics", "Validation"
    ])
    close_s = market["close"].astype(float).reset_index(drop=True)
    ret_s = close_s.pct_change().dropna()
    ret_mean = float(ret_s.mean()) if len(ret_s) else 0.0
    ret_std = float(ret_s.std(ddof=1)) if len(ret_s) > 1 else 0.0

    with qm_tabs[0]:
        st.markdown("**Return distribution and empirical probability**")
        if len(ret_s):
            hist = go.Figure(go.Histogram(x=ret_s * 100, nbinsx=40, marker_color=color("surface"), opacity=0.85, name="Bar returns"))
            hist.add_vline(x=ret_mean * 100, line_dash="dash", line_color=color("accent"), annotation_text="Mean")
            hist.update_layout(title="Empirical bar-return distribution", xaxis_title="Return (%)", yaxis_title="Frequency", height=420, template="plotly_dark", paper_bgcolor=color("bg"), plot_bgcolor=color("bg"), font=dict(color=color("text")))
            st.plotly_chart(hist, use_container_width=True)
            threshold = st.slider("Positive-return threshold (%)", -2.0, 2.0, 0.0, 0.1, key="prob_threshold")
            p_positive = float((ret_s * 100 > threshold).mean() * 100)
            a, b, c = st.columns(3)
            a.metric("Observed bars", f"{len(ret_s):,}")
            b.metric(f"P(return > {threshold:.1f}%)", f"{p_positive:.1f}%")
            c.metric("Mean bar return", f"{ret_mean*100:.4f}%")
        else:
            st.info("Not enough return observations.")

    with qm_tabs[1]:
        st.markdown("**Stochastic price paths (illustrative Monte Carlo process)**")
        sims = st.slider("Number of simulated paths", 10, 200, 60, 10, key="stoch_sims")
        horizon = st.slider("Steps ahead", 10, 120, 40, 10, key="stoch_horizon")
        seed = st.number_input("Random seed", min_value=1, max_value=999999, value=42, key="stoch_seed")
        rng = np.random.default_rng(int(seed))
        sigma = ret_std if ret_std > 0 else 0.001
        shocks = rng.normal(ret_mean, sigma, size=(horizon, sims))
        paths = latest * np.cumprod(1 + shocks, axis=0)
        path_fig = go.Figure()
        for j in range(sims):
            path_fig.add_trace(go.Scatter(x=np.arange(1, horizon+1), y=paths[:, j], mode="lines", line=dict(width=1, color=color("surface")), opacity=0.28, showlegend=False))
        path_fig.add_trace(go.Scatter(x=np.arange(1, horizon+1), y=np.median(paths, axis=1), mode="lines", line=dict(width=3, color=color("accent")), name="Median path"))
        path_fig.update_layout(title="Simulated price paths — not a forecast", xaxis_title="Future step", yaxis_title="Simulated price", height=430, template="plotly_dark", paper_bgcolor=color("bg"), plot_bgcolor=color("bg"), font=dict(color=color("text")))
        st.plotly_chart(path_fig, use_container_width=True)
        st.caption("Assumes historical bar-return mean and standard deviation remain representative. This assumption can fail badly in real markets.")

    with qm_tabs[2]:
        st.markdown("**Rule-based market regime proxy**")
        window = st.slider("Rolling window", 10, 80, 20, 5, key="regime_window")
        rolling_vol = ret_s.rolling(window, min_periods=max(3, window//3)).std()
        rolling_mom = close_s.pct_change(window).reindex(ret_s.index)
        vol_cut = float(rolling_vol.quantile(0.67)) if rolling_vol.notna().any() else 0.0
        regime = np.where(rolling_vol > vol_cut, "High volatility", np.where(rolling_mom > 0, "Positive momentum", "Negative / flat momentum"))
        regime_df = pd.DataFrame({"time": market["time"].iloc[-len(ret_s):].to_numpy(), "rolling_volatility": rolling_vol.to_numpy(), "momentum": rolling_mom.to_numpy(), "regime": regime}).dropna()
        regime_fig = go.Figure()
        regime_colors = {"High volatility": color("sl"), "Positive momentum": color("tp"), "Negative / flat momentum": color("surface")}
        for name, group in regime_df.groupby("regime"):
            regime_fig.add_trace(go.Scatter(x=group["time"], y=group["rolling_volatility"], mode="markers", marker=dict(size=7, color=regime_colors.get(name, color("surface"))), name=name))
        regime_fig.update_layout(title="Regime proxy by rolling volatility", xaxis_title="Time", yaxis_title="Rolling return volatility", height=420, template="plotly_dark", paper_bgcolor=color("bg"), plot_bgcolor=color("bg"), font=dict(color=color("text")))
        st.plotly_chart(regime_fig, use_container_width=True)
        st.caption("Heuristic regime labels only; this is not a trained Hidden Markov Model.")

    with qm_tabs[3]:
        st.markdown("**Factor analysis proxy: standardized price, momentum, volatility**")
        factor_df = pd.DataFrame({"price_z": safe_zscore(close_s), "momentum_z": safe_zscore(market["momentum"]), "volatility_z": safe_zscore(market["volatility"])})
        corr = factor_df.corr()
        heat = go.Figure(go.Heatmap(z=corr.to_numpy(), x=corr.columns, y=corr.index, zmin=-1, zmax=1, colorscale="RdBu", text=np.round(corr.to_numpy(), 2), texttemplate="%{text}", colorbar=dict(title="Correlation")))
        heat.update_layout(title="Correlation matrix (descriptive, not causal)", height=430, template="plotly_dark", paper_bgcolor=color("bg"), font=dict(color=color("text")))
        st.plotly_chart(heat, use_container_width=True)
        st.caption("PCA and formal factor models are not fitted in this prototype; this view shows pairwise correlations of three standardized features.")

    with qm_tabs[4]:
        st.markdown("**Monte Carlo terminal-price distribution**")
        mc_n = st.slider("Simulation count", 500, 10000, 3000, 500, key="mc_n")
        mc_h = st.slider("Horizon (bars)", 5, 100, 20, 5, key="mc_h")
        rng = np.random.default_rng(2026)
        draws = rng.normal(ret_mean, ret_std if ret_std > 0 else 0.001, size=(mc_n, mc_h))
        terminal = latest * np.prod(1 + draws, axis=1)
        mc_fig = go.Figure(go.Histogram(x=terminal, nbinsx=45, marker_color=color("surface"), opacity=0.85))
        mc_fig.add_vline(x=latest, line_dash="dash", line_color=color("accent"), annotation_text="Current reference")
        mc_fig.update_layout(title="Simulated terminal price distribution", xaxis_title="Terminal price", yaxis_title="Frequency", height=420, template="plotly_dark", paper_bgcolor=color("bg"), plot_bgcolor=color("bg"), font=dict(color=color("text")))
        st.plotly_chart(mc_fig, use_container_width=True)
        downside = float(np.quantile(terminal, 0.05))
        st.metric("5th percentile terminal price (scenario estimate)", f"{downside:,.2f}")
        st.caption("Scenario distribution under a simple return model; not a calibrated forecast or guaranteed risk estimate.")

    with qm_tabs[5]:
        st.markdown("**Risk/reward break-even sensitivity**")
        rr_values = np.linspace(0.5, 5.0, 60)
        win_rates = np.linspace(0.1, 0.9, 60)
        RR, WR = np.meshgrid(rr_values, win_rates)
        # Expected R before costs, assuming loss = -1R and win = +RR R.
        ev_surface = WR * RR - (1 - WR)
        opt_fig = go.Figure(go.Surface(x=RR, y=WR*100, z=ev_surface, colorscale="RdYlGn", colorbar=dict(title="Expected R")))
        opt_fig.update_layout(title="Expected R by reward multiple and win rate (before costs)", scene=dict(xaxis_title="Reward multiple", yaxis_title="Win rate (%)", zaxis_title="Expected R / trade"), height=560, template="plotly_dark", paper_bgcolor=color("bg"), font=dict(color=color("text")))
        st.plotly_chart(opt_fig, use_container_width=True)
        st.caption("This is a mathematical sensitivity surface, not an optimization of historical trades. Spread, commission and slippage are excluded.")

    with qm_tabs[6]:
        st.markdown("**Econometrics: rolling return mean and volatility**")
        eco_window = st.slider("Econometrics rolling window", 10, 80, 20, 5, key="eco_window")
        eco = pd.DataFrame({"time": market["time"].iloc[1:].to_numpy(), "return": ret_s.to_numpy()})
        eco["rolling_mean"] = eco["return"].rolling(eco_window, min_periods=max(3, eco_window//3)).mean()
        eco["rolling_std"] = eco["return"].rolling(eco_window, min_periods=max(3, eco_window//3)).std()
        eco_fig = go.Figure()
        eco_fig.add_trace(go.Scatter(x=eco["time"], y=eco["rolling_mean"]*100, mode="lines", line=dict(color=color("tp"), width=2), name="Rolling mean (%)"))
        eco_fig.add_trace(go.Scatter(x=eco["time"], y=eco["rolling_std"]*100, mode="lines", line=dict(color=color("accent"), width=2), name="Rolling std (%)"))
        eco_fig.update_layout(title="Rolling return statistics", xaxis_title="Time", yaxis_title="Percent per bar", height=420, template="plotly_dark", paper_bgcolor=color("bg"), plot_bgcolor=color("bg"), font=dict(color=color("text")))
        st.plotly_chart(eco_fig, use_container_width=True)
        st.caption("Descriptive rolling statistics only; no causal inference or fitted OLS model in this module yet.")

    with qm_tabs[7]:
        st.markdown("**Chronological holdout diagnostic**")
        split_pct = st.slider("Training-period percentage (chronological split)", 50, 90, 75, 5, key="holdout_split")
        split_idx = max(1, min(len(ret_s)-1, int(len(ret_s)*split_pct/100))) if len(ret_s) > 2 else 1
        train_r = ret_s.iloc[:split_idx]
        test_r = ret_s.iloc[split_idx:]
        v1, v2, v3 = st.columns(3)
        v1.metric("Training bars", f"{len(train_r):,}")
        v2.metric("Holdout bars", f"{len(test_r):,}")
        v3.metric("Holdout mean return", f"{test_r.mean()*100:.4f}%" if len(test_r) else "N/A")
        st.caption("This is only a chronological descriptive split. It is not a strategy backtest, walk-forward validation, or evidence of predictive performance.")

# --------------------------- RUNNING P&L ---------------------------
with tabs[2]:
    st.subheader("Running P&L Monitor")
    st.info("Default mode is a simulated paper position. This is not connected to a live broker account.")
    pcol1, pcol2, pcol3 = st.columns(3)
    with pcol1:
        paper_side = st.selectbox("Paper position direction", ["LONG", "SHORT"])
        paper_entry = st.number_input("Paper entry price", value=round(latest, 2), step=0.1, key="paper_entry")
    with pcol2:
        paper_lot = st.number_input("Paper lot size", min_value=0.001, value=0.01, step=0.001, format="%.3f")
        paper_contract = st.number_input("Paper contract size", min_value=0.01, value=100.0, step=1.0)
    with pcol3:
        paper_cost = st.number_input("Estimated round-trip costs (USD)", min_value=0.0, value=0.0, step=0.1)
        paper_mode = st.selectbox("P&L source", ["Latest available market close", "Manual price override"])
    if paper_mode == "Manual price override":
        live_price = st.number_input("Manual current price", value=latest, step=0.1, format="%.2f")
    else:
        live_price = latest
    sign = 1 if paper_side == "LONG" else -1
    gross_pnl = (live_price - paper_entry) * sign * paper_contract * paper_lot
    net_pnl = gross_pnl - paper_cost
    p1, p2, p3 = st.columns(3)
    p1.metric("Latest available price", f"{live_price:,.2f}")
    p2.metric("Gross floating P&L (est.)", f"${gross_pnl:+,.2f}")
    p3.metric("Estimated net P&L", f"${net_pnl:+,.2f}", delta=f"${net_pnl-gross_pnl:+,.2f} costs")
    pnl_path = (market["close"] - paper_entry) * sign * paper_contract * paper_lot - paper_cost
    pnl_color = color("tp") if net_pnl >= 0 else color("sl")
    figp = go.Figure()
    figp.add_trace(go.Scatter3d(x=market["time"], y=market["close"], z=pnl_path,
                                mode="lines", line=dict(color=pnl_color, width=6),
                                name="Illustrative P&L path",
                                hovertemplate="Time %{x}<br>Price %{y:.2f}<br>P&L estimate $%{z:.2f}<extra></extra>"))
    figp.add_trace(go.Scatter3d(x=[market["time"].iloc[-1]], y=[live_price], z=[net_pnl],
                                mode="markers", marker=dict(size=6, color=color("accent")),
                                name="Latest available point"))
    figp.update_layout(title="3D Price × Running P&L", height=650, template="plotly_dark",
                       paper_bgcolor=color("bg"), font=dict(color=color("text")),
                       scene=dict(bgcolor=color("bg"),
                                  xaxis=dict(title="Time", backgroundcolor=color("bg"), gridcolor=color("grid")),
                                  yaxis=dict(title="Price", backgroundcolor=color("bg"), gridcolor=color("grid")),
                                  zaxis=dict(title="Estimated P&L (USD)", backgroundcolor=color("bg"), gridcolor=color("grid"))))
    st.plotly_chart(figp, use_container_width=True, config={"scrollZoom": True, "displaylogo": False})
    st.caption("P&L is an estimate using the configured contract size and lot. It updates when the app receives new data/reruns; demo data is simulated and does not represent live market prices.")
    if auto_refresh:
        st.caption(f"Auto-refresh enabled: rerun approximately every {refresh_seconds}s while this session remains open.")
        time.sleep(refresh_seconds)
        st.rerun()

# --------------------------- RISK / VALIDATION ---------------------------
with tabs[3]:
    st.subheader("Risk & Validation")
    st.warning("These metrics describe the loaded price series; they are not a strategy backtest.")
    rets = market["close"].pct_change().dropna()
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Observations", f"{len(market):,}")
    q2.metric("Mean bar return", f"{rets.mean()*100:.4f}%")
    q3.metric("Return std. dev.", f"{rets.std()*100:.4f}%")
    q4.metric("Max close drawdown", f"{drawdown.min()*100:.2f}%")
    st.markdown("#### Validation modules status")
    status_df = pd.DataFrame([
        ["Walk-forward validation", "Not run", "Requires a defined strategy/model and target labels"],
        ["Out-of-sample testing", "Not run", "Requires a locked final holdout period"],
        ["Net expected value", "Calculator only", "Risk/reward planning available; trade outcome history not yet evaluated"],
        ["Regime & stress testing", "Prototype", "Basic volatility/momentum proxies; full regime backtest not implemented"],
    ], columns=["Module", "Status", "Notes"])
    st.dataframe(status_df, use_container_width=True, hide_index=True)
    st.markdown("#### Rolling volatility")
    roll = market[["time", "volatility"]].copy()
    froll = go.Figure(go.Scatter(x=roll["time"], y=roll["volatility"], mode="lines",
                                 line=dict(color=color("accent"), width=2), name="Rolling volatility"))
    froll.update_layout(height=380, template="plotly_dark", paper_bgcolor=color("bg"),
                        plot_bgcolor=color("bg"), font=dict(color=color("text")))
    froll.update_xaxes(gridcolor=color("grid")); froll.update_yaxes(gridcolor=color("grid"))
    st.plotly_chart(froll, use_container_width=True)

# --------------------------- JOURNAL / SETTINGS ---------------------------
with tabs[4]:
    st.subheader("Journal / Settings")
    st.caption("A starter journal export; no trade is sent to a broker.")
    journal = pd.DataFrame([{
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol, "data_mode": mode, "direction": direction,
        "entry": entry, "stop_loss": sl, "take_profit": tp,
        "risk_reward": actual_rr, "estimated_risk_usd": risk_usd_est,
        "estimated_reward_usd": reward_usd_est, "breakeven_win_rate_pct": breakeven,
        "latest_market_price": latest,
    }])
    st.dataframe(journal, use_container_width=True, hide_index=True)
    st.download_button("Download setup journal CSV", journal.to_csv(index=False).encode("utf-8"),
                       file_name="jump_street_trading_setup_journal.csv", mime="text/csv")
    st.markdown("#### About this build")
    st.write("This is a Streamlit-ready prototype based on the earlier ALPHAQUANT 3D dashboard, extended with a risk/reward planner, configurable colors, and simulated running P&L.")
    st.write("Not included: broker order execution, live account positions, or verified order-book depth. The Machine Learning Lab trains exploratory models, but its demo results are not a validated trading strategy.")

st.divider()
st.caption("JUMP STREET TRADING • Prototype for testing • Signal-only • Verify data, broker contract specs, costs, and calculations before any real trading.")


# --------------------------- MACHINE LEARNING LAB ---------------------------
with tabs[5]:
    st.subheader("Machine Learning Lab")
    st.caption("Exploratory ML trained on the currently loaded OHLCV data. Signal-only; no orders are sent.")
    st.warning("Demo data produces demo-only research. Small datasets, changing market regimes, spread, slippage, and fees can materially change results. This is not a validated trading signal.")

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

        ml = market.copy().sort_values("time").reset_index(drop=True)
        close = ml["close"].astype(float)
        high = ml["high"].astype(float)
        low = ml["low"].astype(float)
        volume = ml["volume"].astype(float)
        ret = close.pct_change()
        ml_features = pd.DataFrame(index=ml.index)
        ml_features["return_1"] = ret
        ml_features["return_3"] = close.pct_change(3)
        ml_features["return_5"] = close.pct_change(5)
        ml_features["momentum_5"] = close.diff(5) / close.shift(5).replace(0, np.nan)
        ml_features["ema_gap_20"] = close / close.ewm(span=20, adjust=False).mean() - 1
        ml_features["ema_gap_50"] = close / close.ewm(span=50, adjust=False).mean() - 1
        ml_features["volatility_10"] = ret.rolling(10).std()
        ml_features["volatility_20"] = ret.rolling(20).std()
        ml_features["range_pct"] = (high - low) / close.replace(0, np.nan)
        ml_features["body_pct"] = (ml["close"] - ml["open"]) / close.replace(0, np.nan)
        ml_features["volume_change"] = volume.pct_change().replace([np.inf, -np.inf], np.nan).clip(-10, 10)
        # Target is the next bar direction; shift(-1) prevents using future return as a feature.
        next_return = close.shift(-1) / close - 1
        target = (next_return > 0).astype(float).where(next_return.notna()).rename("next_bar_up")
        dataset = ml_features.replace([np.inf, -np.inf], np.nan).copy()
        dataset["target"] = target
        dataset = dataset.dropna().reset_index(drop=True)
        dataset["target"] = dataset["target"].astype(int)

        if len(dataset) < 80 or dataset["target"].nunique() < 2:
            st.error(f"Not enough usable, class-diverse data to train the ML models (usable rows: {len(dataset)}). Upload a larger OHLCV dataset, ideally hundreds or thousands of candles.")
        else:
            model_choice = st.selectbox("Model", ["Random Forest", "Logistic Regression", "XGBoost (optional)"])
            test_pct = st.slider("Chronological test set (%)", min_value=15, max_value=40, value=25, step=5)
            threshold = st.slider("Probability threshold for directional label", 0.50, 0.80, 0.55, 0.01)
            feature_cols = list(ml_features.columns)
            split = int(len(dataset) * (1 - test_pct / 100))
            train_df, test_df = dataset.iloc[:split], dataset.iloc[split:]
            X_train, y_train = train_df[feature_cols], train_df["target"]
            X_test, y_test = test_df[feature_cols], test_df["target"]

            if model_choice == "Random Forest":
                model = RandomForestClassifier(n_estimators=250, max_depth=5, min_samples_leaf=4,
                                               class_weight="balanced", random_state=42)
            elif model_choice == "Logistic Regression":
                model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1500, class_weight="balanced"))
            else:
                try:
                    from xgboost import XGBClassifier
                    model = XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.04,
                                          subsample=0.85, colsample_bytree=0.85,
                                          eval_metric="logloss", random_state=42)
                except ImportError:
                    model = None
                    st.error("XGBoost is not installed in this deployment. Add `xgboost` to requirements.txt and redeploy, or select Random Forest / Logistic Regression.")

            if model is not None:
                model.fit(X_train, y_train)
                probabilities = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
                predictions = (probabilities >= threshold).astype(int)
                accuracy = accuracy_score(y_test, predictions)
                precision = precision_score(y_test, predictions, zero_division=0)
                recall = recall_score(y_test, predictions, zero_division=0)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Chronological test accuracy", f"{accuracy * 100:.1f}%")
                m2.metric("Up-class precision", f"{precision * 100:.1f}%")
                m3.metric("Up-class recall", f"{recall * 100:.1f}%")
                m4.metric("Test observations", f"{len(test_df):,}")

                last_x = ml_features.replace([np.inf, -np.inf], np.nan).dropna().iloc[[-1]][feature_cols]
                latest_prob = float(model.predict_proba(last_x)[0][list(model.classes_).index(1)])
                if latest_prob >= threshold:
                    label = "UP-BIAS (research only)"
                elif latest_prob <= 1 - threshold:
                    label = "DOWN-BIAS (research only)"
                else:
                    label = "NO CLEAR BIAS"
                st.markdown("#### Latest model output")
                p1, p2, p3 = st.columns(3)
                p1.metric("Estimated next-bar up probability", f"{latest_prob * 100:.1f}%")
                p2.metric("Estimated down probability", f"{(1-latest_prob) * 100:.1f}%")
                p3.metric("Model classification", label)

                left_ml, right_ml = st.columns(2)
                with left_ml:
                    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
                    cm_fig = go.Figure(go.Heatmap(z=cm, x=["Predicted down", "Predicted up"],
                                                  y=["Actual down", "Actual up"],
                                                  colorscale="Blues", showscale=True,
                                                  text=cm, texttemplate="%{text}"))
                    cm_fig.update_layout(title="Chronological test confusion matrix", height=360,
                                         paper_bgcolor=color("bg"), plot_bgcolor=color("bg"),
                                         font=dict(color=color("text")))
                    st.plotly_chart(cm_fig, use_container_width=True)
                with right_ml:
                    if model_choice == "Random Forest" and hasattr(model, "feature_importances_"):
                        importance = pd.DataFrame({"Feature": feature_cols, "Importance": model.feature_importances_}).sort_values("Importance")
                        imp_fig = go.Figure(go.Bar(x=importance["Importance"], y=importance["Feature"], orientation="h",
                                                   marker_color=color("accent")))
                        imp_fig.update_layout(title="Random Forest feature importance", height=360,
                                              paper_bgcolor=color("bg"), plot_bgcolor=color("bg"),
                                              font=dict(color=color("text")))
                        st.plotly_chart(imp_fig, use_container_width=True)
                    else:
                        st.info("Feature importance chart is currently shown for Random Forest. Logistic Regression uses scaled coefficients; XGBoost has its own importance methods.")

                test_results = pd.DataFrame({"time": ml.loc[dataset.index[split:], "time"].reset_index(drop=True).iloc[:len(y_test)].values
                                             if len(ml) >= len(dataset) else np.arange(len(y_test)),
                                             "actual_next_bar_up": y_test.to_numpy(),
                                             "predicted_next_bar_up": predictions,
                                             "estimated_up_probability": probabilities})
                st.markdown("#### Test predictions")
                st.dataframe(test_results.tail(100), use_container_width=True, hide_index=True)
                st.download_button("Download ML test predictions CSV", test_results.to_csv(index=False).encode("utf-8"),
                                   file_name="jst_ml_test_predictions.csv", mime="text/csv")
                st.caption("Methodology: chronological holdout, next-bar direction target, engineered OHLCV features. No probability calibration, trading-cost-adjusted backtest, or walk-forward validation is claimed here.")
    except ImportError as exc:
        st.error(f"Scikit-learn dependency missing: {exc}. Ensure requirements.txt includes scikit-learn and redeploy.")


# --------------------------- ADVANCED QUANT MATH 3D LAB ---------------------------
with tabs[6]:
    st.subheader("Advanced Quant Mathematics — 3D Laboratory")
    st.caption("Interactive research visualizations. A 3D plot is not evidence of model validity; each panel states whether it is empirical, a mathematical sensitivity surface, or a simulation.")
    st.info("Dark terminal styling and global color controls apply to these charts. The default series may be synthetic unless a real feed or CSV is selected.")

    adv_tabs = st.tabs([
        "Stochastic Calculus", "Volatility Models", "Bayesian Inference", "Regime & Change Points",
        "Copulas & EVT", "Hawkes Processes", "Robust Optimization", "Information Theory", "Statistical Robustness"
    ])
    close_adv = market["close"].astype(float).to_numpy()
    ret_adv = pd.Series(close_adv).pct_change().replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    mu_adv = float(np.nanmean(ret_adv)) if len(ret_adv) else 0.0
    sig_adv = float(np.nanstd(ret_adv, ddof=1)) if len(ret_adv) > 1 else 0.001
    if not np.isfinite(sig_adv) or sig_adv < 1e-8:
        sig_adv = 0.001

    def style_3d(fig, title, x_title, y_title, z_title, height=590):
        fig.update_layout(
            title=title, height=height, template="plotly_dark",
            paper_bgcolor=color("bg"), plot_bgcolor=color("bg"),
            font=dict(color=color("text")), margin=dict(l=0, r=0, t=55, b=0),
            scene=dict(
                bgcolor=color("bg"),
                xaxis=dict(title=x_title, backgroundcolor=color("bg"), gridcolor=color("grid"), zerolinecolor=color("grid")),
                yaxis=dict(title=y_title, backgroundcolor=color("bg"), gridcolor=color("grid"), zerolinecolor=color("grid")),
                zaxis=dict(title=z_title, backgroundcolor=color("bg"), gridcolor=color("grid"), zerolinecolor=color("grid")),
                camera=dict(eye=dict(x=1.55, y=1.55, z=0.95)),
            ),
        )
        return fig

    custom_scale = [[0.0, color("bg")], [0.35, color("surface")], [0.7, color("accent")], [1.0, color("text")]]

    with adv_tabs[0]:
        st.markdown("**Stochastic differential equation / geometric Brownian motion**")
        c1, c2, c3 = st.columns(3)
        with c1: sde_paths = st.slider("Paths", 10, 100, 35, 5, key="adv_sde_paths")
        with c2: sde_steps = st.slider("Steps", 20, 160, 80, 10, key="adv_sde_steps")
        with c3: sde_seed = st.number_input("Simulation seed", 1, 999999, 91, key="adv_sde_seed")
        rng = np.random.default_rng(int(sde_seed))
        dt = 1.0 / max(sde_steps, 1)
        shocks = rng.normal(size=(sde_steps, sde_paths))
        log_paths = np.cumsum((mu_adv - 0.5 * sig_adv**2) * dt + sig_adv * np.sqrt(dt) * shocks, axis=0)
        sde_prices = max(float(close_adv[-1]), 1e-8) * np.exp(log_paths)
        fig = go.Figure()
        for j in range(sde_paths):
            fig.add_trace(go.Scatter3d(x=np.arange(1, sde_steps+1), y=np.full(sde_steps, j), z=sde_prices[:, j], mode="lines", line=dict(color=color("surface"), width=2), opacity=0.45, showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter3d(x=np.arange(1, sde_steps+1), y=np.full(sde_steps, -1), z=np.median(sde_prices, axis=1), mode="lines", line=dict(color=color("accent"), width=6), name="Median simulated path"))
        style_3d(fig, "3D Stochastic Price Paths (GBM-style simulation)", "Future step", "Path index", "Simulated price")
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displaylogo": False})
        st.caption("Simulated GBM-style paths using the loaded series' bar-return mean and volatility as parameters. This is not a fitted forecast and does not capture jumps or changing regimes.")

    with adv_tabs[1]:
        st.markdown("**GARCH-style conditional volatility recursion**")
        g1, g2, g3 = st.columns(3)
        with g1: omega = st.number_input("Omega", min_value=0.00000001, max_value=0.1, value=0.00001, format="%.8f", key="adv_garch_omega")
        with g2: alpha = st.slider("Alpha", 0.01, 0.30, 0.08, 0.01, key="adv_garch_alpha")
        with g3: beta = st.slider("Beta", 0.50, 0.98, 0.88, 0.01, key="adv_garch_beta")
        if alpha + beta >= 0.999:
            st.warning("Alpha + beta is near or above 1; the recursion may be highly persistent/non-stationary.")
        r = np.asarray(ret_adv, dtype=float)
        var = np.full(len(r), max(float(np.var(r)), 1e-8))
        for i in range(1, len(r)):
            var[i] = max(omega + alpha * r[i-1]**2 + beta * var[i-1], 1e-12)
        x = np.arange(len(r))
        pgrid = np.linspace(max(omega, 1e-8), max(omega * 15, float(np.var(r))*0.5, 1e-6), 36)
        bgrid = np.linspace(0.5, 0.98, 36)
        XX, YY = np.meshgrid(pgrid, bgrid)
        ZZ = np.sqrt(np.maximum(omega + alpha * XX + YY * np.var(r), 1e-12))
        fig = go.Figure(go.Surface(x=XX, y=YY, z=ZZ, colorscale=custom_scale, colorbar=dict(title="Volatility"), name="Sensitivity surface"))
        fig.add_trace(go.Scatter3d(x=np.arange(len(var)), y=np.full(len(var), beta), z=np.sqrt(var), mode="lines", line=dict(color=color("tp"), width=5), name="Recursive conditional volatility"))
        style_3d(fig, "GARCH-style Volatility & Parameter Sensitivity", "Variance state / shock proxy", "Beta persistence", "Conditional volatility")
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displaylogo": False})
        st.caption("A transparent GARCH(1,1)-style recursion and sensitivity surface, not a maximum-likelihood GARCH fit. Omega/alpha/beta are user-selected, not estimated parameters.")

    with adv_tabs[2]:
        st.markdown("**Bayesian updating for a positive-return probability**")
        b1, b2, b3 = st.columns(3)
        with b1: prior_a = st.slider("Prior alpha", 0.5, 20.0, 2.0, 0.5, key="adv_bayes_a")
        with b2: prior_b = st.slider("Prior beta", 0.5, 20.0, 2.0, 0.5, key="adv_bayes_b")
        with b3: lookback = st.slider("Observed return lookback", 20, min(200, max(20, len(ret_adv))), min(80, max(20, len(ret_adv))), 10, key="adv_bayes_lookback")
        sample = ret_adv[-int(lookback):] if len(ret_adv) else np.array([])
        wins = int(np.sum(sample > 0)); losses = int(np.sum(sample <= 0))
        post_a, post_b = prior_a + wins, prior_b + losses
        theta = np.linspace(0.001, 0.999, 120)
        aa = np.linspace(max(0.5, prior_a), max(prior_a + 1, prior_a + wins), 50)
        TT, AA = np.meshgrid(theta, aa)
        # Posterior-density surface for beta-binomial family; constants are omitted for display.
        logpdf = (post_a - 1) * np.log(TT) + (post_b - 1) * np.log(1 - TT)
        density = np.exp(logpdf - np.nanmax(logpdf))
        fig = go.Figure(go.Surface(x=TT, y=AA, z=density, colorscale=custom_scale, colorbar=dict(title="Relative density")))
        fig.add_trace(go.Scatter3d(x=[post_a/(post_a+post_b)], y=[post_a], z=[1.0], mode="markers", marker=dict(size=7, color=color("tp")), name="Posterior mean marker"))
        style_3d(fig, "Bayesian Beta-Binomial Probability Surface", "Probability parameter θ", "Alpha coordinate (display axis)", "Relative posterior density")
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displaylogo": False})
        st.metric("Posterior mean P(positive bar)", f"{post_a/(post_a+post_b)*100:.2f}%")
        st.caption(f"Observed positive/ non-positive bars: {wins}/{losses}. Beta prior updates to Beta({post_a:.1f}, {post_b:.1f}); this models bar sign frequency, not trade win probability.")

    with adv_tabs[3]:
        st.markdown("**Market state and change-point proxy in 3D**")
        win = st.slider("State window", 8, 60, 20, 4, key="adv_state_window")
        rr = pd.Series(ret_adv)
        vol_roll = rr.rolling(win, min_periods=max(4, win//2)).std()
        mom_roll = pd.Series(close_adv).pct_change(win).reindex(vol_roll.index)
        state_df = pd.DataFrame({"step": np.arange(len(rr)), "vol": vol_roll, "momentum": mom_roll}).replace([np.inf, -np.inf], np.nan).dropna()
        if len(state_df):
            vcut = state_df["vol"].quantile(0.67)
            state_df["state"] = np.where(state_df["vol"] >= vcut, 2, np.where(state_df["momentum"] > 0, 1, 0))
            state_names = {0: "Negative/flat momentum", 1: "Positive momentum", 2: "High volatility"}
            state_colors = {0: color("bear"), 1: color("tp"), 2: color("sl")}
            fig = go.Figure()
            for state, grp in state_df.groupby("state"):
                fig.add_trace(go.Scatter3d(x=grp["step"], y=grp["momentum"], z=grp["vol"], mode="markers", marker=dict(size=4, color=state_colors[int(state)]), name=state_names[int(state)]))
            style_3d(fig, "3D Rule-Based Market State Map", "Observation", "Rolling momentum", "Rolling volatility")
            st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displaylogo": False})
            st.caption("Rule-based state clustering only. This is not an estimated HMM/HSMM or Bayesian change-point model.")
        else:
            st.warning("Not enough observations for this window.")

    with adv_tabs[4]:
        st.markdown("**Dependence surface and empirical tail-risk view**")
        if len(ret_adv) > 10:
            # Lagged-return dependence as empirical 3D scatter; no claim of causal dependence.
            lag = st.slider("Return lag", 1, min(10, max(1, len(ret_adv)//3)), 1, 1, key="adv_copula_lag")
            a = pd.Series(ret_adv[:-lag]).to_numpy(); b = pd.Series(ret_adv[lag:]).to_numpy()
            count = min(len(a), 900)
            a, b = a[-count:], b[-count:]
            rank_a = pd.Series(a).rank(pct=True).to_numpy(); rank_b = pd.Series(b).rank(pct=True).to_numpy()
            color_val = np.sqrt(a*a + b*b)
            fig = go.Figure(go.Scatter3d(x=rank_a, y=rank_b, z=color_val, mode="markers", marker=dict(size=3, color=color_val, colorscale=custom_scale, opacity=0.7, colorbar=dict(title="Abs. return magnitude")), name="Empirical pseudo-observations"))
            style_3d(fig, "Copula-style Dependence View (rank transformed returns)", "Lagged return rank U", "Return rank V", "Joint magnitude proxy")
            st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displaylogo": False})
            absret = np.abs(ret_adv)
            thresholds = np.linspace(0.5, 0.99, 40)
            quantiles = np.quantile(absret, thresholds)
            horizons = np.arange(1, 31)
            # empirical tail threshold scaled by square-root time as a scenario illustration
            Zx, Zy = np.meshgrid(thresholds*100, horizons)
            Zz = np.outer(np.sqrt(horizons), quantiles)
            evt_fig = go.Figure(go.Surface(x=Zx, y=Zy, z=Zz, colorscale=custom_scale, colorbar=dict(title="Abs. return threshold")))
            style_3d(evt_fig, "Empirical Tail-Threshold Surface (EVT motivation, not fitted EVT)", "Empirical quantile (%)", "Horizon (bars)", "Scaled absolute-return threshold")
            st.plotly_chart(evt_fig, use_container_width=True, config={"scrollZoom": True, "displaylogo": False})
            st.caption("Rank dependence and empirical tail thresholds are descriptive. No parametric copula or Generalized Pareto EVT fit is claimed.")
        else:
            st.warning("Not enough returns to estimate dependence view.")

    with adv_tabs[5]:
        st.markdown("**Hawkes-style self-exciting event intensity simulation**")
        h1, h2, h3 = st.columns(3)
        with h1: base_intensity = st.slider("Baseline intensity", 0.05, 2.0, 0.35, 0.05, key="adv_hawkes_mu")
        with h2: excitation = st.slider("Excitation strength", 0.0, 0.9, 0.45, 0.05, key="adv_hawkes_alpha")
        with h3: decay = st.slider("Decay rate", 0.1, 3.0, 1.2, 0.1, key="adv_hawkes_beta")
        t = np.linspace(0, 20, 500)
        event_times = np.sort(np.random.default_rng(101).uniform(0, 20, 35))
        intensity_line = np.full_like(t, base_intensity)
        for et in event_times:
            intensity_line += excitation * np.exp(-decay * np.maximum(t-et, 0)) * (t >= et)
        lag_axis = np.linspace(0, 4, 45)
        XX, YY = np.meshgrid(t, lag_axis)
        ZZ = base_intensity + excitation * np.exp(-decay * YY) * (1 + np.sin(XX*2.3)**2)
        fig = go.Figure(go.Surface(x=XX, y=YY, z=ZZ, colorscale=custom_scale, opacity=0.88, colorbar=dict(title="Intensity")))
        fig.add_trace(go.Scatter3d(x=t, y=np.zeros_like(t), z=intensity_line, mode="lines", line=dict(color=color("tp"), width=5), name="Illustrative event intensity"))
        fig.add_trace(go.Scatter3d(x=event_times, y=np.zeros_like(event_times), z=np.full_like(event_times, base_intensity), mode="markers", marker=dict(color=color("accent"), size=4), name="Simulated event times"))
        style_3d(fig, "Hawkes-Style Self-Exciting Intensity Surface", "Time", "Lag since event", "Event intensity")
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displaylogo": False})
        st.caption("Illustrative intensity surface with simulated event times; not fitted to exchange trade prints. Real order-flow inference requires timestamped event data.")

    with adv_tabs[6]:
        st.markdown("**Robust optimization landscape**")
        risk_aversion = st.slider("Risk aversion", 0.1, 5.0, 1.5, 0.1, key="adv_opt_risk")
        cost_drag = st.slider("Cost drag (return units)", 0.0, 0.02, 0.002, 0.001, format="%.3f", key="adv_opt_cost")
        exposures = np.linspace(-2, 2, 80)
        uncertainty = np.linspace(0, 3, 70)
        EX, UN = np.meshgrid(exposures, uncertainty)
        # mean-variance utility with an explicit uncertainty penalty and cost drag
        utility = EX * mu_adv - risk_aversion * (EX**2) * (sig_adv**2) - UN * np.abs(EX) * sig_adv - cost_drag*np.abs(EX)
        fig = go.Figure(go.Surface(x=EX, y=UN, z=utility, colorscale=custom_scale, colorbar=dict(title="Utility")))
        style_3d(fig, "Robust Exposure Utility Landscape (illustrative objective)", "Exposure multiplier", "Uncertainty penalty", "Mean-variance utility")
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displaylogo": False})
        best = np.unravel_index(np.nanargmax(utility), utility.shape)
        st.metric("Grid-optimal exposure multiplier", f"{EX[best]:.2f}×")
        st.caption("A toy mean-variance objective with uncertainty and cost penalties; not an executable portfolio allocation recommendation.")

    with adv_tabs[7]:
        st.markdown("**Information entropy and threshold sensitivity**")
        thresholds = np.linspace(0.05, 0.95, 70)
        bin_counts = np.arange(2, 22)
        XX, YY = np.meshgrid(thresholds, bin_counts)
        # Bernoulli entropy for a threshold-dependent positive-event probability proxy
        prob = np.clip(0.5 + (XX - 0.5) * (1 - 1/YY), 1e-6, 1-1e-6)
        entropy = -(prob*np.log2(prob) + (1-prob)*np.log2(1-prob))
        fig = go.Figure(go.Surface(x=XX, y=YY, z=entropy, colorscale=custom_scale, colorbar=dict(title="Entropy (bits)")))
        style_3d(fig, "Entropy Surface: Uncertainty vs Threshold / Binning", "Threshold parameter", "Number of bins", "Bernoulli entropy (bits)")
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displaylogo": False})
        p = float(np.mean(ret_adv > 0)) if len(ret_adv) else 0.5
        p = float(np.clip(p, 1e-9, 1-1e-9))
        ent = -(p*np.log2(p)+(1-p)*np.log2(1-p))
        st.metric("Empirical sign entropy", f"{ent:.4f} bits/bar")
        st.caption("The surface is a mathematical entropy sensitivity illustration; the metric below is computed from observed positive/non-positive bar frequency.")

    with adv_tabs[8]:
        st.markdown("**Statistical robustness / multiple-testing sensitivity**")
        trials = st.slider("Number of tried strategy variants", 1, 500, 50, 1, key="adv_trials")
        observed_sharpe = st.slider("Observed annualized Sharpe (illustrative)", -1.0, 4.0, 1.0, 0.1, key="adv_sharpe")
        sample_n = st.slider("Sample size", 30, 2000, max(30, min(500, len(ret_adv))), 10, key="adv_sample_n")
        corr_penalty = np.linspace(0, 0.99, 70)
        effective_trials = 1 + (trials-1)*(1-corr_penalty)
        # Approximate expected maximum of Gaussian noise trials; educational diagnostic, not formal DSR/PBO.
        expected_noise_sharpe = np.sqrt(2*np.log(np.maximum(effective_trials, 1))) / np.sqrt(max(sample_n, 1)) * np.sqrt(252)
        xs, ys = np.meshgrid(corr_penalty, np.linspace(0.5, 4.0, 60))
        z = ys - (np.sqrt(2*np.log(np.maximum(1+(trials-1)*(1-xs), 1))) / np.sqrt(max(sample_n,1))*np.sqrt(252))
        fig = go.Figure(go.Surface(x=xs, y=ys, z=z, colorscale=custom_scale, colorbar=dict(title="Sharpe less noise proxy")))
        fig.add_trace(go.Scatter3d(x=corr_penalty, y=np.full_like(corr_penalty, observed_sharpe), z=observed_sharpe - expected_noise_sharpe, mode="lines", line=dict(color=color("tp"), width=5), name="Observed Sharpe minus noise proxy"))
        style_3d(fig, "Multiple-Testing / Sharpe Robustness Landscape", "Assumed correlation among trials", "Observed Sharpe", "Sharpe less expected noise proxy")
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displaylogo": False})
        st.caption("Educational approximation only; this is not a formal Deflated Sharpe Ratio or Probability of Backtest Overfitting calculation. Those require the full trial-return matrix and a defined methodology.")

    st.divider()
    st.subheader("3D Visualization Controls")
    st.caption("Global colors can be changed in the sidebar. Use Plotly's toolbar to rotate, zoom, pan, and reset each 3D chart.")
    st.markdown("- Surface palette: background → surface → accent → text.\n- Candlestick colors remain independently configurable in the sidebar.\n- Simulation-based charts are labeled as simulations; empirical charts are derived from the selected dataset.")


# --------------------------- SIGNAL ENGINE / STAGE 5 ---------------------------
with tabs[7]:
    st.subheader("Signal Engine — Research / Paper Signals")
    st.caption("Rule-based signal candidate using the loaded OHLCV series. This is not a guaranteed prediction, and it never places broker orders.")
    if mode == "Twelve Data API" and "synthetic demo fallback" in data_status.lower():
        st.error("The selected API request failed. The chart is using synthetic fallback data; signals below are demo-only.")
    elif mode == "Demo data (safe default)":
        st.warning("Demo mode is synthetic. Any signal is for testing the interface only, not a real market recommendation.")
    else:
        st.info("Verify the data source and timestamp before using these research signals.")

    sig_a, sig_b, sig_c = st.columns(3)
    with sig_a:
        fast_ema_n = st.number_input("Fast EMA", min_value=3, max_value=50, value=12, step=1)
    with sig_b:
        slow_ema_n = st.number_input("Slow EMA", min_value=10, max_value=200, value=26, step=1)
    with sig_c:
        rsi_n = st.number_input("RSI window", min_value=5, max_value=40, value=14, step=1)
    sig_d, sig_e, sig_f = st.columns(3)
    with sig_d:
        signal_threshold = st.slider("Score threshold", 1, 5, 3, help="Rule score, not win probability.")
    with sig_e:
        atr_mult = st.slider("Stop distance (ATR multiple)", 0.5, 4.0, 1.5, 0.25)
    with sig_f:
        reward_multiple = st.select_slider("Target R multiple", options=[1.0, 1.5, 2.0, 3.0, 4.0], value=2.0)

    sigdf = market.copy().reset_index(drop=True)
    close_sig = sigdf["close"].astype(float)
    high_sig = sigdf["high"].astype(float)
    low_sig = sigdf["low"].astype(float)
    ema_fast = close_sig.ewm(span=int(fast_ema_n), adjust=False).mean()
    ema_slow = close_sig.ewm(span=int(slow_ema_n), adjust=False).mean()
    delta = close_sig.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/int(rsi_n), adjust=False, min_periods=int(rsi_n)).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/int(rsi_n), adjust=False, min_periods=int(rsi_n)).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = (100 - 100/(1+rs)).fillna(50).clip(0, 100)
    prev_close = close_sig.shift(1)
    tr = pd.concat([(high_sig-low_sig).abs(), (high_sig-prev_close).abs(), (low_sig-prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=2).mean().bfill().fillna(0.0)
    mom3 = close_sig.pct_change(3).fillna(0)
    trend_up = ema_fast > ema_slow
    trend_down = ema_fast < ema_slow
    rsi_up = (rsi >= 50) & (rsi <= 72)
    rsi_down = (rsi <= 50) & (rsi >= 28)
    mom_up = mom3 > 0
    mom_down = mom3 < 0
    close_above_fast = close_sig > ema_fast
    close_below_fast = close_sig < ema_fast
    # Five transparent rule votes. The score is not a calibrated probability.
    long_score = trend_up.astype(int) + rsi_up.astype(int) + mom_up.astype(int) + close_above_fast.astype(int) + (close_sig.diff() > 0).astype(int)
    short_score = trend_down.astype(int) + rsi_down.astype(int) + mom_down.astype(int) + close_below_fast.astype(int) + (close_sig.diff() < 0).astype(int)
    last_price = float(close_sig.iloc[-1])
    last_atr = float(atr.iloc[-1])
    ls, ss = int(long_score.iloc[-1]), int(short_score.iloc[-1])
    if ls >= signal_threshold and ls > ss:
        sig_label, sig_side, sig_score = "BUY CANDIDATE", "LONG", ls
    elif ss >= signal_threshold and ss > ls:
        sig_label, sig_side, sig_score = "SELL CANDIDATE", "SHORT", ss
    else:
        sig_label, sig_side, sig_score = "NO TRADE", "FLAT", max(ls, ss)
    entry_candidate = last_price
    if sig_side == "LONG":
        sl_candidate = entry_candidate - atr_mult * last_atr
        tp_candidate = entry_candidate + reward_multiple * (entry_candidate - sl_candidate)
    elif sig_side == "SHORT":
        sl_candidate = entry_candidate + atr_mult * last_atr
        tp_candidate = entry_candidate - reward_multiple * (sl_candidate - entry_candidate)
    else:
        sl_candidate = np.nan
        tp_candidate = np.nan
    st.divider()
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Rule-based output", sig_label)
    sc2.metric("Long rule score", f"{ls}/5")
    sc3.metric("Short rule score", f"{ss}/5")
    sc4.metric("ATR(14) proxy", f"{last_atr:,.5f}")
    if sig_side != "FLAT":
        st.dataframe(pd.DataFrame([{
            "Instrument": selected_label, "Direction": sig_side, "Reference entry": entry_candidate,
            "Illustrative stop": sl_candidate, "Illustrative target": tp_candidate,
            "Target R multiple": reward_multiple, "Rule score (not probability)": f"{sig_score}/5",
            "Data mode": mode, "Timestamp": sigdf["time"].iloc[-1]
        }]), use_container_width=True, hide_index=True)
    else:
        st.info("Rules do not currently agree strongly enough. NO TRADE is a valid output.")
    st.markdown("#### Rule breakdown")
    rule_df = pd.DataFrame({
        "Rule": ["Fast EMA vs slow EMA", "RSI regime", "3-bar momentum", "Close vs fast EMA", "Last-bar direction"],
        "Long vote": [bool(trend_up.iloc[-1]), bool(rsi_up.iloc[-1]), bool(mom_up.iloc[-1]), bool(close_above_fast.iloc[-1]), bool((close_sig.diff().iloc[-1] > 0))],
        "Short vote": [bool(trend_down.iloc[-1]), bool(rsi_down.iloc[-1]), bool(mom_down.iloc[-1]), bool(close_below_fast.iloc[-1]), bool((close_sig.diff().iloc[-1] < 0))],
    })
    st.dataframe(rule_df, use_container_width=True, hide_index=True)

    st.markdown("#### Simple historical rule test (illustrative)")
    st.caption("Signal at bar close is applied to the next bar return to reduce look-ahead leakage. This is a simple directional diagnostic, not a full execution simulator; it does not model spread, slippage, commissions, partial fills, or intrabar SL/TP ordering.")
    cost_bps = st.number_input("Assumed round-trip cost (basis points)", min_value=0.0, max_value=100.0, value=5.0, step=1.0)
    min_bars = max(int(slow_ema_n) + 5, 30)
    if len(sigdf) > min_bars:
        side_series = pd.Series(np.where(long_score >= signal_threshold, 1, np.where(short_score >= signal_threshold, -1, 0)), index=sigdf.index)
        # Require the winning side's score to exceed the other side to avoid ambiguous trades.
        side_series = side_series.where(~((long_score >= signal_threshold) & (short_score >= signal_threshold)) | (long_score > short_score), 0)
        side_series = side_series.where(~((long_score >= signal_threshold) & (short_score >= signal_threshold)) | (short_score > long_score), 0)
        next_ret = close_sig.pct_change().shift(-1).fillna(0.0)
        gross = side_series * next_ret
        turnover = side_series.diff().abs().fillna(side_series.abs())
        net = gross - turnover * (float(cost_bps) / 10000.0)
        valid = net.iloc[min_bars:-1]
        eq = (1 + valid).cumprod()
        peak = eq.cummax()
        dd = eq / peak - 1
        trades = side_series.iloc[min_bars:-1].ne(0)
        active_returns = valid[trades.to_numpy()] if len(trades) == len(valid) else valid[valid.index.isin(trades[trades].index)]
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Net cumulative return", f"{(float(eq.iloc[-1])-1)*100:.2f}%" if len(eq) else "N/A")
        h2.metric("Diagnostic max drawdown", f"{float(dd.min())*100:.2f}%" if len(dd) else "N/A")
        h3.metric("Active bars", f"{int((side_series.iloc[min_bars:-1] != 0).sum())}")
        h4.metric("Assumed cost", f"{cost_bps:.1f} bps")
        test_fig = go.Figure()
        test_fig.add_trace(go.Scatter(x=sigdf["time"].iloc[min_bars:-1], y=eq.values, name="Net equity index", line=dict(color=color("tp"), width=2)))
        test_fig.update_layout(title="Illustrative net equity curve (starts at 1.0)", height=350, template="plotly_dark", paper_bgcolor=color("bg"), plot_bgcolor=color("bg"), font=dict(color=color("text")))
        test_fig.update_xaxes(gridcolor=color("grid")); test_fig.update_yaxes(gridcolor=color("grid"))
        st.plotly_chart(test_fig, use_container_width=True)
    else:
        st.info(f"Need more bars for this configuration (at least {min_bars+1}).")
    st.warning("Before live use: verify live data timestamps, symbol mapping, spread/commission/slippage, and test on untouched out-of-sample data. The rule score is not a probability of profit.")
