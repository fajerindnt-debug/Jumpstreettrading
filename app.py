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
def synthetic_ohlcv(n=220, seed=19, start=2350.0):
    rng = np.random.default_rng(seed)
    times = pd.date_range(end=pd.Timestamp.now(tz="UTC").floor("min"), periods=n, freq="5min")
    drift = 0.015 + 0.22 * np.sin(np.arange(n) / 15)
    returns = drift + rng.normal(0, 0.42, n)
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
st.sidebar.caption("Quant Math 3D • Risk Engine • Signal-only")
mode = st.sidebar.selectbox("Data source", ["Demo data (safe default)", "Upload OHLCV CSV", "Twelve Data API"])
symbol = st.sidebar.selectbox("Instrument", ["XAU/USD", "EUR/USD", "BTC/USD"], index=0)
interval_map = {"1 min": "1min", "5 min": "5min", "15 min": "15min", "1 hour": "1h"}
interval_label = st.sidebar.selectbox("Data timeframe", list(interval_map.keys()), index=1)
uploaded = None
if mode == "Upload OHLCV CSV":
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if mode == "Twelve Data API":
    st.sidebar.info("Add TWELVE_DATA_API_KEY under Streamlit app Secrets. API support and quotas depend on your plan.")
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
        market = synthetic_ohlcv()
        data_status = "Fell back to synthetic demo data because CSV could not be read."
elif mode == "Twelve Data API":
    market, data_status = try_twelve_data(symbol, interval_map[interval_label])
    if market is None:
        market = synthetic_ohlcv()
else:
    market = synthetic_ohlcv()

market["time"] = pd.to_datetime(market["time"], utc=True, errors="coerce")
market = market.dropna(subset=["time", "open", "high", "low", "close"]).reset_index(drop=True)
market["return"] = market["close"].pct_change().fillna(0.0)
market["momentum"] = market["close"].diff(5).fillna(0.0)
market["volatility"] = market["return"].rolling(14, min_periods=2).std().fillna(0.0)
market["ema20"] = market["close"].ewm(span=20, adjust=False).mean()
market["ema50"] = market["close"].ewm(span=50, adjust=False).mean()

# --------------------------- HEADER ---------------------------
st.title("JUMP STREET TRADING")
st.caption("XAU/USD Quantitative Research Terminal • Streamlit starter build • Signal-only")
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
tabs = st.tabs(["Market Chart + Risk/Reward", "Quant Math 3D", "Running P&L", "Risk & Validation", "Journal / Settings"])

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
    st.write("Not included: broker order execution, live account positions, verified order-book depth, or a fully validated predictive model.")

st.divider()
st.caption("JUMP STREET TRADING • Prototype for testing • Signal-only • Verify data, broker contract specs, costs, and calculations before any real trading.")
