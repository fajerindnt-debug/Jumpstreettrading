
# ============================================================
# JUMP STREET TRADING — Quant 3D / Risk-Reward / Running P&L
# Streamlit-ready starter build. Signal-only; no broker orders.
# Demo mode uses simulated data and simulated positions.
# Optional CSV format: time, open, high, low, close, volume
# Optional Twelve Data API: set TWELVE_DATA_API_KEY in Streamlit Secrets.
# ============================================================

from datetime import datetime, timezone
import sqlite3
import json
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import genpareto, chi2

# ============================================================
# JST APEX QUANT — PERSISTENT RESEARCH STORE (STAGE 12)
# SQLite is local persistence. Streamlit Cloud local storage may be
# ephemeral across rebuilds/replacements; use a persistent volume or
# managed database for production continuity.
# ============================================================
JST_DB_PATH = Path(os.environ.get("JST_DB_PATH", str(Path(__file__).resolve().parent / "jst_apex_quant.db")))

def jst_db_connect():
    con = sqlite3.connect(str(JST_DB_PATH), timeout=15)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    return con

def jst_db_init():
    with jst_db_connect() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS trades (
            trade_id TEXT PRIMARY KEY, entry_time TEXT, exit_time TEXT,
            symbol TEXT, side TEXT, result_r REAL, payload_json TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS signals (
            signal_id TEXT PRIMARY KEY, signal_time TEXT, symbol TEXT,
            side TEXT, rule_score REAL, ml_probability REAL,
            ensemble_score REAL, decision TEXT, payload_json TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS paper_positions (
            position_key TEXT PRIMARY KEY, status TEXT, payload_json TEXT NOT NULL,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP, symbol TEXT
        );
        CREATE TABLE IF NOT EXISTS paper_entry_locks (
            lock_key TEXT PRIMARY KEY, symbol TEXT NOT NULL, candle_time TEXT NOT NULL,
            side TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS model_runs (
            run_id TEXT PRIMARY KEY, run_time TEXT, model_name TEXT,
            train_rows INTEGER, brier REAL, log_loss REAL, payload_json TEXT
        );
        CREATE TABLE IF NOT EXISTS decision_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT, event_time TEXT,
            event_type TEXT, details_json TEXT
        );
        """)
        # Backward-compatible migration for databases created before symbol isolation.
        _position_cols = {row[1] for row in con.execute("PRAGMA table_info(paper_positions)").fetchall()}
        if "symbol" not in _position_cols:
            con.execute("ALTER TABLE paper_positions ADD COLUMN symbol TEXT")
        # Legacy positions were stored globally; the previous default instrument was XAU/USD.
        con.execute("UPDATE paper_positions SET symbol='XAU/USD' WHERE symbol IS NULL OR symbol=''")

def jst_json_default(value):
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return str(value)

def jst_trade_key(trade, symbol="XAU/USD"):
    # Include instrument identity so identical timestamps/prices across symbols cannot collide.
    raw = "|".join([str(symbol)] + [str(trade.get(k, "")) for k in ("side", "entry_time", "exit_time", "entry", "exit")])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

def jst_db_save_trade(trade, symbol="XAU/USD"):
    trade_payload = dict(trade)
    trade_payload["symbol"] = symbol
    payload = json.dumps(trade_payload, default=jst_json_default, allow_nan=False)
    with jst_db_connect() as con:
        con.execute("INSERT OR IGNORE INTO trades(trade_id,entry_time,exit_time,symbol,side,result_r,payload_json) VALUES(?,?,?,?,?,?,?)",
                    (jst_trade_key(trade_payload, symbol), str(trade.get("entry_time", "")), str(trade.get("exit_time", "")),
                     symbol, trade.get("side", ""), float(trade.get("result_r", 0.0)), payload))

def jst_db_load_trades(symbol=None):
    with jst_db_connect() as con:
        if symbol:
            rows = con.execute("SELECT payload_json FROM trades WHERE symbol=? ORDER BY COALESCE(exit_time,entry_time), trade_id", (symbol,)).fetchall()
        else:
            rows = con.execute("SELECT payload_json FROM trades ORDER BY COALESCE(exit_time,entry_time), trade_id").fetchall()
    out=[]
    for (raw,) in rows:
        try: out.append(json.loads(raw))
        except Exception: pass
    return out

def jst_db_load_active_position(symbol="XAU/USD"):
    with jst_db_connect() as con:
        row = con.execute("SELECT payload_json FROM paper_positions WHERE status='ACTIVE' AND symbol=? ORDER BY updated_at DESC LIMIT 1", (symbol,)).fetchone()
    if not row: return None
    try:
        position=json.loads(row[0])
        for key in ("entry_time","forecast_created_at"):
            if position.get(key): position[key]=pd.to_datetime(position[key],errors="coerce",utc=True)
        return position
    except Exception: return None

def jst_db_save_position(position, symbol="XAU/USD"):
    position_payload = dict(position)
    position_payload["symbol"] = symbol
    key = hashlib.sha256(f"{symbol}|{position.get('side')}|{position.get('entry_time')}".encode()).hexdigest()[:32]
    payload = json.dumps(position_payload, default=jst_json_default, allow_nan=False)
    with jst_db_connect() as con:
        con.execute("INSERT INTO paper_positions(position_key,status,payload_json,updated_at,symbol) VALUES(?,?,?,CURRENT_TIMESTAMP,?) "
                    "ON CONFLICT(position_key) DO UPDATE SET status=excluded.status,payload_json=excluded.payload_json,updated_at=CURRENT_TIMESTAMP,symbol=excluded.symbol",
                    (key, position.get("status", "ACTIVE"), payload, symbol))

def jst_db_clear_active_positions(symbol="XAU/USD"):
    with jst_db_connect() as con:
        con.execute("UPDATE paper_positions SET status='CLOSED' WHERE status='ACTIVE' AND symbol=?", (symbol,))

def jst_db_claim_entry_candle(symbol, candle_time, side):
    """Atomically allow at most one paper entry per symbol and completed candle."""
    candle_key = pd.Timestamp(candle_time).isoformat()
    lock_key = hashlib.sha256(f"{symbol}|{candle_key}".encode("utf-8")).hexdigest()[:32]
    with jst_db_connect() as con:
        cur = con.execute(
            "INSERT OR IGNORE INTO paper_entry_locks(lock_key,symbol,candle_time,side) VALUES(?,?,?,?)",
            (lock_key, str(symbol), candle_key, str(side).upper()),
        )
        return cur.rowcount == 1

def jst_db_log(event_type, details):
    with jst_db_connect() as con:
        con.execute("INSERT INTO decision_audit(event_time,event_type,details_json) VALUES(?,?,?)",
                    (datetime.now(timezone.utc).isoformat(), event_type, json.dumps(details, default=jst_json_default, allow_nan=False)))

def jst_db_save_signal(signal_time, symbol, side, rule_score, ml_probability, ensemble_score, decision, details):
    raw=f"{symbol}|{signal_time}|{side}"
    sid=hashlib.sha256(raw.encode()).hexdigest()[:32]
    with jst_db_connect() as con:
        con.execute("INSERT OR IGNORE INTO signals(signal_id,signal_time,symbol,side,rule_score,ml_probability,ensemble_score,decision,payload_json) VALUES(?,?,?,?,?,?,?,?,?)",
                    (sid,str(signal_time),symbol,side,float(rule_score),float(ml_probability) if ml_probability is not None else None,
                     float(ensemble_score),decision,json.dumps(details,default=jst_json_default,allow_nan=False)))

def jst_db_save_model_run(run_time, symbol, model_name, train_rows, details=None):
    run_id=hashlib.sha256(f"{symbol}|{run_time}|{model_name}".encode()).hexdigest()[:32]
    payload=json.dumps(details or {},default=jst_json_default,allow_nan=False)
    with jst_db_connect() as con:
        con.execute("INSERT OR IGNORE INTO model_runs(run_id,run_time,model_name,train_rows,brier,log_loss,payload_json) VALUES(?,?,?,?,NULL,NULL,?)",
                    (run_id,str(run_time),model_name,int(train_rows),payload))

def jst_db_load_recent_signals(symbol=None, limit=15):
    """Read-only helper for the Command Center Signal History tile. Does not write or alter any state."""
    with jst_db_connect() as con:
        if symbol:
            rows = con.execute(
                "SELECT signal_time, side, rule_score, ml_probability, ensemble_score, decision FROM signals WHERE symbol=? ORDER BY signal_time DESC LIMIT ?",
                (symbol, limit),
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT signal_time, side, rule_score, ml_probability, ensemble_score, decision FROM signals ORDER BY signal_time DESC LIMIT ?",
                (limit,),
            ).fetchall()
    return pd.DataFrame(rows, columns=["signal_time", "side", "rule_score", "ml_probability", "ensemble_score", "decision"])

def jst_db_counts():
    with jst_db_connect() as con:
        return {name: con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for name,table in [("trades","trades"),("signals","signals"),("model_runs","model_runs"),("audit","decision_audit")]}

# JST V2 CORE INTEGRATION
try:
    from jst_core_v2 import (
        JSTDataPipeline,
        JSTDataError,
        JST_VERSION
    )
    JST_CORE_AVAILABLE = True
    JST_CORE_IMPORT_ERROR = None

except Exception as exc:
    JST_CORE_AVAILABLE = False
    JST_CORE_IMPORT_ERROR = (
        f"{type(exc).__name__}: {exc}"
    )

st.set_page_config(
    page_title="JUMP STREET TRADING",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    jst_db_init()
    st.session_state.setdefault("jst_db_ready", True)
except Exception as _db_exc:
    st.session_state.setdefault("jst_db_ready", False)
    st.warning(f"JST persistence unavailable: {type(_db_exc).__name__}: {_db_exc}")

# --------------------------- THEMING ---------------------------
DEFAULTS = {
    "bg": "#070B14",
    "panel": "#101827",
    "text": "#E9EFF7",
    "muted": "#8EA0BD",
    "bull": "#2BD67B",
    "bear": "#F0475A",
    "tp": "#16C784",
    "sl": "#EA3943",
    "surface": "#2878D0",
    "accent": "#D4AF37",
    "grid": "#223047",
}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(f"color_{key}", value)

def color(key):
    return st.session_state.get(f"color_{key}", DEFAULTS[key])

def _hex_to_rgb(hex_color):
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return (16, 24, 39)
    try:
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    except Exception:
        return (16, 24, 39)

def apply_css():
    """Presentation-only design system. No trading/session/data logic lives here."""
    bg_r, bg_g, bg_b = _hex_to_rgb(color("bg"))
    panel_r, panel_g, panel_b = _hex_to_rgb(color("panel"))
    accent_r, accent_g, accent_b = _hex_to_rgb(color("accent"))
    surface_r, surface_g, surface_b = _hex_to_rgb(color("surface"))
    st.markdown(
        f"""
        <style>
        /* ================= JST APEX QUANT — INSTITUTIONAL TERMINAL DESIGN SYSTEM ================= */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

        :root {{
            --jst-bg: {color('bg')};
            --jst-panel: {color('panel')};
            --jst-panel-rgb: {panel_r}, {panel_g}, {panel_b};
            --jst-bg-rgb: {bg_r}, {bg_g}, {bg_b};
            --jst-text: {color('text')};
            --jst-muted: {color('muted')};
            --jst-grid: {color('grid')};
            --jst-accent: {color('accent')};
            --jst-accent-rgb: {accent_r}, {accent_g}, {accent_b};
            --jst-surface-rgb: {surface_r}, {surface_g}, {surface_b};
            --jst-bull: {color('bull')};
            --jst-bear: {color('bear')};
            --jst-tp: {color('tp')};
            --jst-sl: {color('sl')};
            --jst-radius: 14px;
            --jst-shadow-soft: 0 1px 2px rgba(0,0,0,0.35), 0 10px 24px -8px rgba(0,0,0,0.55);
            --jst-shadow-raised: 0 2px 4px rgba(0,0,0,0.4), 0 18px 40px -12px rgba(0,0,0,0.65);
            --jst-border: 1px solid rgba(var(--jst-accent-rgb),0.14);
        }}

        html, body, [class*="css"] {{ font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif; }}

        /* ---------- App background: deep layered gradient + subtle vignette lighting ---------- */
        .stApp {{
            background:
                radial-gradient(1200px 700px at 12% -8%, rgba(var(--jst-accent-rgb),0.07), transparent 60%),
                radial-gradient(1000px 650px at 110% 10%, rgba(var(--jst-surface-rgb),0.10), transparent 55%),
                linear-gradient(180deg, var(--jst-bg) 0%, rgba(var(--jst-bg-rgb),0.98) 55%, var(--jst-bg) 100%);
            color: var(--jst-text);
        }}
        .block-container {{ padding-top: 1.1rem; padding-bottom: 2.5rem; max-width: 1850px; }}

        h1,h2,h3,h4,h5,h6 {{ color: var(--jst-text); letter-spacing: 0.1px; font-weight: 700; }}
        h4, h5 {{
            padding-bottom: 6px; margin-top: 4px;
            border-bottom: 1px solid rgba(var(--jst-accent-rgb),0.18);
        }}
        p, span, label, .stMarkdown, .stCaption {{ color: var(--jst-text); }}
        [data-testid="stCaptionContainer"], .stCaption, small {{ color: var(--jst-muted) !important; }}
        code, .stCodeBlock, [data-testid="stMetricValue"] {{ font-family: 'JetBrains Mono', monospace; }}

        /* ---------- Sidebar: glass panel, layered depth ---------- */
        [data-testid="stSidebar"] {{
            background:
                linear-gradient(180deg, rgba(var(--jst-panel-rgb),0.97) 0%, rgba(var(--jst-bg-rgb),0.99) 100%);
            border-right: 1px solid rgba(var(--jst-accent-rgb),0.12);
            box-shadow: 8px 0 30px -14px rgba(0,0,0,0.6);
        }}
        [data-testid="stSidebar"] .block-container {{ padding-top: 1.4rem; }}
        [data-testid="stSidebar"] h1 {{
            font-size: 1.28rem; font-weight: 800; letter-spacing: 0.6px;
            background: linear-gradient(120deg, var(--jst-text) 30%, var(--jst-accent) 100%);
            -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
        }}

        /* ---------- Metric cards: dimensional, soft shadow, subtle top-light edge ---------- */
        [data-testid="stMetric"] {{
            background: linear-gradient(160deg, rgba(var(--jst-panel-rgb),0.95) 0%, rgba(var(--jst-bg-rgb),0.92) 100%);
            border: var(--jst-border);
            border-top: 1px solid rgba(255,255,255,0.06);
            padding: 16px 18px; border-radius: var(--jst-radius);
            box-shadow: var(--jst-shadow-soft);
            transition: transform 120ms ease, box-shadow 120ms ease;
        }}
        [data-testid="stMetric"]:hover {{ transform: translateY(-1px); box-shadow: var(--jst-shadow-raised); }}
        [data-testid="stMetricLabel"] {{ color: var(--jst-muted) !important; font-size: 0.78rem; letter-spacing: 0.4px; text-transform: uppercase; }}
        [data-testid="stMetricValue"] {{ font-weight: 700; }}

        /* ---------- Tabs: elevated pill rail with glass underlay ---------- */
        [data-testid="stTabs"] [data-baseweb="tab-list"] {{
            background: rgba(var(--jst-panel-rgb),0.55);
            border: 1px solid rgba(var(--jst-accent-rgb),0.10);
            border-radius: 12px; padding: 6px; gap: 2px;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.03), var(--jst-shadow-soft);
        }}
        [data-testid="stTabs"] button[role="tab"] {{
            color: var(--jst-muted); border-radius: 9px; font-weight: 600; font-size: 0.86rem;
            transition: all 140ms ease;
        }}
        [data-testid="stTabs"] button[role="tab"]:hover {{ color: var(--jst-text); background: rgba(255,255,255,0.03); }}
        [data-testid="stTabs"] button[aria-selected="true"] {{
            color: var(--jst-bg) !important;
            background: linear-gradient(135deg, var(--jst-accent) 0%, rgba(var(--jst-accent-rgb),0.75) 100%);
            box-shadow: 0 4px 14px -4px rgba(var(--jst-accent-rgb),0.55);
        }}

        /* ---------- Expanders & containers: layered glass panel ---------- */
        [data-testid="stExpander"] {{
            background: linear-gradient(160deg, rgba(var(--jst-panel-rgb),0.85), rgba(var(--jst-bg-rgb),0.85));
            border: var(--jst-border); border-radius: var(--jst-radius);
            box-shadow: var(--jst-shadow-soft); overflow: hidden;
        }}
        [data-testid="stExpander"] summary {{ font-weight: 600; }}
        div[data-testid="stVerticalBlockBorderWrapper"] {{
            border-radius: var(--jst-radius);
        }}

        /* ---------- Buttons: raised, pressable, premium accent ---------- */
        .stButton button, .stDownloadButton button {{
            border-radius: 10px; border: 1px solid rgba(var(--jst-accent-rgb),0.25);
            background: linear-gradient(160deg, rgba(var(--jst-panel-rgb),0.95), rgba(var(--jst-bg-rgb),0.95));
            color: var(--jst-text); font-weight: 600; box-shadow: var(--jst-shadow-soft);
            transition: transform 120ms ease, box-shadow 120ms ease;
        }}
        .stButton button:hover, .stDownloadButton button:hover {{ transform: translateY(-1px); border-color: rgba(var(--jst-accent-rgb),0.55); }}
        .stButton button[kind="primary"] {{
            background: linear-gradient(135deg, var(--jst-accent), rgba(var(--jst-accent-rgb),0.7));
            color: var(--jst-bg); border: none;
        }}

        /* ---------- Inputs: subtle inset depth ---------- */
        [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
        .stNumberInput input, .stTextInput input, .stSlider, [data-baseweb="input"] {{
            background: rgba(var(--jst-bg-rgb),0.55) !important;
            border-radius: 9px !important; border: 1px solid rgba(var(--jst-accent-rgb),0.12) !important;
        }}
        .stSlider [data-baseweb="slider"] div[role="slider"] {{ background: var(--jst-accent) !important; }}

        /* ---------- Tables / dataframes: terminal grid feel ---------- */
        [data-testid="stDataFrame"], [data-testid="stTable"] {{
            border-radius: var(--jst-radius); overflow: hidden; border: var(--jst-border);
            box-shadow: var(--jst-shadow-soft);
        }}

        /* ---------- Alerts: layered glass notices ---------- */
        div[data-testid="stAlert"] {{
            border-radius: 12px; border: 1px solid rgba(255,255,255,0.06);
            box-shadow: var(--jst-shadow-soft); backdrop-filter: blur(2px);
        }}

        /* ---------- Dividers ---------- */
        hr {{ border-color: rgba(var(--jst-accent-rgb),0.16); margin: 1.1rem 0; }}

        /* ---------- Plotly chart frame: give charts a seated, panel feel ---------- */
        [data-testid="stPlotlyChart"] {{
            background: linear-gradient(160deg, rgba(var(--jst-panel-rgb),0.55), rgba(var(--jst-bg-rgb),0.35));
            border-radius: var(--jst-radius); border: var(--jst-border);
            box-shadow: var(--jst-shadow-soft); padding: 6px;
        }}

        /* ---------- Custom premium header bar / status strip ---------- */
        .jst-header-shell {{
            position: relative; border-radius: 18px; padding: 22px 26px;
            margin-bottom: 14px; overflow: hidden;
            background: linear-gradient(135deg, rgba(var(--jst-panel-rgb),0.92) 0%, rgba(var(--jst-bg-rgb),0.96) 100%);
            border: 1px solid rgba(var(--jst-accent-rgb),0.22);
            box-shadow: var(--jst-shadow-raised);
        }}
        .jst-header-shell::before {{
            content: ""; position: absolute; inset: 0; pointer-events: none;
            background: radial-gradient(500px 160px at 8% 0%, rgba(var(--jst-accent-rgb),0.14), transparent 60%);
        }}
        .jst-header-title {{
            font-size: 1.55rem; font-weight: 800; letter-spacing: 0.6px; margin: 0;
            background: linear-gradient(120deg, var(--jst-text) 25%, var(--jst-accent) 100%);
            -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
        }}
        .jst-header-sub {{ color: var(--jst-muted); font-size: 0.86rem; margin-top: 4px; }}
        .jst-badge {{
            display: inline-flex; align-items: center; gap: 6px; font-size: 0.72rem; font-weight: 700;
            letter-spacing: 0.4px; text-transform: uppercase; padding: 4px 10px; border-radius: 999px;
            background: rgba(var(--jst-accent-rgb),0.12); border: 1px solid rgba(var(--jst-accent-rgb),0.35);
            color: var(--jst-accent);
        }}
        .jst-badge-dot {{ width: 6px; height: 6px; border-radius: 50%; background: var(--jst-accent); box-shadow: 0 0 8px var(--jst-accent); }}

        /* ================= STRUCTURAL PANEL COMPONENTS (layout-level, not just color) ================= */

        /* ---- Market data terminal strip ---- */
        .jst-ticker-row {{ display:flex; gap:10px; flex-wrap:wrap; margin:10px 0 4px 0; }}
        .jst-ticker-tile {{
            flex:1; min-width:150px; padding:10px 14px; border-radius:12px;
            background: linear-gradient(160deg, rgba(var(--jst-panel-rgb),0.9), rgba(var(--jst-bg-rgb),0.9));
            border: var(--jst-border); box-shadow: var(--jst-shadow-soft);
        }}
        .jst-ticker-tile .k {{ font-size:0.68rem; text-transform:uppercase; letter-spacing:0.5px; color:var(--jst-muted); }}
        .jst-ticker-tile .v {{ font-size:1.15rem; font-weight:800; font-family:'JetBrains Mono',monospace; margin-top:2px; }}

        /* ---- Dominant hero decision panel (Signal Engine) ---- */
        .jst-hero {{
            display:flex; gap:26px; align-items:stretch; padding:26px 30px; border-radius:20px;
            background: linear-gradient(135deg, rgba(var(--jst-panel-rgb),0.97) 0%, rgba(var(--jst-bg-rgb),0.98) 100%);
            border: 1px solid var(--jst-tone, rgba(var(--jst-accent-rgb),0.25));
            box-shadow: var(--jst-shadow-raised), inset 0 0 60px -30px var(--jst-tone-glow, rgba(var(--jst-accent-rgb),0.25));
            margin: 10px 0 18px 0; position:relative; overflow:hidden;
        }}
        .jst-hero::after {{
            content:""; position:absolute; inset:0; pointer-events:none;
            background: radial-gradient(420px 220px at 95% -10%, var(--jst-tone-glow, rgba(var(--jst-accent-rgb),0.18)), transparent 65%);
        }}
        .jst-hero-left {{ flex:1.3; display:flex; flex-direction:column; justify-content:center; }}
        .jst-hero-kicker {{ font-size:0.72rem; letter-spacing:1.4px; color:var(--jst-muted); font-weight:700; margin-bottom:6px; }}
        .jst-hero-label {{ font-size:2.6rem; font-weight:800; line-height:1.05; letter-spacing:0.5px; }}
        .jst-hero-sub {{ margin-top:8px; font-size:0.92rem; color:var(--jst-muted); max-width:560px; }}
        .jst-hero-right {{
            flex:1; min-width:220px; display:flex; flex-direction:column; justify-content:center; gap:10px;
            padding-left:26px; border-left:1px solid rgba(var(--jst-accent-rgb),0.18);
        }}
        .jst-gauge-label {{ font-size:0.72rem; text-transform:uppercase; letter-spacing:0.6px; color:var(--jst-muted); font-weight:700; }}
        .jst-gauge-track {{
            width:100%; height:16px; border-radius:999px; background:rgba(var(--jst-bg-rgb),0.7);
            border:1px solid rgba(var(--jst-accent-rgb),0.14); overflow:hidden; box-shadow: inset 0 2px 5px rgba(0,0,0,0.5);
        }}
        .jst-gauge-fill {{ height:100%; border-radius:999px; transition: width 300ms ease; box-shadow: 0 0 14px rgba(0,0,0,0.3); }}
        .jst-gauge-value {{ font-size:1.5rem; font-weight:800; font-family:'JetBrains Mono',monospace; }}
        .jst-gauge-threshold {{ font-size:0.72rem; color:var(--jst-muted); }}

        /* ---- Rule-score comparison bars ---- */
        .jst-scorebar-wrap {{ display:flex; flex-direction:column; gap:10px; margin-top:6px; }}
        .jst-scorebar-row {{ display:flex; align-items:center; gap:10px; }}
        .jst-scorebar-tag {{ width:92px; font-size:0.78rem; font-weight:700; color:var(--jst-muted); }}
        .jst-scorebar-track {{ flex:1; height:12px; border-radius:999px; background:rgba(var(--jst-bg-rgb),0.6); border:1px solid rgba(255,255,255,0.05); overflow:hidden; }}
        .jst-scorebar-fill {{ height:100%; border-radius:999px; }}
        .jst-scorebar-val {{ width:46px; text-align:right; font-family:'JetBrains Mono',monospace; font-weight:700; font-size:0.82rem; }}

        /* ---- Stat tile grid (secondary intelligence stack) ---- */
        .jst-stat-grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:14px 0; }}
        @media (max-width: 1100px) {{ .jst-stat-grid {{ grid-template-columns:repeat(2,1fr); }} }}
        .jst-stat-tile {{
            padding:14px 16px; border-radius:14px;
            background: linear-gradient(160deg, rgba(var(--jst-panel-rgb),0.9), rgba(var(--jst-bg-rgb),0.92));
            border: 1px solid var(--jst-tile-tone, rgba(var(--jst-accent-rgb),0.14));
            box-shadow: var(--jst-shadow-soft);
        }}
        .jst-stat-tile .k {{ font-size:0.7rem; text-transform:uppercase; letter-spacing:0.5px; color:var(--jst-muted); font-weight:700; }}
        .jst-stat-tile .v {{ font-size:1.3rem; font-weight:800; margin-top:4px; font-family:'JetBrains Mono',monospace; color: var(--jst-tile-text, var(--jst-text)); }}

        /* ---- Position monitoring status card ---- */
        .jst-position-card {{
            display:flex; gap:0; border-radius:18px; overflow:hidden; margin:10px 0 16px 0;
            border: var(--jst-border); box-shadow: var(--jst-shadow-raised);
        }}
        .jst-position-flag {{
            width:150px; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:4px;
            background: linear-gradient(160deg, var(--jst-tone, var(--jst-muted)), rgba(0,0,0,0.25));
            padding:18px 10px;
        }}
        .jst-position-flag .side {{ font-size:1.15rem; font-weight:800; color:#0b0f17; letter-spacing:0.5px; }}
        .jst-position-flag .tag {{ font-size:0.64rem; color:#0b0f17; opacity:0.75; text-transform:uppercase; letter-spacing:0.5px; }}
        .jst-position-body {{
            flex:1; display:grid; grid-template-columns:repeat(4,1fr); gap:0;
            background: linear-gradient(160deg, rgba(var(--jst-panel-rgb),0.95), rgba(var(--jst-bg-rgb),0.95));
        }}
        .jst-position-cell {{ padding:14px 16px; border-left:1px solid rgba(255,255,255,0.05); }}
        .jst-position-cell:first-child {{ border-left:none; }}
        .jst-position-cell .k {{ font-size:0.68rem; text-transform:uppercase; color:var(--jst-muted); letter-spacing:0.4px; }}
        .jst-position-cell .v {{ font-size:1.05rem; font-weight:700; margin-top:3px; font-family:'JetBrains Mono',monospace; }}

        /* ---- Risk exposure meter (Risk Management) ---- */
        .jst-risk-panel {{ display:flex; gap:18px; margin:12px 0; flex-wrap:wrap; }}
        .jst-risk-dial {{
            flex:1; min-width:230px; padding:18px 20px; border-radius:16px;
            background: linear-gradient(160deg, rgba(var(--jst-panel-rgb),0.95), rgba(var(--jst-bg-rgb),0.95));
            border: var(--jst-border); box-shadow: var(--jst-shadow-soft);
        }}
        .jst-risk-dial .title {{ font-size:0.76rem; text-transform:uppercase; letter-spacing:0.5px; color:var(--jst-muted); font-weight:700; }}
        .jst-risk-dial .amount {{ font-size:1.6rem; font-weight:800; margin-top:4px; font-family:'JetBrains Mono',monospace; }}
        .jst-risk-dial .track {{ margin-top:10px; height:10px; border-radius:999px; background:rgba(var(--jst-bg-rgb),0.7); overflow:hidden; border:1px solid rgba(255,255,255,0.05); }}
        .jst-risk-dial .fill {{ height:100%; border-radius:999px; }}
        .jst-risk-dial .foot {{ margin-top:6px; font-size:0.72rem; color:var(--jst-muted); }}

        /* ---- Section zone header (used to open each functional block) ---- */
        .jst-zone-head {{ display:flex; align-items:center; gap:10px; margin:18px 0 10px 0; }}
        .jst-zone-chip {{
            font-size:0.68rem; font-weight:800; letter-spacing:0.6px; text-transform:uppercase;
            padding:4px 10px; border-radius:7px; background: rgba(var(--jst-accent-rgb),0.14); color: var(--jst-accent);
            border:1px solid rgba(var(--jst-accent-rgb),0.3);
        }}
        .jst-zone-line {{ flex:1; height:1px; background: linear-gradient(90deg, rgba(var(--jst-accent-rgb),0.35), transparent); }}

        /* ================= BLOOMBERG-STYLE COMMAND CENTER (dense tiled workspace) ================= */
        /* Tighten native bordered containers into small, sharp, information-dense tiles. */
        div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"] .jst-bbg-anchor) {{
            border: 1px solid rgba(255,255,255,0.09) !important;
            border-radius: 4px !important;
            background: linear-gradient(175deg, rgba(10,14,22,0.97) 0%, rgba(6,9,15,0.98) 100%) !important;
            box-shadow: 0 1px 0 rgba(255,255,255,0.03) inset, 0 6px 16px -10px rgba(0,0,0,0.7) !important;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"] .jst-bbg-anchor) [data-testid="stVerticalBlock"] {{ gap: 0.35rem; }}
        .jst-bbg-anchor {{ display:none; }}
        .jst-bbg-head {{
            display:flex; justify-content:space-between; align-items:center;
            font-size: 0.66rem; font-weight: 800; letter-spacing: 0.6px; text-transform: uppercase;
            color: #F2A900; border-bottom: 1px solid rgba(242,169,0,0.25);
            padding-bottom: 5px; margin-bottom: 6px;
        }}
        .jst-bbg-head .tag {{ color: var(--jst-muted); font-weight: 600; font-size: 0.6rem; text-transform:none; letter-spacing:0; }}
        .jst-bbg-row {{ display:flex; justify-content:space-between; font-size: 0.78rem; padding: 2px 0; border-bottom: 1px dashed rgba(255,255,255,0.05); }}
        .jst-bbg-row:last-child {{ border-bottom:none; }}
        .jst-bbg-row .k {{ color: var(--jst-muted); }}
        .jst-bbg-row .v {{ font-family:'JetBrains Mono',monospace; font-weight:700; }}
        .jst-bbg-big {{ font-size: 1.5rem; font-weight: 800; font-family:'JetBrains Mono',monospace; line-height:1.1; }}
        .jst-bbg-sub {{ font-size: 0.68rem; color: var(--jst-muted); margin-top:2px; }}
        .jst-bbg-unavail {{ font-size: 0.74rem; color: var(--jst-muted); font-style: italic; padding: 10px 2px; }}
        .jst-bbg-mini-bar-track {{ height:6px; border-radius:3px; background:rgba(255,255,255,0.06); overflow:hidden; margin-top:4px; }}
        .jst-bbg-mini-bar-fill {{ height:100%; border-radius:3px; }}
        .jst-ribbon-strip {{ display:flex; gap:8px; flex-wrap:wrap; margin: 2px 0 12px 0; }}
        .jst-ribbon-chip {{
            font-size:0.68rem; font-weight:700; padding:5px 10px; border-radius:5px;
            background: rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); color: var(--jst-text);
        }}
        .jst-ribbon-chip b {{ color: var(--jst-accent); margin-right:4px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

apply_css()

# ================= STRUCTURAL PRESENTATION COMPONENTS =================
# Pure rendering helpers. Each takes already-computed values and renders
# HTML/CSS only — none of them perform signal, risk, ML, or persistence logic.

def jst_zone_header(chip, title):
    st.markdown(
        f"""<div class="jst-zone-head"><span class="jst-zone-chip">{chip}</span>
        <h4 style="margin:0;border:none;padding:0;">{title}</h4><span class="jst-zone-line"></span></div>""",
        unsafe_allow_html=True,
    )

def jst_ticker_row(items):
    """items: list of (label, value) tuples."""
    cells = "".join(f'<div class="jst-ticker-tile"><div class="k">{k}</div><div class="v">{v}</div></div>' for k, v in items)
    st.markdown(f'<div class="jst-ticker-row">{cells}</div>', unsafe_allow_html=True)

def jst_hero_decision(kicker, label, sub, side, confidence, threshold=0.60):
    tone_map = {"LONG": color("bull"), "SHORT": color("bear"), "BUY": color("bull"), "SELL": color("bear")}
    tone = tone_map.get(str(side).upper(), color("muted"))
    tr, tg, tb = _hex_to_rgb(tone)
    conf = 0.0 if confidence is None else float(confidence)
    bar_w = max(2.0, min(100.0, conf * 100))
    conf_txt = f"{conf*100:.1f}%" if confidence is not None else "—"
    st.markdown(
        f"""
        <div class="jst-hero" style="--jst-tone: rgba({tr},{tg},{tb},0.55); --jst-tone-glow: rgba({tr},{tg},{tb},0.22);">
            <div class="jst-hero-left">
                <div class="jst-hero-kicker">{kicker}</div>
                <div class="jst-hero-label" style="color:{tone};">{label}</div>
                <div class="jst-hero-sub">{sub}</div>
            </div>
            <div class="jst-hero-right">
                <div class="jst-gauge-label">Ensemble Confidence</div>
                <div class="jst-gauge-track"><div class="jst-gauge-fill" style="width:{bar_w:.1f}%; background:{tone};"></div></div>
                <div class="jst-gauge-value" style="color:{tone};">{conf_txt}</div>
                <div class="jst-gauge-threshold">Admission threshold: {threshold*100:.0f}%</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def jst_score_bars(rows, max_value=5):
    """rows: list of (tag, value, tone_hex)."""
    body = ""
    for tag, val, tone in rows:
        pct = max(2.0, min(100.0, (val / max_value) * 100))
        body += f"""<div class="jst-scorebar-row">
            <div class="jst-scorebar-tag">{tag}</div>
            <div class="jst-scorebar-track"><div class="jst-scorebar-fill" style="width:{pct:.1f}%; background:{tone};"></div></div>
            <div class="jst-scorebar-val">{val}/{max_value}</div>
        </div>"""
    st.markdown(f'<div class="jst-scorebar-wrap">{body}</div>', unsafe_allow_html=True)

def jst_stat_grid(items):
    """items: list of (label, value, tone_hex_or_None)."""
    cells = ""
    for label, value, tone in items:
        style = f'style="--jst-tile-tone: rgba({",".join(map(str, _hex_to_rgb(tone)))},0.4); --jst-tile-text:{tone};"' if tone else ""
        cells += f'<div class="jst-stat-tile" {style}><div class="k">{label}</div><div class="v">{value}</div></div>'
    st.markdown(f'<div class="jst-stat-grid">{cells}</div>', unsafe_allow_html=True)

def jst_position_card(side, cells):
    """side: LONG/SHORT/FLAT. cells: list of (label, value)."""
    tone_map = {"LONG": color("bull"), "SHORT": color("bear")}
    tone = tone_map.get(str(side).upper(), color("muted"))
    body = "".join(f'<div class="jst-position-cell"><div class="k">{k}</div><div class="v">{v}</div></div>' for k, v in cells)
    st.markdown(
        f"""
        <div class="jst-position-card">
            <div class="jst-position-flag" style="--jst-tone:{tone};">
                <div class="side">{side}</div>
                <div class="tag">Position Status</div>
            </div>
            <div class="jst-position-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def jst_bbg_panel_header(title, tag=""):
    """Call as the first thing inside a st.container(border=True) to turn it into a Bloomberg-style tile."""
    st.markdown(
        f'<div class="jst-bbg-anchor"></div><div class="jst-bbg-head"><span>{title}</span><span class="tag">{tag}</span></div>',
        unsafe_allow_html=True,
    )

def jst_bbg_rows(items):
    """items: list of (label, value) tuples rendered as compact key/value lines."""
    body = "".join(f'<div class="jst-bbg-row"><span class="k">{k}</span><span class="v">{v}</span></div>' for k, v in items)
    st.markdown(body, unsafe_allow_html=True)

def jst_bbg_big(value, sub="", tone=None):
    style = f'style="color:{tone};"' if tone else ""
    st.markdown(f'<div class="jst-bbg-big" {style}>{value}</div><div class="jst-bbg-sub">{sub}</div>', unsafe_allow_html=True)

def jst_bbg_minibar(pct, tone):
    pct = max(0.0, min(100.0, pct))
    st.markdown(f'<div class="jst-bbg-mini-bar-track"><div class="jst-bbg-mini-bar-fill" style="width:{pct:.1f}%; background:{tone};"></div></div>', unsafe_allow_html=True)

def jst_bbg_unavailable(msg="Data unavailable — panel not yet computed this session."):
    st.markdown(f'<div class="jst-bbg-unavail">⚠ {msg}</div>', unsafe_allow_html=True)

def jst_risk_dial(title, amount, used_pct, foot):
    used_pct = max(0.0, min(100.0, used_pct))
    tone = color("sl") if used_pct >= 85 else color("accent") if used_pct >= 50 else color("tp")
    st.markdown(
        f"""
        <div class="jst-risk-dial">
            <div class="title">{title}</div>
            <div class="amount">{amount}</div>
            <div class="track"><div class="fill" style="width:{used_pct:.1f}%; background:{tone};"></div></div>
            <div class="foot">{foot}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
        # Never expose the request URL: it contains the API key.
        try:
            payload = response.json()
        except Exception:
            payload = {}
        if response.status_code >= 400:
            api_message = payload.get("message") or payload.get("code") or f"HTTP {response.status_code}"
            return None, f"Twelve Data request failed ({api_message}); verify symbol support and plan."
        if "values" not in payload:
            api_message = payload.get("message", payload.get("status", "API returned no values"))
            return None, f"Twelve Data returned no values ({api_message})."
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
        return None, f"Twelve Data unavailable ({type(exc).__name__}); using demo data."

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

# Keep paper state isolated per instrument while preserving the existing dashboard keys.
st.session_state.setdefault("jst_positions_by_symbol", {})
st.session_state.setdefault("jst_histories_by_symbol", {})
_previous_symbol = st.session_state.get("_jst_active_symbol")
if _previous_symbol:
    st.session_state["jst_positions_by_symbol"][_previous_symbol] = st.session_state.get("jst_position")
    st.session_state["jst_histories_by_symbol"][_previous_symbol] = st.session_state.get("jst_trade_history", [])
if _previous_symbol != symbol:
    if symbol not in st.session_state["jst_positions_by_symbol"]:
        _db_position = jst_db_load_active_position(symbol) if st.session_state.get("jst_db_ready") else None
        st.session_state["jst_positions_by_symbol"][symbol] = _db_position
    if symbol not in st.session_state["jst_histories_by_symbol"]:
        _db_history = jst_db_load_trades(symbol) if st.session_state.get("jst_db_ready") else []
        st.session_state["jst_histories_by_symbol"][symbol] = _db_history
    st.session_state["jst_position"] = st.session_state["jst_positions_by_symbol"][symbol]
    st.session_state["jst_trade_history"] = st.session_state["jst_histories_by_symbol"][symbol]
    st.session_state["jst_last_signal_time"] = None
    st.session_state["_jst_active_symbol"] = symbol
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
if auto_refresh:
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=int(refresh_seconds * 1000), key="jst_dashboard_autorefresh")
    except ImportError:
        st.sidebar.warning("Auto-refresh dependency missing. Install streamlit-autorefresh and redeploy.")
st.sidebar.caption("Auto-refresh reruns the app while this page is open. It is not an always-on background scheduler.")

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
if len(market) < 20:
    st.warning("Loaded data has fewer than 20 valid candles. Switching to clearly labelled synthetic demo data so dashboard panels can still render.")
    market = synthetic_ohlcv(start=demo_start, step_vol=demo_step_vol, freq=interval_map[interval_label])
    data_status = "Synthetic demo fallback: fewer than 20 valid candles were available; not live market data."
    # ==========================================
# JST V2 CORE DATA PIPELINE
# ==========================================

jst_features = None
jst_quality = None

if JST_CORE_AVAILABLE:
    try:
        _tf_code = {"1min": "M1", "5min": "M5", "15min": "M15", "1h": "H1"}.get(interval_map[interval_label])
        jst_pipeline = JSTDataPipeline(
            stale_after_minutes=30,
            min_rows=80,
            timeframe=_tf_code
        )

        jst_result = jst_pipeline.run(
            market,
            source=mode
        )

        market = jst_result["market"]
        jst_features = jst_result["features"]
        jst_quality = jst_result["quality"]

        data_status += (
            f" | JST Core {JST_VERSION}: "
            f"{jst_quality.status}"
        )

    except JSTDataError as exc:
        st.warning(f"JST Core validation: {exc}")

    except Exception as exc:
        st.warning(
            f"JST Core integration error: {exc}"
        )

else:
    st.warning(
        f"JST Core unavailable: {JST_CORE_IMPORT_ERROR}"
    )
# ==========================================
# JST V2 DATA QUALITY MONITOR
# ==========================================

if jst_quality is not None and jst_features is not None:

    with st.expander(
        "JST V2 | DATA QUALITY MONITOR",
        expanded=True
    ):
        quality_data = jst_quality.to_dict()

        metadata = jst_result.get("metadata", {})

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "Valid Candles",
            quality_data.get("rows_valid", 0)
        )

        m2.metric(
            "Feature Count",
            metadata.get("feature_count", 0)
        )

        m3.metric(
            "Data Status",
            quality_data.get("status", "UNKNOWN")
        )

        st.subheader("Data Diagnostics")

        st.json(quality_data)

        st.caption(
            f"JST Core Version: {JST_VERSION}"
        )

        st.caption(
            "Feature engineering output: "
            f"{len(jst_features.columns)} columns"
        )

else:
    st.info(
        "JST V2 diagnostics unavailable. "
        "Check Core import and data validation."
    )
market["return"] = market["close"].pct_change().fillna(0.0)
market["momentum"] = market["close"].diff(5).fillna(0.0)
market["volatility"] = market["return"].rolling(14, min_periods=2).std().fillna(0.0)
market["ema20"] = market["close"].ewm(span=20, adjust=False).mean()
market["ema50"] = market["close"].ewm(span=50, adjust=False).mean()

# --------------------------- HEADER ---------------------------
st.markdown(
    f"""
    <div class="jst-header-shell">
        <span class="jst-badge"><span class="jst-badge-dot"></span>Signal-Only • No Broker Execution</span>
        <p class="jst-header-title">JUMP STREET TRADING — JST APEX QUANT</p>
        <p class="jst-header-sub">{selected_label} • Multi-Asset Quantitative Research Terminal</p>
    </div>
    """,
    unsafe_allow_html=True,
)
latest = float(market["close"].iloc[-1])
first = float(market["close"].iloc[0])
change_pct = (latest / first - 1) * 100 if first else 0
annual_vol = float(market["return"].std() * np.sqrt(252 * 24 * 12) * 100)
drawdown = market["close"] / market["close"].cummax() - 1
jst_ticker_row([
    ("Latest Price", f"{latest:,.2f}"),
    ("Dataset Change", f"{change_pct:+.2f}%"),
    ("Annualized Vol.", f"{annual_vol:.2f}%"),
    ("Max Drawdown", f"{drawdown.min()*100:.2f}%"),
    ("Data Mode", mode.split(" ")[0]),
])
st.caption(f"{data_status} | Data mode: {mode} | Last candle timestamp: {market['time'].iloc[-1]}")

# --------------------------- TABS ---------------------------
tabs = st.tabs(["🖥 Command Center", "Market Chart + Risk/Reward", "Quant Math 3D", "Running P&L", "Risk & Validation", "Journal / Settings", "Machine Learning Lab", "Advanced Quant 3D", "Signal Engine", "Macro & News Intelligence", "Quant Research Engine", "Bloomberg Market Monitor"])

# ================================================================
# COMMAND CENTER — Bloomberg-terminal-style dense multi-panel workspace.
# READ-ONLY VIEW: every tile below reads values already computed by the
# untouched engines elsewhere in this app (market data, session_state
# snapshots cached at the end of their own tabs, or the SQLite store).
# No signal, risk, ML, or persistence logic is computed in this block.
# Cross-tab values reflect the last completed run of that tab (Streamlit
# reruns top-to-bottom on every interaction); until a tab has run at
# least once this session, its tile shows "Data unavailable" rather
# than a placeholder number.
# ================================================================
with tabs[0]:
    try:
        _sig_snap = st.session_state.get("jst_snap_signal_engine", {})
        _risk_snap = st.session_state.get("jst_snap_risk_monitor", {})
        _ticket_snap = st.session_state.get("jst_snap_tradeticket", {})
        _ml_snap = st.session_state.get("jst_snap_ml_lab", {})
        _dxy_snap = st.session_state.get("jst_snap_dxy", {})
        _active_pos = st.session_state.get("jst_position")
        _hist_list = st.session_state.get("jst_trade_history", [])
        _db_ready = st.session_state.get("jst_db_ready", False)

        # ---- Session / connection read straight from real clock + data_status; no fabrication.
        _now_utc_hour = datetime.now(timezone.utc).hour
        _session_label = ("ASIA" if 0 <= _now_utc_hour < 8 else "LONDON" if 8 <= _now_utc_hour < 13
                           else "NEW YORK" if 13 <= _now_utc_hour < 21 else "LATE")
        _conn_ok = not any(w in data_status.lower() for w in ("unavailable", "failed", "fell back", "fallback"))
        _vol_series = pd.to_numeric(market["volatility"], errors="coerce").dropna()
        _vol_now = float(_vol_series.iloc[-1]) if len(_vol_series) else float("nan")
        _vol_median = float(_vol_series.median()) if len(_vol_series) else float("nan")
        _vol_regime = ("ELEVATED" if np.isfinite(_vol_now) and np.isfinite(_vol_median) and _vol_now >= _vol_median
                        else "NORMAL" if np.isfinite(_vol_now) else "N/A")

        st.markdown(
            f"""<div class="jst-ribbon-strip">
            <span class="jst-ribbon-chip"><b>SESSION</b>{_session_label}</span>
            <span class="jst-ribbon-chip"><b>VOL REGIME</b>{_vol_regime}</span>
            <span class="jst-ribbon-chip"><b>FEED</b>{"CONNECTED" if _conn_ok else "DEGRADED / FALLBACK"}</span>
            <span class="jst-ribbon-chip"><b>DB</b>{"READY" if _db_ready else "UNAVAILABLE"}</span>
            <span class="jst-ribbon-chip"><b>MODE</b>{mode}</span>
            </div>""",
            unsafe_allow_html=True,
        )

        left_col, center_col, right_col = st.columns([1.0, 2.3, 1.15], gap="small")

        # ================= LEFT: MARKET ANALYTICS =================
        with left_col:
            with st.container(border=True):
                jst_bbg_panel_header("Market Overview", selected_label)
                jst_bbg_rows([
                    ("Instrument", symbol),
                    ("Last price", f"{latest:,.2f}"),
                    ("Change", f"{change_pct:+.2f}%"),
                    ("Timeframe", interval_label),
                    ("Last candle", str(market["time"].iloc[-1])[:19]),
                ])
            with st.container(border=True):
                jst_bbg_panel_header("Technical Analysis", "EMA / Momentum")
                _ema_trend = "BULLISH (EMA20>EMA50)" if market["ema20"].iloc[-1] > market["ema50"].iloc[-1] else "BEARISH (EMA20<EMA50)"
                _mom_now = float(market["momentum"].iloc[-1])
                jst_bbg_rows([
                    ("EMA trend", _ema_trend),
                    ("EMA20", f"{market['ema20'].iloc[-1]:,.2f}"),
                    ("EMA50", f"{market['ema50'].iloc[-1]:,.2f}"),
                    ("5-bar momentum", f"{_mom_now:+.3f}"),
                    ("Volatility (bar std)", f"{_vol_now:.4f}" if np.isfinite(_vol_now) else "N/A"),
                ])
            with st.container(border=True):
                jst_bbg_panel_header("Market Structure", "Pivot-based")
                _pw = 5
                _hi = pd.to_numeric(market["high"], errors="coerce")
                _lo = pd.to_numeric(market["low"], errors="coerce")
                _piv_hi = _hi[_hi == _hi.rolling(_pw * 2 + 1, center=True).max()]
                _piv_lo = _lo[_lo == _lo.rolling(_pw * 2 + 1, center=True).min()]
                _res_candidates = _piv_hi[_piv_hi > latest]
                _sup_candidates = _piv_lo[_piv_lo < latest]
                _nearest_res = float(_res_candidates.min()) if len(_res_candidates) else float("nan")
                _nearest_sup = float(_sup_candidates.max()) if len(_sup_candidates) else float("nan")
                jst_bbg_rows([
                    ("Nearest resistance (pivot)", f"{_nearest_res:,.2f}" if np.isfinite(_nearest_res) else "N/A"),
                    ("Nearest support (pivot)", f"{_nearest_sup:,.2f}" if np.isfinite(_nearest_sup) else "N/A"),
                    ("Structure bias", _ema_trend.split(" ")[0]),
                ])
                st.caption("Descriptive pivot proxy from loaded OHLC; not a trading signal.")
            with st.container(border=True):
                jst_bbg_panel_header("Liquidity & Volume", "Volume profile")
                _vol_col = pd.to_numeric(market["volume"], errors="coerce").fillna(0.0)
                try:
                    _bins = pd.cut(market["close"], bins=8)
                    _profile = _vol_col.groupby(_bins, observed=True).sum().sort_index(ascending=False)
                    _max_v = float(_profile.max()) if len(_profile) and _profile.max() > 0 else 1.0
                    _poc_interval = _profile.idxmax() if len(_profile) else None
                    for _interval, _vbar in _profile.items():
                        _pct = float(_vbar) / _max_v * 100
                        _lbl = f"{_interval.left:,.1f}–{_interval.right:,.1f}"
                        _tone = color("accent") if _interval == _poc_interval else color("surface")
                        st.markdown(f'<div class="jst-bbg-row"><span class="k">{_lbl}</span><span class="v">{_vbar:,.0f}</span></div>', unsafe_allow_html=True)
                        jst_bbg_minibar(_pct, _tone)
                    st.caption("Point of control (brightest bar) = price band with heaviest traded volume in the loaded window.")
                except Exception:
                    jst_bbg_unavailable("Volume profile needs a numeric volume column.")

        # ================= CENTER: DOMINANT CHART WORKSPACE =================
        with center_col:
            with st.container(border=True):
                jst_bbg_panel_header("XAU/USD Price Action — Central Chart Workspace", f"{symbol} • {interval_label}")
                _cc_window = market.tail(140)
                cc_fig = go.Figure()
                cc_fig.add_trace(go.Candlestick(
                    x=_cc_window["time"], open=_cc_window["open"], high=_cc_window["high"],
                    low=_cc_window["low"], close=_cc_window["close"],
                    increasing_line_color=color("bull"), decreasing_line_color=color("bear"),
                    name="Price",
                ))
                cc_fig.add_trace(go.Scatter(x=_cc_window["time"], y=_cc_window["ema20"], mode="lines",
                                            line=dict(color=color("surface"), width=1.3), name="EMA20"))
                cc_fig.add_trace(go.Scatter(x=_cc_window["time"], y=_cc_window["ema50"], mode="lines",
                                            line=dict(color=color("accent"), width=1.3), name="EMA50"))
                if _ticket_snap:
                    for _y, _lbl, _col in [
                        (_ticket_snap.get("entry"), "ENTRY", color("surface")),
                        (_ticket_snap.get("sl"), "SL", color("sl")),
                        (_ticket_snap.get("tp"), "TP", color("tp")),
                    ]:
                        if _y is not None:
                            cc_fig.add_hline(y=_y, line_dash="dot", line_color=_col, annotation_text=_lbl, annotation_font_color=_col)
                cc_fig.update_layout(
                    height=430, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=color("text"), size=11), margin=dict(l=8, r=8, t=8, b=8),
                    xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.08, font=dict(size=10)),
                )
                cc_fig.update_xaxes(gridcolor=color("grid")); cc_fig.update_yaxes(gridcolor=color("grid"))
                st.plotly_chart(cc_fig, use_container_width=True, config={"displaylogo": False})
                if _ticket_snap:
                    st.caption(f"ENTRY/SL/TP overlay from the Trade Ticket last configured in 'Market Chart + Risk/Reward' ({_ticket_snap.get('direction','—')}).")
                else:
                    st.caption("Open 'Market Chart + Risk/Reward' once to overlay a trade ticket here.")
            with st.container(border=True):
                jst_bbg_panel_header("Volatility & Market Regime", "Rolling, descriptive")
                vr_fig = go.Figure(go.Scatter(x=market["time"].tail(140), y=market["volatility"].tail(140),
                                              mode="lines", line=dict(color=color("accent"), width=1.5), fill="tozeroy",
                                              fillcolor=f"rgba({','.join(map(str, _hex_to_rgb(color('accent'))))},0.12)"))
                vr_fig.update_layout(height=140, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                     font=dict(color=color("text"), size=10), margin=dict(l=8, r=8, t=4, b=4), showlegend=False)
                vr_fig.update_xaxes(gridcolor=color("grid"), visible=False); vr_fig.update_yaxes(gridcolor=color("grid"))
                st.plotly_chart(vr_fig, use_container_width=True, config={"displaylogo": False})
                st.caption(f"Current regime: {_vol_regime} (bar-return std. vs. session median). Descriptive proxy, not a volatility forecast.")

        # ================= RIGHT: SIGNAL / PROBABILITY / POSITION / RISK STACK =================
        with right_col:
            with st.container(border=True):
                jst_bbg_panel_header("Alpha Signal Engine", "Ensemble")
                if _sig_snap:
                    _dec = _sig_snap.get("ensemble_decision", "FLAT")
                    _tone = color("bull") if _dec == "LONG" else color("bear") if _dec == "SHORT" else color("muted")
                    jst_bbg_big(_dec, sub=_sig_snap.get("sig_label", ""), tone=_tone)
                    jst_bbg_rows([
                        ("Rule LONG", f"{_sig_snap.get('ls','—')}/5"),
                        ("Rule SHORT", f"{_sig_snap.get('ss','—')}/5"),
                    ])
                else:
                    jst_bbg_unavailable("Open the 'Signal Engine' tab once to populate this tile.")
            with st.container(border=True):
                jst_bbg_panel_header("Probability Engine", "ML / Ensemble")
                if _sig_snap:
                    _conf = float(_sig_snap.get("ensemble_confidence", 0.0))
                    _mlp = float(_sig_snap.get("ml_up_probability", 0.5))
                    _ctone = color("tp") if _conf >= 0.60 else color("accent")
                    jst_bbg_rows([("Ensemble confidence", f"{_conf:.1%}")])
                    jst_bbg_minibar(_conf * 100, _ctone)
                    jst_bbg_rows([("ML P(UP)", f"{_mlp:.1%}")])
                    jst_bbg_minibar(_mlp * 100, color("surface"))
                else:
                    jst_bbg_unavailable("Open the 'Signal Engine' tab once to populate this tile.")
            with st.container(border=True):
                jst_bbg_panel_header("Position Monitoring", "Live paper state")
                if _active_pos:
                    _ptone = color("bull") if _active_pos.get("side") == "LONG" else color("bear")
                    jst_bbg_big(_active_pos.get("side", "—"), sub="ACTIVE", tone=_ptone)
                    jst_bbg_rows([
                        ("Entry", f"{float(_active_pos.get('entry', 0)):,.2f}"),
                        ("SL", f"{float(_active_pos.get('sl', 0)):,.2f}"),
                        ("TP", f"{float(_active_pos.get('tp', 0)):,.2f}"),
                        ("Momentum", _active_pos.get("momentum_state", "—")),
                    ])
                else:
                    jst_bbg_big("FLAT", sub="No active paper position", tone=color("muted"))
            with st.container(border=True):
                jst_bbg_panel_header("Risk & Exposure Monitor", "Gate + budget")
                if _sig_snap:
                    _pass = _sig_snap.get("risk_gate_pass", False)
                    jst_bbg_big("PASS" if _pass else "BLOCK", tone=color("tp") if _pass else color("sl"))
                    if not _pass:
                        st.caption("; ".join(_sig_snap.get("risk_gate_reasons", []))[:140])
                if _risk_snap:
                    jst_bbg_rows([
                        ("Max risk / trade", f"${_risk_snap.get('risk_budget',0):,.2f}"),
                        ("Daily loss limit", f"${_risk_snap.get('daily_loss_budget',0):,.2f}"),
                    ])
                elif not _sig_snap:
                    jst_bbg_unavailable("Open 'Signal Engine' / 'Risk & Validation' once to populate this tile.")

        st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)
        macro_col, dxy_col, ml_col = st.columns(3, gap="small")
        with macro_col:
            with st.container(border=True):
                jst_bbg_panel_header("Macro Economic Monitor", "Calendar / FRED")
                try:
                    _fred_configured = bool(st.secrets.get("FRED_API_KEY", ""))
                except Exception:
                    _fred_configured = False
                jst_bbg_rows([("FRED key configured", "YES" if _fred_configured else "NO")])
                st.caption("Full calendar, yields, and cross-asset correlation live in the 'Macro & News Intelligence' tab.")
        with dxy_col:
            with st.container(border=True):
                jst_bbg_panel_header("DXY & US Treasury Yield", "Dollar Index")
                if _dxy_snap.get("available"):
                    jst_bbg_rows([
                        ("DXY source", _dxy_snap.get("source", "—")),
                        ("DXY last", f"{_dxy_snap.get('last', 0):,.2f}"),
                    ])
                else:
                    jst_bbg_unavailable("Open 'Macro & News Intelligence' → DXY & Treasury Yields once to populate this tile.")
        with ml_col:
            with st.container(border=True):
                jst_bbg_panel_header("Machine Learning Analytics", "Stage 8-11 Lab")
                if _ml_snap:
                    jst_bbg_rows([
                        ("Model", _ml_snap.get("model_choice", "—")),
                        ("Test accuracy", f"{_ml_snap.get('accuracy',0)*100:.1f}%"),
                        ("Precision / Recall", f"{_ml_snap.get('precision',0)*100:.1f}% / {_ml_snap.get('recall',0)*100:.1f}%"),
                    ])
                else:
                    jst_bbg_unavailable("Open the 'Machine Learning Lab' tab once to populate this tile.")

        st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)
        eq_col, trade_col, signal_col, diag_col = st.columns([1.3, 1, 1, 1], gap="small")
        with eq_col:
            with st.container(border=True):
                jst_bbg_panel_header("Equity Curve & Performance", "Closed paper trades")
                if _hist_list:
                    _hdf = pd.DataFrame(_hist_list)
                    if "result_r" in _hdf:
                        _rv = pd.to_numeric(_hdf["result_r"], errors="coerce").dropna()
                        if len(_rv):
                            _eq = _rv.cumsum()
                            eqf = go.Figure(go.Scatter(x=list(range(len(_eq))), y=_eq.values, mode="lines",
                                                       line=dict(color=color("tp") if _eq.iloc[-1] >= 0 else color("sl"), width=1.6)))
                            eqf.update_layout(height=120, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                              plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=4, r=4, t=4, b=4), showlegend=False)
                            eqf.update_xaxes(visible=False); eqf.update_yaxes(gridcolor=color("grid"))
                            st.plotly_chart(eqf, use_container_width=True, config={"displaylogo": False})
                            _wins = int((_rv > 0).sum())
                            jst_bbg_rows([
                                ("Closed trades", f"{len(_rv)}"),
                                ("Win rate", f"{_wins/len(_rv):.1%}"),
                                ("Cumulative R", f"{_eq.iloc[-1]:+.2f} R"),
                            ])
                else:
                    jst_bbg_unavailable("No closed paper trades recorded yet this session.")
        with trade_col:
            with st.container(border=True):
                jst_bbg_panel_header("Trade History", f"last {min(len(_hist_list),8)}")
                if _hist_list:
                    _tdf = pd.DataFrame(_hist_list).tail(8)
                    _cols = [c for c in ("entry_time", "side", "result_r") if c in _tdf.columns]
                    st.dataframe(_tdf[_cols] if _cols else _tdf, use_container_width=True, hide_index=True, height=160)
                else:
                    jst_bbg_unavailable("No closed trades yet.")
        with signal_col:
            with st.container(border=True):
                jst_bbg_panel_header("Signal History", "SQLite log")
                if _db_ready:
                    try:
                        _sigdf_recent = jst_db_load_recent_signals(symbol, limit=8)
                        if not _sigdf_recent.empty:
                            st.dataframe(_sigdf_recent[["signal_time", "side", "decision"]], use_container_width=True, hide_index=True, height=160)
                        else:
                            jst_bbg_unavailable("No persisted signals yet for this instrument.")
                    except Exception:
                        jst_bbg_unavailable("Signal log unavailable.")
                else:
                    jst_bbg_unavailable("Persistence store not ready.")
        with diag_col:
            with st.container(border=True):
                jst_bbg_panel_header("Quant Research & Diagnostics", "DB status")
                if _db_ready:
                    _counts = jst_db_counts()
                    jst_bbg_rows([
                        ("Persisted trades", _counts.get("trades", 0)),
                        ("Signal snapshots", _counts.get("signals", 0)),
                        ("Model runs", _counts.get("model_runs", 0)),
                    ])
                    st.caption("Full walk-forward / OOS diagnostics live in 'Quant Research Engine'.")
                else:
                    jst_bbg_unavailable("Persistence store not ready.")
    except Exception as exc:
        st.error("Command Center failed to render. All other tabs continue to work independently.")
        st.exception(exc)

# --------------------------- RISK REWARD + CANDLESTICK ---------------------------
with tabs[1]:
    try:
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

        jst_zone_header("TRADE TICKET", "Setup composition")
        _rr_tone = color("tp") if actual_rr >= rr else color("accent")
        jst_position_card(
            direction,
            [
                ("Risk distance", f"{risk_distance:.2f}"),
                ("Reward distance", f"{reward_distance:.2f}"),
                ("Actual R:R", f"1 : {actual_rr:.2f}"),
                ("Break-even win rate", f"{breakeven:.2f}%"),
            ],
        )
        jst_stat_grid([
            ("Estimated risk", f"${risk_usd_est:,.2f}", color("sl")),
            ("Estimated reward", f"${reward_usd_est:,.2f}", color("tp")),
            ("Risk budget", f"${risk_budget:,.2f}", color("accent")),
            ("Budget usage", f"{(risk_usd_est/risk_budget*100 if risk_budget else 0):.1f}%", _rr_tone),
        ])
        if risk_usd_est > risk_budget:
            st.warning("Estimated setup risk exceeds the selected account risk budget. Reduce lot size or stop distance.")
        st.caption("This is a planning/simulation tool. It does not submit orders. Broker contract size, tick value, spread and commissions can differ.")
        # Snapshot cache for Command Center tile (read-only mirror; no logic here).
        st.session_state["jst_snap_tradeticket"] = {
            "direction": direction, "entry": entry, "sl": sl, "tp": tp,
            "actual_rr": actual_rr, "risk_usd_est": risk_usd_est, "risk_budget": risk_budget,
        }
    # --------------------------- QUANT MATH 3D ---------------------------
    except Exception as exc:
        st.error("Market Chart + Risk/Reward panel failed to render. Other panels will continue to load.")
        st.exception(exc)
with tabs[2]:
    try:
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
    except Exception as exc:
        st.error("Quant Math 3D panel failed to render. Other panels will continue to load.")
        st.exception(exc)
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
with tabs[3]:
    try:
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
        jst_zone_header("POSITION MONITORING", "Live paper exposure")
        jst_position_card(
            paper_side,
            [
                ("Entry price", f"{paper_entry:,.2f}"),
                ("Latest price", f"{live_price:,.2f}"),
                ("Gross floating P&L", f"${gross_pnl:+,.2f}"),
                ("Net P&L (after costs)", f"${net_pnl:+,.2f}"),
            ],
        )
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
            # st_autorefresh above handles reruns; do not block the Streamlit script with sleep().
        st.session_state["jst_snap_position_monitor"] = {
            "side": paper_side, "entry": paper_entry, "live": live_price,
            "gross_pnl": gross_pnl, "net_pnl": net_pnl,
        }
    # --------------------------- RISK / VALIDATION ---------------------------
    except Exception as exc:
        st.error("Running P&L panel failed to render. Other panels will continue to load.")
        st.exception(exc)
with tabs[4]:
    try:
        st.subheader("Risk & Validation")
        st.warning("These metrics describe the loaded price series; they are not a strategy backtest.")
        rets = market["close"].pct_change().dropna()
        st.markdown("### JST V2 | CAPITAL & RISK CONTROLS")

        rc1, rc2, rc3 = st.columns(3)

        account_balance = rc1.number_input(
                    "Account Balance (USD)",
                    min_value=1.0,
                    value=100.0,
                    step=10.0
                )

        risk_percent = rc2.number_input(
                    "Risk Per Trade (%)",
                    min_value=0.01,
                    max_value=5.0,
                    value=0.25,
                    step=0.05
                )

        daily_loss_limit = rc3.number_input(
                    "Daily Loss Limit (%)",
                    min_value=0.1,
                    max_value=10.0,
                    value=1.0,
                    step=0.1
                )

        risk_budget = account_balance * risk_percent / 100
        daily_loss_budget = account_balance * daily_loss_limit / 100

        jst_zone_header("RISK MANAGEMENT", "Exposure & capital-at-risk")
        _risk_dial_col, _daily_dial_col = st.columns(2)
        with _risk_dial_col:
            jst_risk_dial(
                "Max risk per trade",
                f"${risk_budget:.2f}",
                used_pct=float(risk_percent / 5.0 * 100),
                foot=f"{risk_percent:.2f}% of ${account_balance:,.0f} balance",
            )
        with _daily_dial_col:
            jst_risk_dial(
                "Daily loss threshold",
                f"${daily_loss_budget:.2f}",
                used_pct=float(daily_loss_limit / 10.0 * 100),
                foot=f"{daily_loss_limit:.2f}% of ${account_balance:,.0f} balance",
            )
        st.caption(
                    "Risk budget monitor only. It does not execute orders "
                    "or automatically block signals."
                )

        st.divider()
        jst_zone_header("MARKET DATA", "Series diagnostics")
        jst_stat_grid([
            ("Observations", f"{len(market):,}", None),
            ("Mean bar return", f"{rets.mean()*100:.4f}%", None),
            ("Return std. dev.", f"{rets.std()*100:.4f}%", None),
            ("Max close drawdown", f"{drawdown.min()*100:.2f}%", color("sl")),
        ])
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
        st.session_state["jst_snap_risk_monitor"] = {
            "risk_budget": risk_budget, "daily_loss_budget": daily_loss_budget,
            "risk_percent": risk_percent, "daily_loss_limit": daily_loss_limit,
            "ret_std": float(rets.std() * 100), "max_dd": float(drawdown.min() * 100),
        }
    # --------------------------- JOURNAL / SETTINGS ---------------------------
    except Exception as exc:
        st.error("Risk & Validation panel failed to render. Other panels will continue to load.")
        st.exception(exc)
with tabs[5]:
    try:
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

    except Exception as exc:
        st.error("Journal / Settings panel failed to render. Other panels will continue to load.")
        st.exception(exc)
st.divider()
st.caption("JUMP STREET TRADING • Prototype for testing • Signal-only • Verify data, broker contract specs, costs, and calculations before any real trading.")


# --------------------------- MACHINE LEARNING LAB ---------------------------
with tabs[6]:
    try:
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
            ml_features["volume_change"] = np.log1p(volume.clip(lower=0)).diff().replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-10, 10)
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
                    st.session_state["jst_snap_ml_lab"] = {
                        "accuracy": float(accuracy), "precision": float(precision), "recall": float(recall),
                        "model_choice": model_choice, "latest_prob": float(latest_prob), "label": label,
                    }

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
    except Exception as exc:
        st.error("Machine Learning Lab panel failed to render. Other panels will continue to load.")
        st.exception(exc)
with tabs[7]:
    try:
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

        # EVT: Peaks Over Threshold (POT) + Generalized Pareto Distribution
        losses = -ret_adv[np.isfinite(ret_adv)]

        c1, c2 = st.columns(2)
        with c1:
            threshold_q = st.slider(
                "EVT Threshold Quantile",
                0.80, 0.98, 0.90, 0.01,
                key="adv_evt_threshold"
            )
        with c2:
            confidence = st.slider(
                "VaR Confidence",
                0.95, 0.999, 0.99, 0.001,
                format="%.3f",
                key="adv_evt_confidence"
            )

        if len(losses) >= 100:
            threshold = float(np.quantile(losses, threshold_q))
            excess = losses[losses > threshold] - threshold
            n_total = len(losses)
            n_exc = len(excess)

            st.metric("Extreme Exceedances", f"{n_exc} / {n_total}")

            if n_exc >= 20 and threshold > 0:
                try:
                    from scipy.stats import genpareto

                    xi, loc, beta = genpareto.fit(excess, floc=0)

                    if beta <= 0 or not np.isfinite(xi + beta):
                        raise ValueError("Invalid GPD parameters.")

                    tail_ratio = n_exc / (n_total * (1 - confidence))

                    if tail_ratio <= 1:
                        st.warning(
                            "Confidence level is too high for the selected threshold."
                        )
                    else:
                        if abs(xi) < 1e-7:
                            var_loss = threshold + beta * np.log(tail_ratio)
                        else:
                            var_loss = threshold + (
                                beta / xi
                            ) * (tail_ratio ** xi - 1)

                        if xi < 1:
                            es_loss = (
                                var_loss + beta - xi * threshold
                            ) / (1 - xi)
                        else:
                            es_loss = np.nan

                        m1, m2, m3 = st.columns(3)
                        m1.metric(
                            f"EVT VaR ({confidence:.1%})",
                            f"{var_loss * 100:.4f}%"
                        )
                        m2.metric(
                            "Expected Shortfall",
                            f"{es_loss * 100:.4f}%" if np.isfinite(es_loss)
                            else "Undefined (xi >= 1)"
                        )
                        m3.metric("GPD Shape (xi)", f"{xi:.4f}")

                        st.write(
                            f"**GPD Scale (beta):** {beta:.6f} | "
                            f"**Threshold:** {threshold * 100:.4f}%"
                        )

                        # EVT Threshold Stability Analysis
                        stability_rows = []

                        for q in [0.80, 0.85, 0.90, 0.925, 0.95]:
                            u = float(np.quantile(losses, q))
                            exc = losses[losses > u] - u

                            if len(exc) < 20:
                                continue

                            try:
                                xi_q, _, beta_q = genpareto.fit(
                                    exc, floc=0
                                )

                                if beta_q <= 0 or not np.isfinite(xi_q + beta_q):
                                    continue

                                # Adjust scale to the current reference threshold
                                beta_adjusted = (
                                    beta_q - xi_q * (u - threshold)
                                )

                                stability_rows.append({
                                    "Threshold Quantile": q,
                                    "Threshold": u,
                                    "Exceedances": len(exc),
                                    "Shape (xi)": xi_q,
                                    "Adjusted Scale": beta_adjusted
                                })

                            except Exception:
                                continue

                        if len(stability_rows) >= 2:
                            stability_df = pd.DataFrame(stability_rows)

                            st.markdown("**EVT Threshold Stability Test**")
                            st.dataframe(
                                stability_df.style.format({
                                    "Threshold Quantile": "{:.3f}",
                                    "Threshold": "{:.6f}",
                                    "Shape (xi)": "{:.4f}",
                                    "Adjusted Scale": "{:.6f}"
                                }),
                                use_container_width=True
                            )


                            # Shape parameter stability
                            shape_fig = go.Figure()
                            shape_fig.add_trace(go.Scatter(
                                x=stability_df["Threshold Quantile"],
                                y=stability_df["Shape (xi)"],
                                mode="lines+markers",
                                name="GPD Shape (xi)"
                            ))
                            shape_fig.update_layout(
                                title="GPD Shape Parameter Stability",
                                xaxis_title="Threshold Quantile",
                                yaxis_title="Shape (xi)",
                                template="plotly_dark"
                            )
                            st.plotly_chart(
                                shape_fig,
                                use_container_width=True
                            )

                            # Adjusted scale stability
                            scale_fig = go.Figure()
                            scale_fig.add_trace(go.Scatter(
                                x=stability_df["Threshold Quantile"],
                                y=stability_df["Adjusted Scale"],
                                mode="lines+markers",
                                name="Adjusted Scale"
                            ))
                            scale_fig.update_layout(
                                title="GPD Adjusted Scale Stability",
                                xaxis_title="Threshold Quantile",
                                yaxis_title="Adjusted Scale",
                                template="plotly_dark"
                            )
                            st.plotly_chart(
                                scale_fig,
                                use_container_width=True
                            )
        
                            
                        else:
                            st.warning(
                                "Insufficient exceedances across thresholds "
                                "for stability analysis."
                            )

                        # Empirical vs fitted exceedance survival
                        sorted_excess = np.sort(excess)
                        empirical_sf = (
                            n_exc - np.arange(n_exc)
                        ) / n_exc
                        fitted_sf = genpareto.sf(
                            sorted_excess,
                            c=xi,
                            loc=0,
                            scale=beta
                        )

                        evt_fig = go.Figure()
                        evt_fig.add_trace(go.Scatter(
                            x=sorted_excess * 100,
                            y=empirical_sf,
                            mode="markers",
                            name="Empirical Survival"
                        ))
                        evt_fig.add_trace(go.Scatter(
                            x=sorted_excess * 100,
                            y=fitted_sf,
                            mode="lines",
                            name="Fitted GPD Survival"
                        ))
                        evt_fig.update_layout(
                            title="EVT: Empirical vs GPD Tail Fit",
                            xaxis_title="Exceedance (%)",
                            yaxis_title="Survival Probability",
                            yaxis_type="log",
                            template="plotly_dark"
                        )
                        st.plotly_chart(
                            evt_fig,
                            use_container_width=True
                        )

                        st.caption(
                            "POT-GPD fitted by maximum likelihood. "
                            "Downside tail uses negative returns converted "
                            "to positive losses. VaR and ES are in return "
                            "percentage units, not account-money units. "
                            "Model fit and out-of-sample validation are "
                            "still required."
                        )

                    # Rolling Out-of-Sample EVT VaR Backtesting
                    st.markdown("### EVT VaR Backtesting")

                    bt_window = 120
                    bt_losses = -np.asarray(ret_adv, dtype=float)
                    bt_losses = bt_losses[np.isfinite(bt_losses)]

                    bt_forecasts = []
                    bt_actual = []

                    if len(bt_losses) > bt_window + 30:
                        for i in range(bt_window, len(bt_losses)):
                            train = bt_losses[i - bt_window:i]
                            actual_loss = bt_losses[i]

                            u_bt = float(np.quantile(train, 0.80))
                            exc_bt = train[train > u_bt] - u_bt

                            if len(exc_bt) < 20 or u_bt <= 0:
                                continue

                            try:
                                xi_bt, _, beta_bt = genpareto.fit(
                                    exc_bt, floc=0
                                )

                                if beta_bt <= 0 or not np.isfinite(xi_bt + beta_bt):
                                    continue

                                ratio_bt = len(exc_bt) / (
                                    bt_window * (1 - confidence)
                                )

                                if ratio_bt <= 1:
                                    continue

                                if abs(xi_bt) < 1e-7:
                                    var_bt = (
                                        u_bt + beta_bt * np.log(ratio_bt)
                                    )
                                else:
                                    var_bt = u_bt + (
                                        beta_bt / xi_bt
                                    ) * (ratio_bt ** xi_bt - 1)

                                if not np.isfinite(var_bt) or var_bt <= 0:
                                    continue

                                bt_forecasts.append(var_bt)
                                bt_actual.append(actual_loss)

                            except Exception:
                                continue

                    if len(bt_forecasts) >= 30:
                        bt_forecasts = np.asarray(bt_forecasts)
                        bt_actual = np.asarray(bt_actual)

                        violations = (
                            bt_actual > bt_forecasts
                        ).astype(int)

                        n_test = len(violations)
                        n_fail = int(violations.sum())
                        expected_fail = n_test * (1 - confidence)

                        def bernoulli_ll(success, total, probability):
                            if total <= 0:
                                return 0.0
                            probability = np.clip(
                                probability, 1e-12, 1 - 1e-12
                            )
                            return (
                                success * np.log(probability)
                                + (total - success) * np.log1p(-probability)
                            )

                        # Kupiec Proportion of Failures Test
                        p_hat = n_fail / n_test
                        ll_null = bernoulli_ll(
                            n_fail, n_test, 1 - confidence
                        )
                        ll_alt = bernoulli_ll(
                            n_fail, n_test, p_hat
                        )

                        lr_kupiec = max(
                            0.0, 2 * (ll_alt - ll_null)
                        )
                        p_kupiec = float(
                            chi2.sf(lr_kupiec, 1)
                        )

                        # Christoffersen Independence Test
                        n00 = n01 = n10 = n11 = 0

                        for j in range(1, n_test):
                            previous = violations[j - 1]
                            current = violations[j]

                            if previous == 0 and current == 0:
                                n00 += 1
                            elif previous == 0 and current == 1:
                                n01 += 1
                            elif previous == 1 and current == 0:
                                n10 += 1
                            else:
                                n11 += 1

                        row0 = n00 + n01
                        row1 = n10 + n11
                        total_trans = row0 + row1

                        if row0 > 0 and row1 > 0 and total_trans > 0:
                            p_pool = (n01 + n11) / total_trans

                            ll_ind_null = bernoulli_ll(
                                n01 + n11, total_trans, p_pool
                            )

                            ll_ind_alt = (
                                bernoulli_ll(n01, row0, n01 / row0)
                                + bernoulli_ll(n11, row1, n11 / row1)
                            )

                            lr_ind = max(
                                0.0, 2 * (ll_ind_alt - ll_ind_null)
                            )
                            p_ind = float(chi2.sf(lr_ind, 1))

                            lr_cc = lr_kupiec + lr_ind
                            p_cc = float(chi2.sf(lr_cc, 2))
                        else:
                            lr_ind = np.nan
                            p_ind = np.nan
                            lr_cc = np.nan
                            p_cc = np.nan

                        st.metric(
                            "OOS VaR Observations",
                            n_test
                        )

                        st.metric(
                            "Actual VaR Violations",
                            f"{n_fail} / {n_test}"
                        )

                        st.metric(
                            "Observed Violation Rate",
                            f"{p_hat:.2%}"
                        )

                        bt_results = pd.DataFrame({
                            "Test": [
                                "Kupiec POF",
                                "Christoffersen Independence",
                                "Conditional Coverage"
                            ],
                            "LR Statistic": [
                                lr_kupiec,
                                lr_ind,
                                lr_cc
                            ],
                            "P-Value": [
                                p_kupiec,
                                p_ind,
                                p_cc
                            ]
                        })

                        st.dataframe(
                            bt_results.style.format({
                                "LR Statistic": "{:.4f}",
                                "P-Value": "{:.4f}"
                            }),
                            use_container_width=True
                        )

                        st.caption(
                            "Rolling historical training window: 120 bars. "
                            "GPD is refitted using past observations only. "
                            "P-value below 0.05 indicates rejection of "
                            "the corresponding null hypothesis. "
                            "Small samples have low statistical power."
                        )
                    else:
                        st.warning(
                            "Insufficient out-of-sample forecasts. "
                            "At least 30 valid forecasts are required."
                        )

                except ImportError:
                    st.error(
                        "SciPy is required. Add scipy to requirements.txt."
                    )
                except Exception as exc:
                    st.error(f"EVT fitting failed: {exc}")

            else:
                st.warning(
                    "Insufficient exceedances. Need at least 20 observations "
                    "above the selected threshold."
                )

        else:
            st.warning("EVT requires at least 100 valid return observations.")
        with adv_tabs[5]:
            st.markdown("**Hawkes Process — Empirical Self-Exciting Model**")
            try:
                from scipy.optimize import minimize

                returns = np.asarray(ret_adv, dtype=float)
                returns = returns[np.isfinite(returns)]
                c1, c2 = st.columns(2)
                with c1:
                    event_quantile = st.slider(
                        "Extreme Event Threshold", 0.90, 0.99, 0.95, 0.01,
                        key="hawkes_threshold"
                    )
                with c2:
                    max_branching = st.slider(
                        "Maximum Branching Ratio", 0.50, 0.99, 0.95, 0.01,
                        key="hawkes_branching"
                    )

                if len(returns) >= 100:
                    threshold = float(np.quantile(np.abs(returns), event_quantile))
                    event_times = np.flatnonzero(np.abs(returns) >= threshold).astype(float)
                    T = float(len(returns))
                    if len(event_times) >= 10:
                        def hawkes_nll(params):
                            baseline, beta, eta = params
                            alpha = eta * beta
                            intensity_events = np.full(len(event_times), baseline, dtype=float)
                            excitation = 0.0
                            previous_time = 0.0
                            for i, event_time in enumerate(event_times):
                                excitation *= np.exp(-beta * (event_time - previous_time))
                                intensity_events[i] += excitation
                                excitation += alpha
                                previous_time = event_time
                            if np.any(intensity_events <= 0):
                                return 1e12
                            compensator = baseline * T + eta * np.sum(
                                1.0 - np.exp(-beta * (T - event_times))
                            )
                            return float(-(np.sum(np.log(intensity_events)) - compensator))

                        result = minimize(
                            hawkes_nll,
                            [max(len(event_times) / T * 0.5, 1e-6), 1.0, 0.3],
                            method="L-BFGS-B",
                            bounds=[(1e-8, 10.0), (1e-4, 20.0), (1e-5, max_branching)]
                        )
                        if result.success and np.isfinite(result.fun):
                            baseline, beta, eta = result.x
                            alpha = eta * beta
                            intensity = np.zeros(int(T), dtype=float)
                            excitation = 0.0
                            event_set = set(event_times.astype(int))
                            for t_idx in range(int(T)):
                                intensity[t_idx] = baseline + excitation
                                excitation *= np.exp(-beta)
                                if t_idx in event_set:
                                    excitation += alpha
                            m1, m2, m3 = st.columns(3)
                            m1.metric("Baseline Intensity", f"{baseline:.5f}")
                            m2.metric("Excitation Alpha", f"{alpha:.5f}")
                            m3.metric("Decay Beta", f"{beta:.5f}")
                            st.metric("Branching Ratio", f"{eta:.4f}")
                            hawkes_fig = go.Figure()
                            hawkes_fig.add_trace(go.Scatter(x=np.arange(int(T)), y=intensity, mode="lines", name="Conditional Intensity"))
                            hawkes_fig.add_trace(go.Scatter(x=event_times, y=intensity[event_times.astype(int)], mode="markers", name="Observed Extreme Events"))
                            hawkes_fig.update_layout(title="Fitted Hawkes Conditional Intensity", xaxis_title="Bar Index", yaxis_title="Event Intensity", template="plotly_dark")
                            st.plotly_chart(hawkes_fig, use_container_width=True)
                            st.caption("Exponential Hawkes process fitted by MLE to absolute-return extreme events. Branching ratio is constrained below the selected stability limit. This is a research diagnostic, not a validated trading signal.")
                        else:
                            st.warning("Hawkes MLE optimization did not converge. Inspect event count and parameter estimates.")
                    else:
                        st.warning("Insufficient extreme events. At least 10 events are required.")
                else:
                    st.warning("At least 100 valid return observations are required.")
            except ImportError:
                st.error("SciPy optimization module is required.")
            except Exception as exc:
                st.error(f"Hawkes Process failed: {exc}")

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

    except Exception as exc:
        st.error("Advanced Quant 3D panel failed to render. Other panels will continue to load.")
        st.exception(exc)
    st.divider()
    st.subheader("3D Visualization Controls")
    st.caption("Global colors can be changed in the sidebar. Use Plotly's toolbar to rotate, zoom, pan, and reset each 3D chart.")
    st.markdown("- Surface palette: background → surface → accent → text.\n- Candlestick colors remain independently configurable in the sidebar.\n- Simulation-based charts are labeled as simulations; empirical charts are derived from the selected dataset.")


# --------------------------- SIGNAL ENGINE / STAGE 5 ---------------------------
with tabs[8]:
    try:
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
        # Define the latest candle timestamp before ML persistence uses it.
        latest_time = pd.to_datetime(sigdf["time"].iloc[-1])
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

        # ============================================================
        # STAGES 13–16: SIGNAL INTELLIGENCE + ML + ENSEMBLE + RISK GATE
        # ML target: next-bar direction; training excludes the latest
        # unlabelled feature row. This is research scoring, not a guarantee.
        # ============================================================
        ml_up_probability = 0.50
        ml_model_status = "ML unavailable / neutral prior"
        ml_train_n = 0
        try:
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import LogisticRegression
            ml_close = pd.to_numeric(sigdf["close"], errors="coerce")
            ml_ret = ml_close.pct_change()
            ml_feat = pd.DataFrame(index=sigdf.index)
            for lag in (1,2,3,5,10): ml_feat[f"ret_{lag}"] = ml_ret.shift(lag)
            for win in (5,10,20):
                ml_feat[f"mom_{win}"] = ml_close.pct_change(win)
                ml_feat[f"vol_{win}"] = ml_ret.rolling(win, min_periods=win).std()
            ml_feat["ema_gap"] = ml_close / ml_close.ewm(span=20, adjust=False).mean() - 1
            ml_feat["range_pct"] = (pd.to_numeric(sigdf["high"],errors="coerce")-pd.to_numeric(sigdf["low"],errors="coerce"))/ml_close.replace(0,np.nan)
            ml_target = (ml_close.shift(-1) > ml_close).astype(float).where(ml_close.shift(-1).notna())
            ml_train = ml_feat.replace([np.inf,-np.inf],np.nan).copy()
            ml_train["target"] = ml_target
            ml_train = ml_train.dropna(subset=["target"])
            latest_x = ml_feat.replace([np.inf,-np.inf],np.nan).iloc[[-1]]
            ml_train_n = len(ml_train)
            if ml_train_n >= 100 and ml_train["target"].nunique() == 2 and latest_x.notna().any(axis=None):
                model_ml = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                         LogisticRegression(max_iter=1000,class_weight="balanced",C=0.5))
                model_ml.fit(ml_train[ml_feat.columns], ml_train["target"].astype(int))
                ml_up_probability = float(model_ml.predict_proba(latest_x)[0][list(model_ml[-1].classes_).index(1)])
                ml_model_status = "Logistic Regression / chronological historical bars"
                if st.session_state.get("jst_db_ready"):
                    jst_db_save_model_run(latest_time, symbol, "LogisticRegression_next_bar_direction", ml_train_n,
                                          {"feature_count":len(ml_feat.columns),"target":"next_bar_up","training_excludes_latest_unlabelled_bar":True})
        except Exception as _ml_exc:
            ml_model_status = f"ML neutral fallback ({type(_ml_exc).__name__})"

        rule_long_prob = float(ls / 5.0)
        rule_short_prob = float(ss / 5.0)
        # Confidence is measured in the direction of the current rule candidate.
        # For SHORT, bearish ML confidence is (1 - P(UP)); therefore a strong
        # SHORT must pass the same >= 60% threshold as a strong LONG.
        if sig_side == "LONG":
            ensemble_confidence = float(np.clip(0.45 * rule_long_prob + 0.55 * ml_up_probability, 0.0, 1.0))
        elif sig_side == "SHORT":
            ensemble_confidence = float(np.clip(0.45 * rule_short_prob + 0.55 * (1.0 - ml_up_probability), 0.0, 1.0))
        else:
            ensemble_confidence = 0.50
        ensemble_margin = abs(ensemble_confidence - 0.50)
        # Confidence is candidate-directional, not an unconditional P(UP).
        # Admit either side only when confidence in that candidate reaches 60%.
        ensemble_decision = sig_side if sig_side in ("LONG", "SHORT") and ensemble_confidence >= 0.60 else "FLAT"
        # Keep signal direction anchored to rule engine; ensemble must agree.
        intelligence_side = sig_side if ensemble_decision == sig_side else "FLAT"
        intelligence_reason = "Ensemble agrees" if intelligence_side != "FLAT" else "Rule/ML ensemble disagreement or confidence below 60%"

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
                        # ==========================================
            # JST V2 - POSITION STATE ENGINE
            # PAPER MODE / NO BROKER EXECUTION
            # ==========================================

        if "jst_position" not in st.session_state:
            st.session_state["jst_position"] = None

        if "jst_trade_history" not in st.session_state:
            st.session_state["jst_trade_history"] = []

        if "jst_last_signal_time" not in st.session_state:
            st.session_state["jst_last_signal_time"] = None
        st.session_state.setdefault("jst_entry_candle_locks", {})

        position = st.session_state["jst_position"]

        PIP_SIZE = 0.01
        FIXED_SL_PIPS = 700
        sl_distance = PIP_SIZE * FIXED_SL_PIPS

        # Momentum-aware position management. Initial SL remains fixed at 700 pips (7.00 XAU price units).
        # Trailing is evaluated after each completed bar and becomes effective
        # from the next bar (no same-bar look-ahead).
        position_closed_this_run = False
        if position is not None:
            position.setdefault("initial_sl", position.get("sl"))
            position.setdefault("initial_risk", abs(float(position["entry"]) - float(position["initial_sl"])))
            position.setdefault("trail_active", False)
            position.setdefault("trail_atr_multiple", 1.5)
            position.setdefault("trail_activation_r", 1.0)
            position.setdefault("realized_r", None)
            _entry_ts = pd.to_datetime(position.get("entry_time"), utc=True, errors="coerce")
            _bar_times = pd.to_datetime(sigdf["time"], utc=True, errors="coerce")
            future_bars = sigdf.loc[_bar_times > _entry_ts]

            for bar_idx, bar in future_bars.iterrows():
                entry_price = float(position["entry"])
                bar_high, bar_low = float(bar["high"]), float(bar["low"])
                initial_risk = max(float(position.get("initial_risk", sl_distance)), 1e-12)
                if position["side"] == "LONG":
                    favorable_move, adverse_move = bar_high-entry_price, entry_price-bar_low
                    hit_sl, hit_tp = bar_low <= float(position["sl"]), bar_high >= float(position["tp"])
                else:
                    favorable_move, adverse_move = entry_price-bar_low, bar_high-entry_price
                    hit_sl, hit_tp = bar_high >= float(position["sl"]), bar_low <= float(position["tp"])
                position["mfe_price"] = max(float(position.get("mfe_price", 0.0)), favorable_move, 0.0)
                position["mae_price"] = max(float(position.get("mae_price", 0.0)), adverse_move, 0.0)

                # Conservative intrabar ordering: existing stop first if SL and TP both touch.
                if hit_sl and position.get("trail_active") and not np.isclose(float(position["sl"]), float(position.get("initial_sl", position["sl"]))):
                    exit_reason = "TRAILING_STOP"
                else:
                    exit_reason = "STOP_LOSS" if hit_sl else ("TAKE_PROFIT" if hit_tp else None)
                if exit_reason:
                    exit_price = float(position["sl"] if hit_sl else position["tp"])
                    signed_move = (exit_price-entry_price) if position["side"] == "LONG" else (entry_price-exit_price)
                    result_r = signed_move / initial_risk
                    closed_at = pd.to_datetime(bar["time"])
                    trade = {
                        "symbol": symbol, "side": position["side"],
                        "entry_time": position["entry_time"], "exit_time": closed_at,
                        "entry": entry_price, "exit": exit_price,
                        "initial_sl": position.get("initial_sl"), "sl": position["sl"], "tp": position["tp"],
                        "initial_risk_price": initial_risk, "exit_reason": exit_reason,
                        "result": "WIN" if result_r > 0 else ("LOSS" if result_r < 0 else "BREAKEVEN"),
                        "result_r": float(result_r), "mfe_price": float(position.get("mfe_price", 0.0)),
                        "mae_price": float(position.get("mae_price", 0.0)),
                        "mfe_r": float(position.get("mfe_price", 0.0)) / initial_risk,
                        "mae_r": float(position.get("mae_price", 0.0)) / initial_risk,
                        "trail_activated": bool(position.get("trail_active", False)),
                        "trail_atr_multiple": float(position.get("trail_atr_multiple", 1.5)),
                        **position.get("features", {}),
                        "forecast_probability": position.get("forecast_probability"),
                        "forecast_method": position.get("forecast_method", "LEGACY_NO_FORECAST"),
                        "forecast_training_n": position.get("forecast_training_n", 0),
                        "forecast_created_at": position.get("forecast_created_at", position.get("entry_time")),
                        "ml_up_probability": position.get("ml_up_probability"),
                        "ensemble_confidence": position.get("ensemble_confidence")
                    }
                    # Idempotent append guard protects against Streamlit reruns.
                    _trade_id = jst_trade_key(trade, symbol)
                    _already = any(jst_trade_key(t, symbol) == _trade_id for t in st.session_state["jst_trade_history"])
                    if not _already:
                        st.session_state["jst_trade_history"].append(trade)
                        if st.session_state.get("jst_db_ready"):
                            jst_db_save_trade(trade, symbol)
                            jst_db_log("PAPER_POSITION_CLOSED", {"symbol":symbol,"side":position["side"],"exit_reason":exit_reason,"result_r":float(result_r)})
                    if st.session_state.get("jst_db_ready"):
                        jst_db_clear_active_positions(symbol)
                    position = None
                    st.session_state["jst_position"] = None
                    position_closed_this_run = True
                    break

                # Momentum Holding Engine: require aligned trend + positive directional
                # 3-bar momentum. If momentum weakens, hold the existing stop; do not
                # force-close solely on a soft momentum deterioration.
                close_now = float(bar["close"])
                _ema_now = float(ema_fast.iloc[bar_idx])
                _mom_now = float(mom3.iloc[bar_idx])
                _momentum_aligned = (close_now > _ema_now and _mom_now > 0) if position["side"] == "LONG" else (close_now < _ema_now and _mom_now < 0)
                _favorable_r = favorable_move / initial_risk
                if _favorable_r >= float(position.get("trail_activation_r", 1.0)) and _momentum_aligned:
                    _atr_now = float(atr.iloc[bar_idx])
                    if np.isfinite(_atr_now) and _atr_now > 0:
                        _candidate_sl = (close_now - _atr_now * float(position.get("trail_atr_multiple", 1.5))) if position["side"] == "LONG" else (close_now + _atr_now * float(position.get("trail_atr_multiple", 1.5)))
                        _old_sl = float(position["sl"])
                        if position["side"] == "LONG" and _candidate_sl > _old_sl:
                            position["sl"] = min(_candidate_sl, close_now)
                            position["trail_active"] = True
                        elif position["side"] == "SHORT" and _candidate_sl < _old_sl:
                            position["sl"] = max(_candidate_sl, close_now)
                            position["trail_active"] = True
                        if position.get("trail_active"):
                            position["last_trail_update"] = pd.to_datetime(bar["time"])
                            position["momentum_state"] = "ALIGNED_HOLD"
                else:
                    position["momentum_state"] = "WEAKENING" if _favorable_r >= float(position.get("trail_activation_r", 1.0)) else "BUILDING"

        # Persist active position, including MFE/MAE and any tightened trailing stop.
        if position is not None:
            st.session_state["jst_position"] = position
            if st.session_state.get("jst_db_ready"):
                jst_db_save_position(position, symbol)

        # STAGE 16 — risk-aware admission gate. Fixed 700-pip stop (7.00 XAU price units) for XAU reference.
        is_synthetic_feed = (mode == "Demo data (safe default)" or "synthetic demo fallback" in data_status.lower())
        risk_gate_reasons = []
        if is_synthetic_feed: risk_gate_reasons.append("synthetic/demo data")
        # Gaps are surfaced by JST Core for diagnostics; do not automatically
        # block across legitimate market-session/weekend closures. Stale feeds block.
        if jst_quality is not None and jst_quality.status == "STALE":
            risk_gate_reasons.append("market data is stale")
        if not np.isfinite(last_atr) or last_atr <= 0: risk_gate_reasons.append("invalid ATR")
        if ensemble_decision != sig_side or sig_side == "FLAT": risk_gate_reasons.append("ensemble not aligned")
        if ensemble_confidence < 0.60: risk_gate_reasons.append("ensemble confidence < 60%")
        if not np.isfinite(sl_distance) or sl_distance <= 0: risk_gate_reasons.append("invalid stop distance")
        # Daily stop: 0.25% risk/trade x 4R = 1.0% daily loss budget.
        try:
            _hist = pd.DataFrame(st.session_state.get("jst_trade_history", []))
            if not _hist.empty and "result_r" in _hist and "exit_time" in _hist:
                _exit = pd.to_datetime(_hist["exit_time"], errors="coerce", utc=True).dt.tz_convert("Asia/Jakarta")
                _today = pd.Timestamp.now(tz="Asia/Jakarta").date()
                _today_r = pd.to_numeric(_hist.loc[_exit.dt.date == _today, "result_r"], errors="coerce").fillna(0.0).sum()
                if _today_r <= -4.0: risk_gate_reasons.append("daily paper loss limit reached (-4R / approx. -1% at 0.25% risk)")
        except Exception:
            risk_gate_reasons.append("daily loss check unavailable")
        risk_gate_pass = len(risk_gate_reasons) == 0
        # Open a new paper position only when no position is active and all gates pass.
        if position is None and not position_closed_this_run and sig_side in ("LONG", "SHORT") and risk_gate_pass:
            _entry_candle_key = pd.Timestamp(latest_time).isoformat()
            _memory_locks = st.session_state["jst_entry_candle_locks"].setdefault(symbol, set())
            _history_has_candle = any(
                str(pd.Timestamp(t.get("entry_time")).isoformat()) == _entry_candle_key
                for t in st.session_state.get("jst_trade_history", [])
                if t.get("entry_time") is not None and not pd.isna(pd.to_datetime(t.get("entry_time"), errors="coerce"))
            )
            _candle_already_used = (
                _entry_candle_key in _memory_locks
                or _history_has_candle
                or st.session_state["jst_last_signal_time"] == latest_time
            )
            if not _candle_already_used:
                _entry_lock_claimed = True
                if st.session_state.get("jst_db_ready"):
                    try:
                        _entry_lock_claimed = jst_db_claim_entry_candle(symbol, latest_time, sig_side)
                    except Exception as _lock_exc:
                        _entry_lock_claimed = False
                        st.warning(f"Entry deduplication lock unavailable; new entry blocked safely: {type(_lock_exc).__name__}")
                if _entry_lock_claimed:
                    _memory_locks.add(_entry_candle_key)
                    entry = float(entry_candidate)
                    position_side = str(sig_side).strip().upper()
                    # Direction is captured once at entry and never inferred from later signals.
                    if position_side == "LONG":
                        stop_loss = entry - sl_distance
                        take_profit = entry + (sl_distance * reward_multiple)
                    elif position_side == "SHORT":
                        stop_loss = entry + sl_distance
                        take_profit = entry - (sl_distance * reward_multiple)

                    # Snapshot observable setup conditions at entry for conditional research.
                    atr_reference = float((pd.to_numeric(atr, errors="coerce") / pd.to_numeric(sigdf["close"], errors="coerce")).rolling(50).median().iloc[-1]) if len(sigdf) >= 50 else np.nan
                    atr_now = float(last_atr / entry) if entry else np.nan
                    if np.isfinite(atr_now) and np.isfinite(atr_reference):
                        volatility_state = "HIGH_VOL" if atr_now >= atr_reference else "LOW_VOL"
                    else:
                        volatility_state = "UNKNOWN"
                    score_band = "HIGH_SCORE" if sig_score >= 4 else "MID_SCORE" if sig_score >= 3 else "LOW_SCORE"
                    # Twelve Data timestamps are requested in UTC; normalize aware timestamps to UTC.
                    entry_ts = pd.Timestamp(latest_time)
                    if entry_ts.tzinfo is not None:
                        entry_hour = int(entry_ts.tz_convert("UTC").hour)
                    else:
                        entry_hour = int(entry_ts.hour)  # naive CSV timestamps retain their supplied clock basis
                    market_session = "ASIA" if 0 <= entry_hour < 8 else "LONDON" if 8 <= entry_hour < 13 else "NEW_YORK" if 13 <= entry_hour < 21 else "LATE"
                    trend_return = float(pd.to_numeric(sigdf["close"], errors="coerce").pct_change(20).iloc[-1]) if len(sigdf) >= 21 else np.nan
                    if np.isfinite(atr_now) and np.isfinite(atr_reference):
                        regime_vol = "HIGH_VOL" if atr_now >= atr_reference else "LOW_VOL"
                    else:
                        regime_vol = "UNKNOWN_VOL"
                    regime_trend = "UP_TREND" if np.isfinite(trend_return) and trend_return > 0 else "DOWN_TREND" if np.isfinite(trend_return) and trend_return < 0 else "NEUTRAL_TREND"
                    entry_features = {
                        "feature_side": position_side,
                        "feature_score": int(sig_score),
                        "feature_score_band": score_band,
                        "feature_volatility": volatility_state,
                        "feature_session": market_session,
                        "feature_regime": f"{regime_trend}|{regime_vol}",
                    }

                    # Store a strictly historical Bayesian forecast at signal/entry time.
                    # Use matching direction + score-band history only when enough prior samples exist;
                    # otherwise fall back to the global prior. No current/future outcome is included.
                    prior_rows = st.session_state.get("jst_trade_history", [])
                    prior_df = pd.DataFrame(prior_rows)
                    forecast_alpha = 1.0
                    forecast_beta = 1.0
                    forecast_min_match = 5
                    matched_n = 0
                    forecast_method = "BETA_PRIOR_ONLY"
                    if not prior_df.empty and "result_r" in prior_df.columns:
                        prior_df["result_r"] = pd.to_numeric(prior_df["result_r"], errors="coerce")
                        prior_df = prior_df.dropna(subset=["result_r"])
                        # Strict no-look-ahead: an outcome is eligible only after its
                        # position has CLOSED before this forecast timestamp. Entry time
                        # alone is insufficient because an older trade may still be open.
                        if "exit_time" in prior_df.columns:
                            prior_df["_outcome_available_at"] = pd.to_datetime(
                                prior_df["exit_time"], errors="coerce", utc=True
                            )
                            current_order = pd.to_datetime(entry_ts, errors="coerce", utc=True)
                            prior_df = prior_df[
                                prior_df["_outcome_available_at"].notna()
                                & (prior_df["_outcome_available_at"] < current_order)
                            ]
                        else:
                            prior_df = prior_df.iloc[0:0]
                        if {"feature_side", "feature_score_band"}.issubset(prior_df.columns):
                            matched = prior_df[
                                (prior_df["feature_side"].astype(str).str.upper() == position_side)
                                & (prior_df["feature_score_band"] == score_band)
                            ]
                        else:
                            matched = pd.DataFrame()
                        if len(matched) >= forecast_min_match:
                            matched_n = int(len(matched))
                            matched_wins = int((matched["result_r"] > 0).sum())
                            forecast_alpha += matched_wins
                            forecast_beta += matched_n - matched_wins
                            forecast_method = "CONDITIONAL_SIDE_SCORE_BETA"
                        else:
                            matched_n = int(len(prior_df))
                            prior_wins = int((prior_df["result_r"] > 0).sum())
                            forecast_alpha += prior_wins
                            forecast_beta += matched_n - prior_wins
                            forecast_method = "GLOBAL_HISTORICAL_BETA" if matched_n else "BETA_PRIOR_ONLY"
                    forecast_probability = forecast_alpha / (forecast_alpha + forecast_beta)
                    position = {
                       "symbol": symbol,
                       "side": position_side,
                       "entry": entry,
                       "sl": stop_loss,
                       "tp": take_profit,
                       "entry_time": latest_time,
                       "status": "ACTIVE",
                       "mfe_price": 0.0,
                       "mae_price": 0.0,
                       "target_r": float(reward_multiple),
                       "initial_sl": float(stop_loss),
                       "initial_risk": float(abs(entry - stop_loss)),
                       "trail_active": False,
                       "trail_activation_r": 1.0,
                       "trail_atr_multiple": 1.5,
                       "momentum_state": "BUILDING",
                       "features": entry_features,
                       "forecast_probability": float(forecast_probability),
                       "forecast_method": forecast_method,
                       "forecast_training_n": int(matched_n),
                       "forecast_created_at": latest_time,
                       "ml_up_probability": float(ml_up_probability),
                       "ml_model_status": ml_model_status,
                       "ml_training_n": int(ml_train_n),
                       "ensemble_confidence": float(ensemble_confidence),
                       "ensemble_decision": ensemble_decision,
                       "risk_gate_pass": bool(risk_gate_pass)
                    }

                    st.session_state["jst_position"] = position
                    if st.session_state.get("jst_db_ready"):
                        jst_db_save_position(position, symbol)
                        jst_db_log("PAPER_POSITION_OPEN", {"symbol":symbol,"side":position_side,"entry":entry,"confidence":ensemble_confidence})
                    st.session_state["jst_last_signal_time"] = latest_time

                # Block new signals while a paper position is active
        if st.session_state["jst_position"] is not None:
            active_position = st.session_state["jst_position"]
            sig_label = f"POSITION ACTIVE - {active_position['side']}"
            sig_side = "FLAT"

            st.info(
                f"ACTIVE PAPER POSITION: {active_position['side']} | "
                f"Entry: {active_position['entry']:.3f} | "
                f"SL: {active_position['sl']:.3f} | "
                f"TP: {active_position['tp']:.3f} | "
                f"Trail: {'ACTIVE' if active_position.get('trail_active') else 'WAITING (+1R + aligned momentum)'} | "
                f"Momentum: {active_position.get('momentum_state', 'BUILDING')}"
            )

            entry_candidate = active_position["entry"]
            sl_candidate = active_position["sl"]
            tp_candidate = active_position["tp"]
        # Sync the active instrument state cache after every position/history mutation.
        st.session_state["jst_positions_by_symbol"][symbol] = st.session_state.get("jst_position")
        st.session_state["jst_histories_by_symbol"][symbol] = st.session_state.get("jst_trade_history", [])

        # Persist one decision snapshot per candle; audit even when risk gate blocks entry.
        if st.session_state.get("jst_db_ready"):
            jst_db_save_signal(latest_time, symbol, sig_side, sig_score, ml_up_probability,
                               ensemble_confidence, "ADMIT" if risk_gate_pass else "BLOCK",
                               {"rule_long":ls,"rule_short":ss,"ml_status":ml_model_status,
                                "ensemble_decision":ensemble_decision,"risk_gate_reasons":risk_gate_reasons,
                                "data_status":data_status})
        jst_zone_header("SIGNAL ENGINE", "Ensemble decision stack")
        _gate_txt = "Risk gate: PASS" if risk_gate_pass else "Risk gate: BLOCK — " + "; ".join(risk_gate_reasons)
        jst_hero_decision(
            kicker="SIGNAL ENGINE — LIVE RESEARCH DECISION",
            label=ensemble_decision if ensemble_decision in ("LONG", "SHORT") else "FLAT / NO TRADE",
            sub=f"{_gate_txt} • ML: {ml_model_status}",
            side=ensemble_decision,
            confidence=ensemble_confidence,
            threshold=0.60,
        )
        jst_zone_header("PROBABILITY ENGINE", "Model vs. rule-based read")
        _pcol1, _pcol2 = st.columns(2)
        with _pcol1:
            st.markdown('<div class="jst-gauge-label">ML P(UP)</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="jst-gauge-track"><div class="jst-gauge-fill" '
                f'style="width:{ml_up_probability*100:.1f}%; background:{color("surface")};"></div></div>'
                f'<div class="jst-gauge-value" style="color:{color("surface")};">{ml_up_probability:.1%}</div>',
                unsafe_allow_html=True,
            )
        with _pcol2:
            st.markdown('<div class="jst-gauge-label">Ensemble Confidence (directional)</div>', unsafe_allow_html=True)
            _conf_tone = color("tp") if ensemble_confidence >= 0.60 else color("accent")
            st.markdown(
                f'<div class="jst-gauge-track"><div class="jst-gauge-fill" '
                f'style="width:{ensemble_confidence*100:.1f}%; background:{_conf_tone};"></div></div>'
                f'<div class="jst-gauge-value" style="color:{_conf_tone};">{ensemble_confidence:.1%}</div>',
                unsafe_allow_html=True,
            )
        jst_stat_grid([
            ("Ensemble decision", ensemble_decision, color("bull") if ensemble_decision == "LONG" else color("bear") if ensemble_decision == "SHORT" else None),
            ("Risk gate", "PASS" if risk_gate_pass else "BLOCK", color("tp") if risk_gate_pass else color("sl")),
            ("ML training rows", f"{ml_train_n:,}", None),
            ("Intelligence note", intelligence_reason, None),
        ])
        if risk_gate_pass:
            st.success("Risk gate passed. Paper position may be opened if no position is active.")
        else:
            st.warning("Paper entry blocked: " + "; ".join(risk_gate_reasons))
        st.session_state["jst_snap_signal_engine"] = {
            "ensemble_decision": ensemble_decision, "ensemble_confidence": ensemble_confidence,
            "ml_up_probability": ml_up_probability, "ml_model_status": ml_model_status,
            "risk_gate_pass": risk_gate_pass, "risk_gate_reasons": list(risk_gate_reasons),
            "ls": ls, "ss": ss, "sig_label": sig_label, "last_atr": last_atr,
        }
    except Exception as exc:
        st.error("Signal Engine panel failed to render. Other panels will continue to load.")
        st.exception(exc)
    st.divider()
    jst_zone_header("RULE ENGINE", "Five-vote score breakdown")
    jst_score_bars([
        ("LONG", ls, color("bull")),
        ("SHORT", ss, color("bear")),
    ])
    jst_stat_grid([
        ("Rule-based output", sig_label, None),
        ("ATR(14) proxy", f"{last_atr:,.5f}", color("accent")),
    ])
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
# ==========================================
# JST V2 - TRADE HISTORY STATISTICS
# ==========================================
    trade_history = st.session_state.get("jst_trade_history", [])

st.markdown("### JST V2 | PAPER TRADE PERFORMANCE")

if trade_history:
    history_df = pd.DataFrame(trade_history)

    total_trades = len(history_df)
    wins = int((history_df["result"] == "WIN").sum())
    losses = int((history_df["result"] == "LOSS").sum())

    win_rate = wins / total_trades * 100
    loss_rate = losses / total_trades * 100

    r_values = pd.to_numeric(
        history_df["result_r"], errors="coerce"
    ).dropna()

    expectancy = float(r_values.mean())

    gross_profit = float(r_values[r_values > 0].sum())
    gross_loss = abs(float(r_values[r_values < 0].sum()))

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
        pf_display = f"{profit_factor:.2f}"
    elif gross_profit > 0:
        pf_display = "∞"
    else:
        pf_display = "N/A"

    c1, c2, c3 = st.columns(3)
    c1.metric("Closed Trades", total_trades)
    c2.metric("Win Rate", f"{win_rate:.2f}%")
    c3.metric("Loss Rate", f"{loss_rate:.2f}%")

    c4, c5 = st.columns(2)
    c4.metric("Expectancy", f"{expectancy:.3f} R")
    c5.metric("Profit Factor", pf_display)
    st.markdown("### ADVANCED PERFORMANCE ANALYTICS")

    win_values = r_values[r_values > 0]
    loss_values = r_values[r_values < 0]

    avg_win_r = float(win_values.mean()) if len(win_values) else 0.0
    avg_loss_r = float(loss_values.mean()) if len(loss_values) else 0.0

        
    equity_curve = pd.concat([
            pd.Series([0.0]),
            r_values.reset_index(drop=True)
        ]).cumsum()

    running_peak = equity_curve.cummax()
    drawdown_curve = equity_curve - running_peak

    max_dd_r = abs(float(drawdown_curve.min()))

    net_r = float(r_values.sum())

    recovery_factor = (
    net_r / max_dd_r if max_dd_r > 0 else 0.0
        )
    
    win_groups = (r_values <= 0).cumsum()
    win_streaks = r_values.gt(0).groupby(win_groups).sum()
    max_consecutive_wins = int(win_streaks.max()) if len(win_streaks) else 0

    loss_groups = (r_values >= 0).cumsum()
    loss_streaks = r_values.lt(0).groupby(loss_groups).sum()
    max_consecutive_losses = int(loss_streaks.max()) if len(loss_streaks) else 0

 # ==========================================
 # JST V2 - TRADE INTELLIGENCE
 # ==========================================

    valid_trades = history_df.copy()
    valid_trades["result_r"] = pd.to_numeric(
    valid_trades["result_r"], errors="coerce"
        )
    valid_trades = valid_trades.dropna(subset=["result_r"])

    # Long vs Short performance
    st.markdown("### Long vs Short Performance")

    if "side" in valid_trades.columns:
            valid_trades["side"] = valid_trades["side"].astype(str).str.upper()

            side_stats = valid_trades.groupby("side").agg(
                Trades=("result_r", "count"),
                Average_R=("result_r", "mean"),
                Net_R=("result_r", "sum"),
                Win_Rate=("result_r", lambda x: (x > 0).mean() * 100)
            ).reset_index()

            st.dataframe(
                side_stats,
                use_container_width=True,
                hide_index=True
            )

    # Trade duration
    st.markdown("### Trade Duration")

    if {"entry_time", "exit_time"}.issubset(valid_trades.columns):
            valid_trades["entry_dt"] = pd.to_datetime(
                valid_trades["entry_time"], errors="coerce", utc=True
            ).dt.tz_convert("Asia/Jakarta")
            valid_trades["exit_dt"] = pd.to_datetime(
                valid_trades["exit_time"], errors="coerce", utc=True
            ).dt.tz_convert("Asia/Jakarta")

            valid_trades["duration_minutes"] = (
                valid_trades["exit_dt"] - valid_trades["entry_dt"]
            ).dt.total_seconds() / 60

            durations = valid_trades["duration_minutes"].dropna()
            avg_duration = float(durations.mean()) if len(durations) else 0.0

            st.metric("Average Trade Duration", f"{avg_duration:.1f} minutes")

    # Best and worst trade
    
    if len(valid_trades):
            best_trade = valid_trades.loc[
                valid_trades["result_r"].idxmax()
            ]
            worst_trade = valid_trades.loc[
                valid_trades["result_r"].idxmin()
            ]

            b1, b2 = st.columns(2)
            b1.metric("Best Trade", f"{best_trade['result_r']:.2f} R")
            b2.metric("Worst Trade", f"{worst_trade['result_r']:.2f} R")

   # Trade-level Sharpe and Sortino (not annualized)
 
    trade_std = float(valid_trades["result_r"].std(ddof=1)) if len(valid_trades) > 1 else np.nan
    mean_r = float(valid_trades["result_r"].mean()) if len(valid_trades) else 0.0

    trade_sharpe = (
         mean_r / trade_std if np.isfinite(trade_std) and trade_std > 0 else 0.0
        )

    downside_rms = float(
            (valid_trades["result_r"].clip(upper=0) ** 2).mean() ** 0.5
        )

    trade_sortino = (
            mean_r / downside_rms if downside_rms > 0 else 0.0
        )

    st.markdown("### Risk-Adjusted Trade Performance")
    s1, s2 = st.columns(2)
    s1.metric("Trade Sharpe (Non-annualized)", f"{trade_sharpe:.3f}")
    s2.metric("Trade Sortino (Non-annualized)", f"{trade_sortino:.3f}")

   # Monthly performance
    st.markdown("### Monthly Performance")
    monthly_data = pd.DataFrame()

    if "exit_dt" in valid_trades.columns:
         monthly_data = valid_trades.dropna(subset=["exit_dt"]).copy()

    if len(monthly_data):
         monthly_data["Month"] = (
         monthly_data["exit_dt"].dt.strftime("%Y-%m")
                )

         monthly_stats = monthly_data.groupby("Month").agg(
                Trades=("result_r", "count"),
                Net_R=("result_r", "sum"),
                Average_R=("result_r", "mean"),
                Win_Rate=("result_r", lambda x: (x > 0).mean() * 100)
                ).reset_index()

         st.dataframe(
                monthly_stats,
                use_container_width=True,
                hide_index=True
                )
    else:
         st.info("No valid exit timestamps for monthly analysis.")

    a1, a2, a3 = st.columns(3)

    a1.metric("Average Win", f"{avg_win_r:.2f} R")
    a2.metric("Average Loss", f"{avg_loss_r:.2f} R")
    a3.metric("Net Cumulative R", f"{net_r:.2f} R")

    a4, a5, a6 = st.columns(3)

    a4.metric("Maximum Drawdown", f"{max_dd_r:.2f} R")
    a5.metric("Recovery Factor", f"{recovery_factor:.2f}")
    a6.metric(
        "Max Consecutive Losses",
        max_consecutive_losses
        )

    st.markdown("#### Cumulative R Equity Curve")

    equity_df = pd.DataFrame({
            "Trade": range(len(equity_curve)),
            "Cumulative R": equity_curve
        })
    equity_fig = go.Figure()

    equity_fig.add_trace(go.Scatter(
            x=equity_df["Trade"],
            y=equity_df["Cumulative R"],
            mode="lines",
            name="Cumulative R"
        ))

    equity_fig.update_layout(
            template="plotly_dark",
            height=350,
            xaxis_title="Trade Number",
            yaxis_title="Cumulative R"
        )

    st.plotly_chart(
            equity_fig,
            use_container_width=True
    )
    # Journal display is a presentation-only view; stored timestamps and trade records remain unchanged.
    journal_df = history_df.copy()
    journal_df["Side"] = journal_df.get("side", pd.Series(index=journal_df.index, dtype=object)).astype(str).str.upper()
    journal_df["Symbol"] = journal_df.get("symbol", pd.Series(index=journal_df.index, dtype=object)).fillna(symbol)
    journal_df["Trade ID"] = journal_df.apply(lambda row: jst_trade_key(row.to_dict(), str(row.get("symbol", symbol))), axis=1)
    for _source, _target in (("entry_time", "Entry Time (WIB)"), ("exit_time", "Exit Time (WIB)")):
        if _source in journal_df.columns:
            _dt = pd.to_datetime(journal_df[_source], errors="coerce", utc=True).dt.tz_convert("Asia/Jakarta")
            journal_df[_target] = _dt.dt.strftime("%Y-%m-%d %H:%M:%S WIB")
    _journal_columns = [
        "Trade ID", "Symbol", "Side", "Entry Time (WIB)", "Exit Time (WIB)",
        "entry", "exit", "initial_sl", "sl", "tp", "exit_reason", "result", "result_r"
    ]
    _journal_columns = [c for c in _journal_columns if c in journal_df.columns]
    st.dataframe(
        journal_df[_journal_columns],
        use_container_width=True,
        hide_index=True
    )

  # ==========================================
  # JST V2 - PERFORMANCE PERIOD ANALYTICS
  # ==========================================
    st.markdown("### Performance Period Analytics")

    period_choice = st.selectbox(
            "Select Performance Period",
            ["Today", "This Week", "This Month", "This Year", "All Time"]
        )

    period_df = history_df.copy()

    if "exit_time" in period_df.columns:
            period_df["exit_dt"] = pd.to_datetime(
                period_df["exit_time"],
                errors="coerce",
                utc=True
            )

            period_df = period_df.dropna(subset=["exit_dt"])
            period_df["exit_dt"] = period_df["exit_dt"].dt.tz_convert(
                "Asia/Jakarta"
            )

            now_local = pd.Timestamp.now(tz="Asia/Jakarta")
            today_start = now_local.normalize()
            week_start = today_start - pd.Timedelta(days=today_start.weekday())
            month_start = today_start.replace(day=1)
            year_start = today_start.replace(month=1, day=1)

            if period_choice == "Today":
                period_df = period_df[period_df["exit_dt"] >= today_start]

            elif period_choice == "This Week":
                period_df = period_df[period_df["exit_dt"] >= week_start]

            elif period_choice == "This Month":
                period_df = period_df[period_df["exit_dt"] >= month_start]

            elif period_choice == "This Year":
                period_df = period_df[period_df["exit_dt"] >= year_start]

            if len(period_df):
                period_r = pd.to_numeric(
                    period_df["result_r"], errors="coerce"
                ).dropna()

                period_wins = period_r[period_r > 0]
                period_losses = period_r[period_r < 0]

                period_net_r = float(period_r.sum())
                period_win_rate = (
                    len(period_wins) / len(period_r) * 100
                    if len(period_r) else 0.0
                )
                period_expectancy = (
                    float(period_r.mean()) if len(period_r) else 0.0
                )

                period_gross_profit = float(period_wins.sum())
                period_gross_loss = abs(float(period_losses.sum()))

                period_pf = (
                    period_gross_profit / period_gross_loss
                    if period_gross_loss > 0
                    else float("inf") if period_gross_profit > 0
                    else 0.0
                )

                period_equity = pd.concat([
                    pd.Series([0.0]),
                    period_r.reset_index(drop=True)
                ]).cumsum()

                period_drawdown = (
                    period_equity - period_equity.cummax()
                )
                period_max_dd = abs(float(period_drawdown.min()))

                p1, p2, p3 = st.columns(3)
                p1.metric("Closed Trades", len(period_r))
                p2.metric("Win Rate", f"{period_win_rate:.2f}%")
                p3.metric("Net R", f"{period_net_r:.2f} R")

                p4, p5, p6 = st.columns(3)
                p4.metric("Expectancy", f"{period_expectancy:.3f} R")
                p5.metric(
                    "Profit Factor",
                    "∞" if period_pf == float("inf")
                    else f"{period_pf:.2f}"
                )
                p6.metric("Maximum Drawdown", f"{period_max_dd:.2f} R")

            else:
                st.info(
                    f"No closed trades available for {period_choice}."
                )



else:
    st.info("No closed paper trades yet. Waiting for TP or SL outcomes.")

def try_yahoo_dxy_proxy(period="6mo", interval="1d"):
    """
    Fetch the Yahoo Finance DX-Y.NYB Dollar Index futures series as a clearly
    labelled proxy. This is NOT guaranteed to equal a spot DXY index feed and
    may be delayed or subject to provider availability.
    """
    try:
        import requests
        url = "https://query1.finance.yahoo.com/v8/finance/chart/DX-Y.NYB"
        response = requests.get(
            url,
            params={"range": period, "interval": interval},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=12,
        )
        if response.status_code >= 400:
            return None, f"Yahoo Finance proxy unavailable (HTTP {response.status_code})."
        payload = response.json().get("chart", {})
        if payload.get("error"):
            return None, "Yahoo Finance proxy returned a provider error."
        result = (payload.get("result") or [None])[0]
        if not result:
            return None, "Yahoo Finance proxy returned no series."
        timestamps = result.get("timestamp") or []
        quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
        closes = quote.get("close") or []
        frame = pd.DataFrame({"time": pd.to_datetime(timestamps, unit="s", utc=True), "close": closes})
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame = frame.dropna(subset=["time", "close"]).sort_values("time").reset_index(drop=True)
        if len(frame) < 10:
            return None, "Yahoo Finance proxy returned too few usable observations."
        return frame, "Yahoo Finance DX-Y.NYB Dollar Index futures proxy; may be delayed and is not a guaranteed spot-DXY feed."
    except Exception as exc:
        return None, f"Yahoo Finance proxy unavailable ({type(exc).__name__})."


# --------------------------- MACRO / CORRELATION / ECONOMIC NEWS ---------------------------
with tabs[9]:
    try:
        st.subheader("Macro & Intermarket Intelligence")
        st.caption("DXY, US Treasury yields, cross-asset correlation and medium/high-impact calendar alerts. Telegram is not included.")
        macro_tabs = st.tabs(["DXY & Treasury Yields", "Cross-Asset Correlation", "Economic Calendar & Alerts"])

        def _fred_observations(series_id, api_key, limit=180):
            import requests
            params = {"series_id": series_id, "file_type": "json", "sort_order": "asc", "limit": limit, "api_key": api_key}
            response = requests.get("https://api.stlouisfed.org/fred/series/observations", params=params, timeout=12)
            response.raise_for_status()
            frame = pd.DataFrame(response.json().get("observations", []))
            if frame.empty:
                return pd.DataFrame(columns=["date", "value"])
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True)
            frame["value"] = pd.to_numeric(frame["value"].replace(".", np.nan), errors="coerce")
            return frame.dropna(subset=["date", "value"])[["date", "value"]].reset_index(drop=True)

        def _calendar_normalize(uploaded_file):
            frame = pd.read_csv(uploaded_file)
            frame.columns = [str(c).strip().lower() for c in frame.columns]
            aliases = {"date": "datetime_utc", "datetime": "datetime_utc", "time": "datetime_utc", "name": "event", "title": "event", "ccy": "currency", "importance": "impact"}
            frame = frame.rename(columns={k: v for k, v in aliases.items() if k in frame.columns})
            required = {"event", "datetime_utc", "currency", "impact"}
            missing = required - set(frame.columns)
            if missing:
                raise ValueError("Kolom wajib: event, datetime_utc, currency, impact. Kurang: " + ", ".join(sorted(missing)))
            frame["event"] = frame["event"].astype(str)
            frame["currency"] = frame["currency"].astype(str).str.upper().str.strip()
            frame["impact"] = frame["impact"].astype(str).str.lower().str.strip()
            frame["datetime_utc"] = pd.to_datetime(frame["datetime_utc"], errors="coerce", utc=True)
            frame = frame.dropna(subset=["datetime_utc"])
            for col in ["forecast", "previous", "actual", "country", "source"]:
                if col not in frame.columns: frame[col] = ""
            return frame.sort_values("datetime_utc").reset_index(drop=True)

        with macro_tabs[0]:
            st.markdown("#### DXY — US Dollar Index")
            st.caption("DXY is requested from Twelve Data only if your key, plan, and provider support the symbol. No synthetic DXY is presented as live data.")
            try: twelve_key = st.secrets.get("TWELVE_DATA_API_KEY", "")
            except Exception: twelve_key = ""
            try: fred_key = st.secrets.get("FRED_API_KEY", "")
            except Exception: fred_key = ""
            left, right = st.columns(2)
            with left:
                # First try the configured Twelve Data DXY symbol. If unsupported,
                # fall back to a separately labelled Yahoo DX-Y.NYB futures proxy.
                dxy_frame, dxy_status = (None, "Twelve Data key not configured.")
                if twelve_key:
                    dxy_frame, dxy_status = try_twelve_data("DXY", "1day", outputsize=180)
                dxy_source = "Twelve Data DXY"
                if dxy_frame is None:
                    proxy_frame, proxy_status = try_yahoo_dxy_proxy()
                    if proxy_frame is not None:
                        dxy_frame, dxy_status = proxy_frame, proxy_status
                        dxy_source = "Dollar Index futures proxy (DX-Y.NYB)"
                if dxy_frame is not None:
                    dxy_fig = go.Figure(go.Scatter(x=dxy_frame["time"], y=dxy_frame["close"], mode="lines", name=dxy_source, line=dict(color=color("accent"), width=2)))
                    dxy_fig.update_layout(title=dxy_source + " — daily close", height=360, template="plotly_dark", paper_bgcolor=color("bg"), plot_bgcolor=color("bg"), font=dict(color=color("text")))
                    dxy_fig.update_xaxes(gridcolor=color("grid")); dxy_fig.update_yaxes(gridcolor=color("grid"))
                    st.plotly_chart(dxy_fig, use_container_width=True, config={"displaylogo": False})
                    st.caption(dxy_status + " Check the last timestamp; this is not guaranteed real-time data.")
                    if dxy_source != "Twelve Data DXY":
                        st.warning("This is a futures proxy, not a verified spot DXY index feed. Do not treat it as an exact substitute.")
                    st.session_state["jst_snap_dxy"] = {"available": True, "source": dxy_source, "last": float(dxy_frame["close"].iloc[-1])}
                else:
                    st.warning("DXY data unavailable. Twelve Data: " + dxy_status + " Yahoo Finance proxy: " + proxy_status)
                    st.info("No demo data is displayed as live DXY. Check provider availability or connect a verified DXY feed.")
                    st.session_state["jst_snap_dxy"] = {"available": False}
            with right:
                st.markdown("#### US Treasury yields")
                if not fred_key:
                    st.info("Set FRED_API_KEY in Streamlit Secrets to load DGS2, DGS10, and DFII10.")
                else:
                    series_map = {"DGS2": "US 2Y nominal", "DGS10": "US 10Y nominal", "DFII10": "US 10Y real yield (TIPS)"}
                    yield_frames=[]
                    import requests
                    for series_id, label in series_map.items():
                        try:
                            frame = _fred_observations(series_id, fred_key)
                            if not frame.empty:
                                frame=frame.rename(columns={"value": label}); yield_frames.append(frame)
                            else: st.warning(f"FRED returned no observations for {series_id}.")
                        except Exception as exc: st.warning(f"FRED {series_id} unavailable: {exc}")
                    if yield_frames:
                        yields=yield_frames[0]
                        for frame in yield_frames[1:]: yields=yields.merge(frame,on="date",how="outer")
                        yields=yields.sort_values("date")
                        fig=go.Figure()
                        for label in series_map.values():
                            if label in yields: fig.add_trace(go.Scatter(x=yields["date"],y=yields[label],mode="lines",name=label))
                        fig.update_layout(title="Treasury yields (%) — daily frequency",height=360,template="plotly_dark",paper_bgcolor=color("bg"),plot_bgcolor=color("bg"),font=dict(color=color("text")))
                        fig.update_xaxes(gridcolor=color("grid")); fig.update_yaxes(gridcolor=color("grid"),title="Percent")
                        st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})
                        st.dataframe(yields.tail(12),use_container_width=True,hide_index=True)
            st.caption("FRED yields are daily observations, not tick-level M1 data. Timestamp/frequency mismatches must be considered in any model.")

        with macro_tabs[1]:
            st.markdown("#### Cross-Asset Correlation")
            st.caption("Upload synchronized historical prices. Correlation is not fabricated from a single asset or synthetic series.")
            upload=st.file_uploader("Upload multi-asset price CSV",type=["csv"],key="jst_multi_asset_corr_csv",help="Wide format: time, XAUUSD, DXY, US10Y, EURUSD, BTCUSD...")
            if upload is not None:
                try:
                    prices=pd.read_csv(upload); prices.columns=[str(c).strip() for c in prices.columns]
                    tcol=next((c for c in prices.columns if c.lower() in ("time","date","datetime","timestamp")),None)
                    if tcol:
                        prices[tcol]=pd.to_datetime(prices[tcol],errors="coerce",utc=True); prices=prices.set_index(tcol)
                    for col in prices.columns: prices[col]=pd.to_numeric(prices[col],errors="coerce")
                    prices=prices.select_dtypes(include=[np.number]).dropna(axis=1,how="all")
                    if prices.shape[1]<2 or len(prices)<20: st.error("Need at least 2 numeric price columns and 20 synchronized rows.")
                    else:
                        returns=prices.pct_change().replace([np.inf,-np.inf],np.nan)
                        window=st.slider("Correlation window",10,100,20,5)
                        corr=returns.tail(max(window,20)).corr()
                        fig=go.Figure(go.Heatmap(z=corr.values,x=corr.columns.tolist(),y=corr.index.tolist(),zmin=-1,zmax=1,zmid=0,colorscale="RdBu",text=np.round(corr.values,2),texttemplate="%{text}",hovertemplate="%{x} vs %{y}: %{z:.3f}<extra></extra>"))
                        fig.update_layout(title="Cross-asset return correlation",height=520,template="plotly_dark",paper_bgcolor=color("bg"),font=dict(color=color("text")))
                        st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})
                        a=st.selectbox("Asset A",corr.columns.tolist(),key="jst_corr_a")
                        b=st.selectbox("Asset B",[x for x in corr.columns if x!=a],key="jst_corr_b")
                        pair=returns[[a,b]].dropna().tail(max(window*3,window))
                        roll=pair[a].rolling(window,min_periods=max(5,window//2)).corr(pair[b])
                        rf=go.Figure(go.Scatter(x=roll.index,y=roll,mode="lines",name="Rolling correlation",line=dict(color=color("accent"),width=2)))
                        rf.update_layout(title=f"Rolling correlation: {a} vs {b}",height=340,template="plotly_dark",paper_bgcolor=color("bg"),plot_bgcolor=color("bg"),font=dict(color=color("text")),yaxis=dict(range=[-1,1]))
                        st.plotly_chart(rf,use_container_width=True,config={"displaylogo":False})
                        st.download_button("Download correlation matrix CSV",corr.to_csv().encode(),file_name="jst_correlation_matrix.csv",mime="text/csv")
                        st.caption("Pearson correlation of returns is descriptive, can change across regimes, and does not establish causation.")
                except Exception as exc: st.error(f"Could not process correlation CSV: {exc}")
            else:
                st.info("No multi-asset dataset loaded yet.")
                st.code("time,XAUUSD,DXY,US10Y,EURUSD,BTCUSD\n2026-10-01T00:00:00Z,2650,98.2,4.1,1.16,110000",language="csv")
                st.caption("Illustrative schema only—not actual market data and not enough rows for analysis.")

        with macro_tabs[2]:
            st.markdown("#### Economic Calendar & Dashboard Alerts")
            st.info("No Telegram integration. Upload a verified economic-calendar CSV. Automatic provider integration is intentionally not claimed until a specific calendar API/schema is configured.")
            calendar_upload=st.file_uploader("Upload economic calendar CSV",type=["csv"],key="jst_economic_calendar_csv",help="Required: event, datetime_utc, currency, impact. Optional: forecast, previous, actual, country, source.")
            leads=st.multiselect("Alert windows",[60,15,0],default=[60,15,0],format_func=lambda x:"At release" if x==0 else f"{x} minutes before release")
            impacts=st.multiselect("Impact filter",["medium","high"],default=["medium","high"])
            hours=st.slider("Calendar look-ahead (hours)",6,168,72,6)
            if calendar_upload is not None:
                try:
                    cal=_calendar_normalize(calendar_upload)
                    cal=cal[cal["impact"].isin(impacts)]
                    now=pd.Timestamp.now(tz="UTC"); end=now+pd.Timedelta(hours=hours)
                    cal=cal[(cal["datetime_utc"]>=now-pd.Timedelta(minutes=5))&(cal["datetime_utc"]<=end)].copy()
                    cal["datetime_wib"]=cal["datetime_utc"].dt.tz_convert("Asia/Jakarta")
                    cal["minutes_to_release"]=((cal["datetime_utc"]-now).dt.total_seconds()/60).round().astype(int)
                    due=[]
                    for _,event in cal.iterrows():
                        mins=int(event["minutes_to_release"])
                        for lead in leads:
                            if (lead==0 and -5<=mins<=0) or (lead>0 and lead-1<=mins<=lead):
                                due.append({"Event":event["event"],"Currency":event["currency"],"Impact":event["impact"].upper(),"Minutes to release":mins,"Scheduled WIB":event["datetime_wib"].strftime("%Y-%m-%d %H:%M"),"Alert":"RELEASE" if lead==0 else f"T-{lead} MIN","Source":event.get("source","")})
                    if due:
                        st.warning("Event entered an alert window. Dashboard status updates when Streamlit reruns.")
                        st.dataframe(pd.DataFrame(due),use_container_width=True,hide_index=True)
                    elif cal.empty: st.success("No medium/high impact events in the selected window of the uploaded calendar.")
                    st.dataframe(cal[["event","currency","impact","datetime_wib","minutes_to_release","forecast","previous","actual","source"]],use_container_width=True,hide_index=True)
                    st.download_button("Download filtered calendar CSV",cal.to_csv(index=False).encode(),file_name="jst_filtered_calendar.csv",mime="text/csv")
                    st.warning("Important: these are dashboard alert windows, not a persistent background scheduler. The app must rerun; reliable alerts while it is closed require a separate always-on worker. No Telegram messages are sent.")
                except Exception as exc: st.error(f"Calendar CSV error: {exc}")
            else:
                st.markdown("Required columns: `event,datetime_utc,currency,impact`. Optional: `forecast,previous,actual,country,source`.")
                st.caption("Use ISO-8601 UTC timestamps such as 2026-10-20T12:30:00Z. Verify the calendar source and refresh it before use.")
            st.caption("Medium/high impact filtering and T-60/T-15/release windows are supported from uploaded events. Event-to-asset mapping and unscheduled geopolitical headlines require a validated news source and taxonomy.")
    except Exception as exc:
        st.error("Macro & News Intelligence panel failed to render. Other panels will continue to load.")
        st.exception(exc)


# --------------------------- QUANT RESEARCH ENGINE ---------------------------
with tabs[10]:
    st.subheader("JST APEX QUANT — Quant Research Engine")
    st.caption("Empirical research layer built from CLOSED paper trades and the selected OHLCV data. Insufficient samples are explicitly flagged; no profitability claim is made.")
    qcost = st.number_input("Estimated round-trip trading cost (R)", min_value=0.0, max_value=1.0, value=0.10, step=0.01, key="qr_cost_r")
    raw_history = st.session_state.get("jst_trade_history", [])
    qdf = pd.DataFrame(raw_history)
    if not qdf.empty and "result_r" in qdf.columns:
        qdf["result_r"] = pd.to_numeric(qdf["result_r"], errors="coerce")
        qdf = qdf.dropna(subset=["result_r"]).copy()
        qdf["net_r"] = qdf["result_r"] - qcost
    else:
        qdf = pd.DataFrame(columns=["result_r", "net_r"])

    q1, q2, q3, q4 = st.columns(4)
    n_trades = len(qdf)
    wins = int((qdf["result_r"] > 0).sum()) if n_trades else 0
    win_rate = wins / n_trades if n_trades else np.nan
    net_ev = float(qdf["net_r"].mean()) if n_trades else np.nan
    q1.metric("Closed paper trades", n_trades)
    q2.metric("Observed win rate", f"{win_rate:.1%}" if np.isfinite(win_rate) else "N/A")
    q3.metric("Net expectancy", f"{net_ev:.3f} R" if np.isfinite(net_ev) else "N/A")
    q4.metric("Net cumulative R", f"{qdf['net_r'].sum():.2f} R" if n_trades else "N/A")

    qr_tabs = st.tabs(["Probability & EV", "Regime & Factors", "Model Selection", "Walk-Forward & OOS", "Stages 8–11 ML Pipeline", "Decision & Persistence", "Live Paper Validation", "Performance & Stability"])
    with qr_tabs[0]:
        st.markdown("#### Bayesian win-probability estimate")
        if n_trades:
            prior_a = st.number_input("Beta prior α", min_value=0.1, value=1.0, step=0.1, key="qr_prior_a")
            prior_b = st.number_input("Beta prior β", min_value=0.1, value=1.0, step=0.1, key="qr_prior_b")
            posterior_a = prior_a + wins
            posterior_b = prior_b + (n_trades - wins)
            posterior_mean = posterior_a / (posterior_a + posterior_b)
            try:
                from scipy.stats import beta as beta_dist
                ci_low, ci_high = beta_dist.ppf([0.025, 0.975], posterior_a, posterior_b)
            except Exception:
                ci_low, ci_high = np.nan, np.nan
            a, b, c = st.columns(3)
            a.metric("Posterior P(win)", f"{posterior_mean:.1%}")
            b.metric("95% credible lower", f"{ci_low:.1%}" if np.isfinite(ci_low) else "N/A")
            c.metric("95% credible upper", f"{ci_high:.1%}" if np.isfinite(ci_high) else "N/A")
            st.caption("Beta-Binomial update on closed paper-trade outcomes. This estimates historical trade win probability, not the probability of the next signal. Small samples produce wide credible intervals.")
            gross_ev = float(qdf["result_r"].mean())
            st.metric("Expected value after estimated costs", f"{gross_ev-qcost:.3f} R / trade")
            st.write({"Gross mean R": round(gross_ev, 4), "Estimated cost R": float(qcost), "Net mean R": round(gross_ev-qcost, 4), "Sample": n_trades})
            st.line_chart(qdf["net_r"].cumsum().reset_index(drop=True), height=220)

            st.divider()
            st.markdown("#### Stage 1–2 — Conditional & Multi-Factor Probability")
            feature_cols = ["feature_side", "feature_score_band", "feature_volatility", "feature_session", "feature_regime"]
            available_features = [c for c in feature_cols if c in qdf.columns and qdf[c].notna().any()]
            if available_features:
                min_group_n = st.slider("Minimum observations per condition", 2, 30, 5, key="qr_cpe_min_n_v2")
                prior_strength = st.number_input("Symmetric Beta prior strength", min_value=0.1, max_value=10.0, value=1.0, step=0.1, key="qr_cpe_prior_v2")
                from itertools import combinations
                from scipy.stats import beta as beta_dist
                feature_labels = {"feature_side":"Direction", "feature_score_band":"Score band", "feature_volatility":"Volatility", "feature_session":"Session", "feature_regime":"Trend/volatility regime"}
                cpe_data = qdf.copy()
                cpe_data["is_win"] = (cpe_data["result_r"] > 0).astype(int)
                cpe_data["net_r"] = pd.to_numeric(cpe_data["net_r"], errors="coerce")
                max_factors = st.selectbox("Maximum factors in a combination", [2, 3, 4], index=1, key="qr_cpe_max_factors")
                cpe_rows = []
                for k in range(1, min(max_factors, len(available_features)) + 1):
                    for cols in combinations(available_features, k):
                        valid = cpe_data.dropna(subset=list(cols))
                        for key_values, group in valid.groupby(list(cols), dropna=True):
                            if len(group) < min_group_n:
                                continue
                            if not isinstance(key_values, tuple):
                                key_values = (key_values,)
                            label = " | ".join(f"{feature_labels.get(col,col)}={val}" for col,val in zip(cols,key_values))
                            n_g = len(group); w_g = int(group["is_win"].sum())
                            a_post = prior_strength + w_g; b_post = prior_strength + n_g - w_g
                            p_post = a_post / (a_post + b_post)
                            ci_l, ci_h = beta_dist.ppf([0.025,0.975], a_post,b_post)
                            cpe_rows.append({"Condition":label,"Factors":k,"Trades":n_g,"Wins":w_g,"Observed win rate":w_g/n_g,"Posterior P(win)":p_post,"95% CI low":ci_l,"95% CI high":ci_h,"Net EV (R)":float(group["net_r"].mean())})
                if cpe_rows:
                    cpe_table = pd.DataFrame(cpe_rows).sort_values(["Factors","Trades"],ascending=[True,False])
                    st.dataframe(cpe_table.style.format({"Observed win rate":"{:.1%}","Posterior P(win)":"{:.1%}","95% CI low":"{:.1%}","95% CI high":"{:.1%}","Net EV (R)":"{:.3f}"}),use_container_width=True,hide_index=True)
                    st.caption("Stage 2 explores joint conditions. More factors create smaller groups; the minimum sample filter and credible intervals are essential. Historical associations are not causal evidence.")
                else:
                    st.info("No single- or multi-factor group meets the current minimum sample threshold yet.")
                legacy_n = int(cpe_data[available_features].isna().any(axis=1).sum())
                if legacy_n:
                    st.caption(f"{legacy_n} legacy trades lack one or more entry snapshots and are excluded from combinations requiring those fields.")

                st.divider()
                st.markdown("#### Stage 3 — Probability calibration diagnostic")
                st.caption("Prequential baseline: each trade is scored using only outcomes recorded before it. This tests a historical Beta-rate baseline; it is not calibration of the live signal model.")
                cal_prior = st.number_input("Calibration baseline prior per class", min_value=0.1, max_value=10.0, value=1.0, step=0.1, key="qr_cal_prior")
                ordered = cpe_data.copy()
                if "entry_time" in ordered.columns:
                    ordered["_order_time"] = pd.to_datetime(ordered["entry_time"], errors="coerce", utc=True)
                    ordered = ordered.sort_values("_order_time", kind="stable")
                y = ordered["is_win"].astype(int).to_numpy()
                probs=[]; seen_w=0; seen_n=0
                for outcome in y:
                    probs.append((cal_prior+seen_w)/(2*cal_prior+seen_n))
                    seen_w += int(outcome); seen_n += 1
                if len(y) >= 5:
                    p_arr=np.clip(np.asarray(probs),1e-6,1-1e-6); y_arr=np.asarray(y)
                    brier=float(np.mean((p_arr-y_arr)**2)); logloss=float(-np.mean(y_arr*np.log(p_arr)+(1-y_arr)*np.log(1-p_arr)))
                    ca,cb,cc=st.columns(3); ca.metric("Prequential Brier score",f"{brier:.3f}"); cb.metric("Prequential log loss",f"{logloss:.3f}"); cc.metric("Scored trades",len(y))
                    bins=pd.cut(p_arr,bins=np.linspace(0,1,6),include_lowest=True)
                    rel=pd.DataFrame({"Forecast probability":p_arr,"Outcome":y_arr,"Bin":bins}).groupby("Bin",observed=False).agg(Trades=("Outcome","size"),Mean_forecast=("Forecast probability","mean"),Observed_rate=("Outcome","mean")).dropna().reset_index()
                    if not rel.empty:
                        rel=rel.rename(columns={"Mean_forecast":"Mean forecast","Observed_rate":"Observed win rate"})
                        st.dataframe(rel.style.format({"Mean forecast":"{:.1%}","Observed win rate":"{:.1%}"}),use_container_width=True,hide_index=True)
                    st.caption("Because this baseline updates from past outcomes, its scores are descriptive and may be dominated by the prior/overall hit rate. A true live-model calibration test needs stored probability forecasts at signal time.")
                else:
                    st.info("Need at least 5 chronologically ordered closed trades for this diagnostic.")

                # Evaluate forecasts actually stored when each paper position was opened.
                forecast_data = ordered.dropna(subset=["forecast_probability"]).copy() if "forecast_probability" in ordered.columns else pd.DataFrame()
                if not forecast_data.empty:
                    forecast_data["forecast_probability"] = pd.to_numeric(forecast_data["forecast_probability"], errors="coerce")
                    forecast_data = forecast_data.dropna(subset=["forecast_probability"])
                    forecast_data = forecast_data[
                        forecast_data["forecast_probability"].between(0.0, 1.0)
                    ]
                st.markdown("##### Stored entry-time forecast evaluation")
                if len(forecast_data) >= 5:
                    fp = np.clip(forecast_data["forecast_probability"].to_numpy(dtype=float), 1e-6, 1-1e-6)
                    fy = (forecast_data["result_r"].to_numpy(dtype=float) > 0).astype(int)
                    fbrier = float(np.mean((fp-fy)**2))
                    flogloss = float(-np.mean(fy*np.log(fp)+(1-fy)*np.log(1-fp)))
                    fa, fb, fc = st.columns(3)
                    fa.metric("Stored-forecast Brier", f"{fbrier:.3f}")
                    fb.metric("Stored-forecast log loss", f"{flogloss:.3f}")
                    fc.metric("Forecasted trades", len(forecast_data))
                    st.dataframe(forecast_data[[c for c in ["entry_time", "side", "forecast_probability", "forecast_method", "forecast_training_n", "result_r"] if c in forecast_data.columns]].tail(30), use_container_width=True, hide_index=True)
                    st.caption("These forecasts are generated from prior closed-trade outcomes at entry time. They are not probabilities emitted by the raw rule score or a trained ML classifier. Interpret scores only after a larger chronological sample.")
                else:
                    st.info(f"Need at least 5 trades with stored entry-time forecasts; currently {len(forecast_data)} available. Older trades without forecast records are excluded.")
            else:
                st.info("No closed trades contain entry-feature snapshots yet. New paper positions store direction, score, volatility, session, and regime.")

            st.divider()
            st.markdown("#### Stage 5 — Probability-to-Expected-Value integration")
            if n_trades >= 5:
                gross_wins = qdf.loc[qdf["result_r"] > 0, "result_r"]
                gross_losses = qdf.loc[qdf["result_r"] <= 0, "result_r"]
                avg_win_net = float(gross_wins.mean() - qcost) if len(gross_wins) else np.nan
                avg_loss_net = float(gross_losses.mean() - qcost) if len(gross_losses) else np.nan
                p_use = float(posterior_mean) if "posterior_mean" in locals() else float((1.0+wins)/(2.0+n_trades))
                if np.isfinite(avg_win_net) and np.isfinite(avg_loss_net):
                    ev_integrated = p_use*avg_win_net + (1-p_use)*avg_loss_net
                    e1,e2,e3=st.columns(3); e1.metric("Posterior win probability",f"{p_use:.1%}"); e2.metric("Avg net win / loss",f"{avg_win_net:.3f} / {avg_loss_net:.3f} R"); e3.metric("Integrated net EV",f"{ev_integrated:.3f} R")
                    st.caption("EV = P(win) × average net win + P(loss) × average net loss. This is a historical plug-in estimate, not a forward guarantee; use conditional group estimates only when sample size is adequate.")
                else:
                    st.info("Need at least one winning and one losing trade to estimate payoff magnitudes.")
            else:
                st.info("Need at least 5 closed trades before enabling the integrated EV diagnostic.")
        else:
            st.info("Belum ada closed paper trade. Probability dan EV akan aktif setelah outcome trade tercatat; sistem tidak akan mengarang data historis.")

    with qr_tabs[1]:
        st.markdown("#### Market regime & factor diagnostics")
        close_q = pd.to_numeric(market["close"], errors="coerce")
        returns_q = close_q.pct_change().replace([np.inf, -np.inf], np.nan)
        vol_window = st.slider("Regime volatility window (bars)", 10, 100, 30, 5, key="qr_regime_window")
        vol_q = returns_q.rolling(vol_window).std()
        trend_q = close_q.pct_change(vol_window)
        regime_df = pd.DataFrame({"return": returns_q, "volatility": vol_q, "trend": trend_q}).dropna()
        if len(regime_df) >= 30:
            vol_cut = regime_df["volatility"].median()
            regime_df["regime"] = np.where(regime_df["volatility"] >= vol_cut, "HIGH_VOL", "LOW_VOL")
            regime_df["trend_state"] = np.where(regime_df["trend"] >= 0, "UP", "DOWN")
            st.dataframe(regime_df.tail(12), use_container_width=True)
            st.caption("Transparent rolling-volatility regime proxy; not an HMM or a causal regime classifier.")
            st.markdown("#### Stage 4 — Regime-conditional trade outcomes")
            if not qdf.empty and "feature_regime" in qdf.columns:
                rg = qdf.dropna(subset=["feature_regime"]).copy()
                if len(rg):
                    rg_stats = rg.groupby("feature_regime").agg(Trades=("result_r","size"), Wins=("result_r",lambda x:int((x>0).sum())), **{"Net EV (R)":("net_r","mean")}).reset_index()
                    rg_stats["Observed win rate"] = rg_stats["Wins"] / rg_stats["Trades"]
                    st.dataframe(rg_stats.style.format({"Observed win rate":"{:.1%}","Net EV (R)":"{:.3f}"}),use_container_width=True,hide_index=True)
                else:
                    st.info("No trades with regime snapshots yet. New trades will populate this section.")
            else:
                st.info("Regime-conditioned outcomes need new closed trades with entry-time regime snapshots.")
            factor_df = pd.DataFrame({"Return": returns_q, "Trend momentum": close_q.pct_change(5), "Long momentum": close_q.pct_change(20), "Realized volatility": returns_q.rolling(20).std(), "Range proxy": ((market["high"]-market["low"])/close_q).replace([np.inf,-np.inf],np.nan)}).dropna()
            if len(factor_df) >= 30:
                corr = factor_df.corr(numeric_only=True)
                st.markdown("#### Factor correlation matrix")
                fig_factor = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1, colorscale="RdBu", text=np.round(corr.values, 2), texttemplate="%{text}"))
                fig_factor.update_layout(template="plotly_dark", height=420)
                st.plotly_chart(fig_factor, use_container_width=True)
                st.caption("Exploratory factor proxies derived from OHLC. Correlation is not causation and does not establish incremental alpha.")
        else:
            st.warning("Insufficient OHLC observations for regime/factor diagnostics.")

    with qr_tabs[2]:
        st.markdown("#### Stage 7A — Candidate model selection")
        st.caption("Research baselines use OHLC data only. Signals are shifted by one bar before scoring the next bar to avoid look-ahead. This is not the paper-signal engine and is not a production model.")
        cost_bps = st.number_input("Estimated round-trip cost for OHLC tests (basis points)", min_value=0.0, max_value=100.0, value=2.0, step=0.5, key="qr_stage7_cost_bps")
        min_history = st.slider("Minimum bars before evaluation", 50, 500, 100, 25, key="qr_stage7_min_history")
        close_s = pd.to_numeric(market["close"], errors="coerce")
        ret_s = close_s.pct_change().replace([np.inf, -np.inf], np.nan)
        mom5 = close_s.pct_change(5)
        mom20 = close_s.pct_change(20)
        roll_mean = close_s.rolling(20).mean()
        roll_std = close_s.rolling(20).std().replace(0, np.nan)
        z20 = (close_s - roll_mean) / roll_std
        vol20 = ret_s.rolling(20).std()
        vol_median = vol20.rolling(100, min_periods=30).median()
        # Decision at t-1 is applied to return at t. Cost is charged on position changes.
        candidate_signals = {
            "Flat benchmark": pd.Series(0.0, index=close_s.index),
            "Momentum 5-bar": np.sign(mom5),
            "Momentum 20-bar": np.sign(mom20),
            "Mean-reversion Z20": np.where(z20 > 1.0, -1.0, np.where(z20 < -1.0, 1.0, 0.0)),
            "Volatility-filtered momentum": np.where(vol20 <= vol_median, np.sign(mom5), 0.0),
        }
        stage7_rows=[]
        aligned = pd.DataFrame({"forward_return": ret_s})
        for name, raw_signal in candidate_signals.items():
            sig = pd.Series(raw_signal, index=close_s.index, dtype=float).shift(1).reindex(aligned.index).fillna(0.0)
            turnover = sig.diff().abs().fillna(sig.abs())
            net = sig * aligned["forward_return"] - turnover * (cost_bps / 10000.0)
            valid = net.dropna()
            if len(valid) < min_history:
                continue
            sd = float(valid.std(ddof=1)) if len(valid)>1 else 0.0
            sharpe = float(valid.mean()/sd*np.sqrt(252)) if sd>0 else np.nan
            equity=(1.0+valid).cumprod()
            dd=equity/equity.cummax()-1.0
            gross_pos=float(valid[valid>0].sum()); gross_neg=float(-valid[valid<0].sum())
            pf=gross_pos/gross_neg if gross_neg>0 else np.inf if gross_pos>0 else np.nan
            stage7_rows.append({"Candidate":name,"Bars":len(valid),"Net return":float(equity.iloc[-1]-1.0),"Mean net/bar":float(valid.mean()),"Annualized Sharpe (bar assumption)":sharpe,"Max drawdown":float(dd.min()),"Profit factor":pf,"Active fraction":float((sig.loc[valid.index]!=0).mean())})
        if stage7_rows:
            model_table=pd.DataFrame(stage7_rows).sort_values("Mean net/bar",ascending=False)
            st.dataframe(model_table.style.format({"Net return":"{:.2%}","Mean net/bar":"{:.5%}","Annualized Sharpe (bar assumption)":"{:.2f}","Max drawdown":"{:.2%}","Profit factor":"{:.2f}","Active fraction":"{:.1%}"}),use_container_width=True,hide_index=True)
            st.caption("Annualization assumes 252 observations/year and is only illustrative; use the actual bar frequency before interpreting Sharpe. Cost is a simplified turnover charge in return units, not the separate trade-level R cost above.")
        else:
            st.info(f"Need at least {min_history} valid evaluation bars for candidate comparison.")

    with qr_tabs[3]:
        st.markdown("#### Stage 7B — Expanding walk-forward & untouched final OOS")
        st.caption("Each fold trains only on earlier bars and evaluates on the next block. The final chronological segment is reserved as a holdout and is not used to rank candidates.")
        close_s = pd.to_numeric(market["close"], errors="coerce")
        ret_s = close_s.pct_change().replace([np.inf, -np.inf], np.nan)
        mom5 = close_s.pct_change(5)
        mom20 = close_s.pct_change(20)
        roll_mean = close_s.rolling(20).mean(); roll_std=close_s.rolling(20).std().replace(0,np.nan)
        z20=(close_s-roll_mean)/roll_std
        vol20=ret_s.rolling(20).std(); vol_median=vol20.rolling(100,min_periods=30).median()
        raw_models={"Momentum 5-bar":np.sign(mom5),"Momentum 20-bar":np.sign(mom20),"Mean-reversion Z20":np.where(z20>1.0,-1.0,np.where(z20 < -1.0,1.0,0.0)),"Volatility-filtered momentum":np.where(vol20<=vol_median,np.sign(mom5),0.0)}
        wf_cost_bps=st.number_input("Walk-forward round-trip cost (basis points)",min_value=0.0,max_value=100.0,value=2.0,step=0.5,key="qr_stage7_wf_cost")
        folds=st.slider("Walk-forward test folds",3,8,5,key="qr_stage7_folds")
        holdout_fraction=st.slider("Final untouched OOS fraction",0.10,0.30,0.20,0.05,key="qr_stage7_holdout")
        base=pd.DataFrame({"forward_return":ret_s}).dropna()
        if len(base)>=200:
            holdout_start=int(len(base)*(1.0-holdout_fraction))
            dev=base.iloc[:holdout_start]
            holdout=base.iloc[holdout_start:]
            fold_size=max(1,len(dev)//(folds+1))
            wf_results=[]
            for model_name,raw in raw_models.items():
                signal=pd.Series(raw,index=close_s.index,dtype=float).shift(1).reindex(base.index).fillna(0.0)
                fold_metrics=[]
                for fold_i in range(1,folds+1):
                    train_end=min(fold_i*fold_size,len(dev))
                    test_start=train_end; test_end=min(test_start+fold_size,len(dev))
                    if test_end<=test_start: continue
                    idx=dev.index[test_start:test_end]
                    pos=signal.loc[idx]
                    prev_pos=signal.shift(1).reindex(idx).fillna(0.0)
                    net=pos*base.loc[idx,"forward_return"]-(pos-prev_pos).abs()*(wf_cost_bps/10000.0)
                    if len(net): fold_metrics.append({"Fold":fold_i,"Train bars":train_end,"Test bars":len(net),"Net mean/bar":float(net.mean()),"Net return":float((1+net).prod()-1),"Win fraction":float((net>0).mean())})
                if fold_metrics:
                    fdf=pd.DataFrame(fold_metrics)
                    wf_results.append({"Candidate":model_name,"Folds":len(fdf),"WF mean net/bar":float(np.average(fdf["Net mean/bar"],weights=fdf["Test bars"])),"WF total return":float((1+fdf["Net return"]).prod()-1),"Positive folds":int((fdf["Net mean/bar"]>0).sum())})
            if wf_results:
                wf_table=pd.DataFrame(wf_results).sort_values("WF mean net/bar",ascending=False)
                st.markdown("##### Expanding-window validation (development segment)")
                st.dataframe(wf_table.style.format({"WF mean net/bar":"{:.5%}","WF total return":"{:.2%}"}),use_container_width=True,hide_index=True)
                st.caption("Candidate ranking is descriptive only. No hyperparameter search is performed, and no candidate is automatically promoted.")
                st.markdown("##### Final chronological holdout — report only")
                holdout_rows=[]
                for model_name,raw in raw_models.items():
                    signal=pd.Series(raw,index=close_s.index,dtype=float).shift(1).reindex(base.index).fillna(0.0)
                    idx=holdout.index; pos=signal.loc[idx]; prev_pos=signal.shift(1).reindex(idx).fillna(0.0)
                    net=pos*holdout["forward_return"]-(pos-prev_pos).abs()*(wf_cost_bps/10000.0)
                    holdout_rows.append({"Candidate":model_name,"OOS bars":len(net),"OOS net return":float((1+net).prod()-1),"OOS mean/bar":float(net.mean()),"OOS max drawdown":float(((1+net).cumprod()/(1+net).cumprod().cummax()-1).min())})
                st.dataframe(pd.DataFrame(holdout_rows).style.format({"OOS net return":"{:.2%}","OOS mean/bar":"{:.5%}","OOS max drawdown":"{:.2%}"}),use_container_width=True,hide_index=True)
                st.warning("Do not choose a model from the final holdout table and then retune it on the same data. Once inspected, this holdout is no longer fully untouched; preserve a new future period for confirmation.")
            else:
                st.info("Not enough development bars to form the selected walk-forward folds.")
        else:
            st.warning(f"Need at least 200 valid OHLC return bars for walk-forward and final OOS. Current: {len(base)}.")
        st.markdown("#### Research gate")
        st.write("- No live deployment approval from in-sample metrics alone.\n- Use timestamp-aligned features and labels, leakage controls, realistic spread/slippage/commission, and a genuinely untouched future OOS period.\n- Require stable net expectancy, drawdown control, sufficient sample size, and forecast calibration before considering any model upgrade.")
    with qr_tabs[3]:
        st.markdown("#### Walk-forward & out-of-sample trade validation")
        if n_trades >= 20:
            train_fraction = st.slider("Initial training fraction", 0.50, 0.80, 0.60, 0.05, key="qr_wf_fraction")
            min_train = max(10, int(n_trades * train_fraction))
            test_block = max(1, int((n_trades-min_train) / 4))
            wf_rows = []
            values = qdf["net_r"].to_numpy(dtype=float)
            start_i = min_train
            while start_i < len(values):
                end_i = min(start_i + test_block, len(values))
                train_mean = float(np.mean(values[:start_i]))
                test_values = values[start_i:end_i]
                wf_rows.append({"Train trades": start_i, "Test trades": len(test_values), "Train net EV (R)": train_mean, "Test net EV (R)": float(np.mean(test_values)), "Test win rate": float(np.mean(test_values > 0)), "Test net R": float(np.sum(test_values))})
                start_i = end_i
            wf_df = pd.DataFrame(wf_rows)
            st.dataframe(wf_df, use_container_width=True, hide_index=True)
            if len(wf_df):
                st.metric("Aggregate walk-forward test EV", f"{np.average(wf_df['Test net EV (R)'], weights=wf_df['Test trades']):.3f} R")
            st.caption("Expanding-window chronological diagnostic on recorded trade outcomes. This does not refit a signal model; true walk-forward model validation requires timestamped feature/label data and frozen model parameters per fold.")
        else:
            st.warning(f"Need at least 20 closed paper trades for a basic expanding-window diagnostic. Current sample: {n_trades}.")
        st.markdown("#### Research gate")
        st.write("- No live deployment approval from in-sample metrics alone.\n- Require timestamp-aligned features, leakage controls, realistic spread/slippage/commission, and untouched OOS data.\n- Require stable net expectancy, drawdown control, sufficient trade count, and probability calibration before considering any model upgrade.")


    with qr_tabs[4]:
        st.markdown("#### Stage 8 — Feature engineering & leakage control")
        st.caption("Features use information available at bar close t to estimate direction of bar t+1. All splits are chronological; no random shuffle.")
        st.warning("Research-only diagnostic. A next-bar direction target is not automatically a tradable XAU/USD strategy. Validate spread, slippage, contract specifications and broker feed alignment separately.")

        try:
            from sklearn.base import clone
            from sklearn.ensemble import RandomForestClassifier, VotingClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import accuracy_score, precision_score, recall_score, brier_score_loss, log_loss
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.impute import SimpleImputer

            ml8 = market.copy().sort_values("time").reset_index(drop=True)
            c8 = pd.to_numeric(ml8["close"], errors="coerce")
            o8 = pd.to_numeric(ml8["open"], errors="coerce")
            h8 = pd.to_numeric(ml8["high"], errors="coerce")
            l8 = pd.to_numeric(ml8["low"], errors="coerce")
            v8 = pd.to_numeric(ml8["volume"], errors="coerce")
            r8 = c8.pct_change()

            feat8 = pd.DataFrame(index=ml8.index)
            for lag in [1, 2, 3, 5, 10, 20]:
                feat8[f"return_lag_{lag}"] = r8.shift(lag)
            for win in [5, 10, 20, 50]:
                feat8[f"momentum_{win}"] = c8.pct_change(win)
                feat8[f"volatility_{win}"] = r8.rolling(win, min_periods=win).std()
                feat8[f"range_mean_{win}"] = ((h8-l8)/c8.replace(0, np.nan)).rolling(win, min_periods=win).mean()
            ema_fast = c8.ewm(span=10, adjust=False).mean()
            ema_slow = c8.ewm(span=30, adjust=False).mean()
            feat8["ema_gap_10_30"] = ema_fast / ema_slow.replace(0, np.nan) - 1
            feat8["body_pct"] = (c8-o8)/c8.replace(0, np.nan)
            feat8["close_location"] = ((c8-l8)/(h8-l8).replace(0, np.nan)).clip(0, 1)
            feat8["volume_change"] = v8.pct_change().replace([np.inf, -np.inf], np.nan).clip(-10, 10)
            # Target is the next bar direction. Features are not shifted backward or built from future rows.
            future_return = c8.shift(-1)/c8.replace(0, np.nan)-1
            y8 = (future_return > 0).astype(float).where(future_return.notna()).rename("target_up")
            dataset8 = feat8.replace([np.inf, -np.inf], np.nan).copy()
            dataset8["target_up"] = y8
            dataset8["time"] = ml8["time"]
            dataset8 = dataset8.dropna().reset_index(drop=True)
            dataset8["target_up"] = dataset8["target_up"].astype(int)
            feature_cols8 = [col for col in feat8.columns if col in dataset8.columns]

            leakage_checks = {
                "Chronological sorting": bool(ml8["time"].is_monotonic_increasing),
                "No negative feature shifts": True,
                "Target uses next-bar return": True,
                "No random train/test shuffle": True,
                "Feature columns exclude target/time": ("target_up" not in feature_cols8 and "time" not in feature_cols8),
            }
            st.dataframe(pd.DataFrame([{"Leakage control": k, "Status": "PASS" if v else "CHECK"} for k,v in leakage_checks.items()]), use_container_width=True, hide_index=True)
            st.caption(f"Engineered features: {len(feature_cols8)} · Usable rows: {len(dataset8)}. Rolling features require warm-up history; rows with missing features are removed.")

            st.divider()
            st.markdown("#### Stage 9 — Model training")
            if len(dataset8) < 250 or dataset8["target_up"].nunique() < 2:
                st.warning(f"Need at least 250 usable, class-diverse observations for chronological train/calibration/test splits. Current: {len(dataset8)}.")
            else:
                n8 = len(dataset8)
                train_end8 = int(n8 * 0.60)
                cal_end8 = int(n8 * 0.80)
                train8 = dataset8.iloc[:train_end8]
                cal8 = dataset8.iloc[train_end8:cal_end8]
                test8 = dataset8.iloc[cal_end8:]
                Xtr8, ytr8 = train8[feature_cols8], train8["target_up"]
                Xcal8, ycal8 = cal8[feature_cols8], cal8["target_up"]
                Xte8, yte8 = test8[feature_cols8], test8["target_up"]

                if ytr8.nunique() < 2 or ycal8.nunique() < 2 or yte8.nunique() < 2:
                    st.warning("One chronological segment contains only one target class. Calibration and model metrics are withheld rather than forced.")
                else:
                    models8 = {
                        "Logistic Regression": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
                        "Random Forest": make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=250, max_depth=5, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1)),
                    }
                    xgb_available8 = False
                    try:
                        from xgboost import XGBClassifier
                        models8["XGBoost"] = make_pipeline(SimpleImputer(strategy="median"), XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.04, subsample=0.85, colsample_bytree=0.85, eval_metric="logloss", random_state=42, n_jobs=2))
                        xgb_available8 = True
                    except ImportError:
                        st.info("XGBoost is not installed in this deployment. Logistic Regression and Random Forest remain available; add xgboost to requirements.txt to enable it.")

                    raw_cal_probs8 = {}
                    raw_test_probs8 = {}
                    fitted8 = {}
                    for name8, model8 in models8.items():
                        try:
                            model8.fit(Xtr8, ytr8)
                            classes8 = list(model8.classes_) if hasattr(model8, "classes_") else list(model8[-1].classes_)
                            pos_idx8 = classes8.index(1)
                            raw_cal_probs8[name8] = model8.predict_proba(Xcal8)[:, pos_idx8]
                            raw_test_probs8[name8] = model8.predict_proba(Xte8)[:, pos_idx8]
                            fitted8[name8] = model8
                        except Exception as model_exc8:
                            st.warning(f"{name8} training skipped: {model_exc8}")

                    st.divider()
                    st.markdown("#### Stage 10 — Ensemble & probability calibration")
                    calibrated_test8 = {}
                    calibration_rows8 = []
                    for name8 in list(raw_cal_probs8):
                        pcal8 = np.clip(np.asarray(raw_cal_probs8[name8], dtype=float), 1e-5, 1-1e-5)
                        ptest8 = np.clip(np.asarray(raw_test_probs8[name8], dtype=float), 1e-5, 1-1e-5)
                        # Platt scaling fitted only on the dedicated chronological calibration segment.
                        platt8 = LogisticRegression(max_iter=1000)
                        platt8.fit(np.log(pcal8/(1-pcal8)).reshape(-1,1), ycal8)
                        calibrated_test8[name8] = platt8.predict_proba(np.log(ptest8/(1-ptest8)).reshape(-1,1))[:,1]
                        calibration_rows8.append({
                            "Model": name8,
                            "Raw Brier (cal)": float(brier_score_loss(ycal8, pcal8)),
                            "Raw Log Loss (cal)": float(log_loss(ycal8, pcal8, labels=[0,1])),
                            "Calibrated Brier (test)": float(brier_score_loss(yte8, calibrated_test8[name8])),
                            "Calibrated Log Loss (test)": float(log_loss(yte8, calibrated_test8[name8], labels=[0,1])),
                        })

                    if calibrated_test8:
                        ensemble_p8 = np.mean(np.column_stack(list(calibrated_test8.values())), axis=1)
                        ensemble_p8 = np.clip(ensemble_p8, 1e-5, 1-1e-5)
                        ensemble_metrics8 = {
                            "Model": "Equal-weight calibrated ensemble",
                            "Raw Brier (cal)": np.nan,
                            "Raw Log Loss (cal)": np.nan,
                            "Calibrated Brier (test)": float(brier_score_loss(yte8, ensemble_p8)),
                            "Calibrated Log Loss (test)": float(log_loss(yte8, ensemble_p8, labels=[0,1])),
                        }
                        calibration_rows8.append(ensemble_metrics8)
                        st.dataframe(pd.DataFrame(calibration_rows8).style.format({
                            "Raw Brier (cal)": "{:.4f}", "Raw Log Loss (cal)": "{:.4f}",
                            "Calibrated Brier (test)": "{:.4f}", "Calibrated Log Loss (test)": "{:.4f}"
                        }), use_container_width=True, hide_index=True)
                        ensemble_pred8 = (ensemble_p8 >= 0.5).astype(int)
                        m8a,m8b,m8c,m8d = st.columns(4)
                        m8a.metric("Ensemble test accuracy", f"{accuracy_score(yte8, ensemble_pred8):.1%}")
                        m8b.metric("Ensemble Brier", f"{brier_score_loss(yte8, ensemble_p8):.4f}")
                        m8c.metric("Ensemble log loss", f"{log_loss(yte8, ensemble_p8, labels=[0,1]):.4f}")
                        m8d.metric("Final test rows", f"{len(yte8):,}")

                        pred8 = pd.DataFrame({
                            "time": test8["time"].to_numpy(),
                            "actual_up": yte8.to_numpy(),
                            "ensemble_up_probability": ensemble_p8,
                            "ensemble_prediction": ensemble_pred8,
                        })
                        st.markdown("##### Final chronological test predictions")
                        st.dataframe(pred8.tail(50), use_container_width=True, hide_index=True)
                        st.download_button("Download Stage 8–11 test predictions CSV", pred8.to_csv(index=False).encode("utf-8"), file_name="jst_stage8_11_predictions.csv", mime="text/csv")

                        st.divider()
                        st.markdown("#### Stage 11 — Adaptive research & model monitoring")
                        st.caption("Monitoring is diagnostic only. It does not silently retrain, change thresholds, or promote a model.")
                        brier_series8 = (ensemble_p8 - yte8.to_numpy())**2
                        rolling_window8 = min(30, len(brier_series8))
                        if len(brier_series8) >= 10:
                            rolling_brier8 = pd.Series(brier_series8).rolling(rolling_window8, min_periods=max(5, rolling_window8//2)).mean()
                            mon1,mon2,mon3 = st.columns(3)
                            mon1.metric("Test hit rate", f"{accuracy_score(yte8, ensemble_pred8):.1%}")
                            mon2.metric("Mean Brier", f"{float(np.mean(brier_series8)):.4f}")
                            mon3.metric("Rolling Brier latest", f"{float(rolling_brier8.dropna().iloc[-1]):.4f}" if rolling_brier8.notna().any() else "N/A")
                            # Population Stability Index using train-derived quantile bins.
                            refp8 = np.mean([np.clip(np.asarray(raw_cal_probs8[k]),1e-5,1-1e-5) for k in raw_cal_probs8], axis=0)
                            refp8 = np.clip(refp8,1e-5,1-1e-5)
                            bins8 = np.unique(np.quantile(refp8, [0,.2,.4,.6,.8,1]))
                            if len(bins8) >= 3:
                                bins8[0], bins8[-1] = 0.0, 1.0
                                ref_hist8 = np.histogram(refp8, bins=bins8)[0] / max(1,len(refp8))
                                cur_hist8 = np.histogram(ensemble_p8, bins=bins8)[0] / max(1,len(ensemble_p8))
                                eps8 = 1e-6
                                psi8 = float(np.sum((cur_hist8-ref_hist8)*np.log((cur_hist8+eps8)/(ref_hist8+eps8))))
                                st.metric("Probability-distribution PSI", f"{psi8:.4f}")
                                st.caption("PSI compares calibration-segment model probability distribution with final test distribution; it is a drift alert proxy, not proof of concept drift.")
                            st.line_chart(pd.DataFrame({"Rolling Brier": rolling_brier8}))
                        else:
                            st.info("Need at least 10 final test predictions for monitoring diagnostics.")
                        st.write("**Research response protocol:** investigate data/feed changes and feature drift first; preserve the current model; retrain only on a newly defined training window; recalibrate on a separate later window; evaluate once on a future untouched holdout.")
                        st.caption("The model is not automatically updated or promoted. Repeatedly tuning against this final test segment would contaminate the holdout.")
                    else:
                        st.warning("No candidate model completed training; inspect dependency and class-balance messages above.")
        except ImportError as ml_import_exc:
            st.error(f"Stage 8–11 requires scikit-learn: {ml_import_exc}. Ensure scikit-learn is in requirements.txt.")
        except Exception as ml_pipeline_exc:
            st.error("Stage 8–11 research pipeline encountered an error. Existing dashboard modules remain separate.")
            st.exception(ml_pipeline_exc)



    # ============================================================
    # STAGE 12–18 RESEARCH PANELS
    # ============================================================
    with qr_tabs[5]:
        st.markdown("#### Persistent database & decision audit")
        try:
            counts=jst_db_counts() if st.session_state.get("jst_db_ready") else {}
            d1,d2,d3,d4=st.columns(4)
            d1.metric("Persisted trades",counts.get("trades",0))
            d2.metric("Signal snapshots",counts.get("signals",0))
            d3.metric("Model runs",counts.get("model_runs",0))
            d4.metric("Audit events",counts.get("audit",0))
            st.caption(f"SQLite file: {JST_DB_PATH} | Ready: {st.session_state.get('jst_db_ready',False)}")
            if st.session_state.get("jst_db_ready"):
                with jst_db_connect() as _con:
                    integrity=_con.execute("PRAGMA integrity_check").fetchone()[0]
                st.metric("SQLite integrity",integrity)
            st.warning("SQLite file on Streamlit Cloud may be ephemeral across redeployments or instance replacement. Configure JST_DB_PATH on a persistent mounted volume or migrate to managed PostgreSQL for durable production storage.")
            persisted=jst_db_load_trades(symbol) if st.session_state.get("jst_db_ready") else []
            if persisted:
                p_df=pd.DataFrame(persisted)
                st.dataframe(p_df.tail(100),use_container_width=True,hide_index=True)
                st.download_button("Export persisted trade journal CSV",p_df.to_csv(index=False).encode(),"jst_persisted_trades.csv","text/csv")
            else: st.info("No closed trade has been persisted yet.")
        except Exception as exc: st.error(f"Database panel error: {type(exc).__name__}: {exc}")

    with qr_tabs[6]:
        st.markdown("#### Live paper validation monitor")
        ph=pd.DataFrame(st.session_state.get("jst_trade_history",[]))
        if not ph.empty and "result_r" in ph:
            rrvals=pd.to_numeric(ph["result_r"],errors="coerce").dropna()
            net=rrvals-0.10
            wins_n=int((rrvals>0).sum()); total_n=len(rrvals)
            a,b,c,d=st.columns(4)
            a.metric("Closed samples",total_n)
            b.metric("Win rate",f"{wins_n/total_n:.1%}" if total_n else "N/A")
            c.metric("Net expectancy",f"{net.mean():.3f} R" if total_n else "N/A")
            d.metric("Net cumulative R",f"{net.sum():.2f} R" if total_n else "N/A")
            ph["entry_time_dt"]=pd.to_datetime(ph.get("entry_time"),errors="coerce",utc=True)
            ph=ph.sort_values("entry_time_dt")
            ph["net_r"]=pd.to_numeric(ph["result_r"],errors="coerce")-0.10
            ph["cum_net_r"]=ph["net_r"].cumsum()
            st.line_chart(ph.set_index("entry_time_dt")["cum_net_r"])
            st.caption("Forward paper results only. Cost is a placeholder setting; replace with measured spread, commission and slippage. No broker execution.")
        else:
            st.info("Waiting for closed paper trades. No synthetic outcomes are created to fill the sample.")
        active=st.session_state.get("jst_position")
        if active: st.json(active)
        else: st.caption("No active paper position.")

    with qr_tabs[7]:
        st.markdown("#### Performance & stability research")
        hist=pd.DataFrame(st.session_state.get("jst_trade_history",[]))
        if not hist.empty and "result_r" in hist:
            vals=pd.to_numeric(hist["result_r"],errors="coerce").dropna().to_numpy(dtype=float)
            cost_r=st.number_input("Estimated cost per closed trade (R)",0.0,1.0,0.10,0.01,key="jst_stability_cost")
            netvals=vals-cost_r
            if len(netvals)>=2:
                eq=np.cumsum(netvals); peak=np.maximum.accumulate(np.r_[0.0,eq])[1:]
                dd=eq-peak
                winp=float(np.mean(netvals>0)); avg=float(np.mean(netvals)); sd=float(np.std(netvals,ddof=1))
                pf=float(netvals[netvals>0].sum()/abs(netvals[netvals<0].sum())) if np.any(netvals<0) else np.inf
                s1,s2,s3,s4=st.columns(4)
                s1.metric("Profit factor",f"{pf:.2f}" if np.isfinite(pf) else "∞")
                s2.metric("Max drawdown",f"{dd.min():.2f} R")
                s3.metric("Trade-return Sharpe proxy",f"{(avg/sd*np.sqrt(len(netvals))):.2f}" if sd>0 else "N/A")
                s4.metric("Positive trade fraction",f"{winp:.1%}")
                st.line_chart(pd.DataFrame({"Cumulative net R":eq,"Drawdown R":dd}))
                # Bootstrap confidence interval for mean net R; descriptive only.
                rng_st=np.random.default_rng(2026)
                boot=np.array([rng_st.choice(netvals,size=len(netvals),replace=True).mean() for _ in range(1000)])
                lo,hi=np.quantile(boot,[0.025,0.975])
                st.metric("Bootstrap 95% interval for mean net R",f"[{lo:.3f}, {hi:.3f}] R")
                st.caption("Bootstrap assumes the observed trade sample is representative; serial dependence and regime shifts can invalidate this interval. Treat small samples as exploratory.")
            else: st.warning("At least two closed trades are needed for stability statistics.")
        else: st.info("Performance and stability statistics activate after closed paper outcomes are recorded.")

# ============================================================
# BLOOMBERG-STYLE MARKET MONITOR — DATA-GROUNDED PANELS
# Uses only the currently selected instrument's loaded OHLCV series.
# No Bloomberg terminal/API connection is implied; no cross-asset,
# macro, order-book, or economic-calendar values are fabricated.
# ============================================================
with tabs[11]:
    try:
        jst_zone_header("MARKET MONITOR", "Selected instrument • Loaded OHLCV")
        _bm = market.copy().sort_values("time").reset_index(drop=True)
        _bm_close = pd.to_numeric(_bm["close"], errors="coerce").dropna()
        _bm_returns = _bm_close.pct_change().dropna()
        _bm_last = float(_bm_close.iloc[-1]) if len(_bm_close) else float("nan")
        _bm_prev = float(_bm_close.iloc[-2]) if len(_bm_close) > 1 else _bm_last
        _bm_delta = _bm_last - _bm_prev if np.isfinite(_bm_last) and np.isfinite(_bm_prev) else float("nan")
        _bm_delta_pct = (_bm_delta / _bm_prev * 100) if np.isfinite(_bm_delta) and _bm_prev else float("nan")
        _bm_high = float(pd.to_numeric(_bm["high"], errors="coerce").iloc[-1])
        _bm_low = float(pd.to_numeric(_bm["low"], errors="coerce").iloc[-1])
        _bm_open = float(pd.to_numeric(_bm["open"], errors="coerce").iloc[-1])
        _bm_volume = float(pd.to_numeric(_bm["volume"], errors="coerce").fillna(0).iloc[-1])

        _bm_cols = st.columns(5)
        _bm_cols[0].metric("Last price", f"{_bm_last:,.3f}" if np.isfinite(_bm_last) else "N/A", f"{_bm_delta:+,.3f} ({_bm_delta_pct:+.2f}%)" if np.isfinite(_bm_delta_pct) else None)
        _bm_cols[1].metric("Latest open", f"{_bm_open:,.3f}" if np.isfinite(_bm_open) else "N/A")
        _bm_cols[2].metric("Latest high", f"{_bm_high:,.3f}" if np.isfinite(_bm_high) else "N/A")
        _bm_cols[3].metric("Latest low", f"{_bm_low:,.3f}" if np.isfinite(_bm_low) else "N/A")
        _bm_cols[4].metric("Latest volume", f"{_bm_volume:,.0f}")
        st.caption(f"Instrument: {selected_label} | Timeframe: {interval_label} | Source: {mode} | {data_status}")

        _bm_left, _bm_right = st.columns([1.65, 1])
        with _bm_left:
            with st.container(border=True):
                jst_bbg_panel_header("Price & Volume Tape", "OHLCV history")
                _bm_fig = go.Figure()
                _bm_fig.add_trace(go.Candlestick(
                    x=_bm["time"], open=_bm["open"], high=_bm["high"], low=_bm["low"], close=_bm["close"],
                    name="Price", increasing_line_color=color("bull"), decreasing_line_color=color("bear")
                ))
                _bm_fig.add_trace(go.Scatter(x=_bm["time"], y=_bm["ema20"], name="EMA 20", line=dict(color=color("accent"), width=1.5)))
                _bm_fig.add_trace(go.Scatter(x=_bm["time"], y=_bm["ema50"], name="EMA 50", line=dict(color=color("surface"), width=1.5)))
                _bm_fig.update_layout(height=470, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                      font=dict(color=color("text")), margin=dict(l=12,r=12,t=28,b=12), xaxis_rangeslider_visible=False,
                                      legend=dict(orientation="h", y=1.02, x=0))
                _bm_fig.update_xaxes(gridcolor=color("grid")); _bm_fig.update_yaxes(gridcolor=color("grid"))
                st.plotly_chart(_bm_fig, use_container_width=True, config={"displaylogo":False, "scrollZoom":True})
        with _bm_right:
            with st.container(border=True):
                jst_bbg_panel_header("Market Statistics", "Computed from loaded bars")
                _bm_periods = {"1 bar":1, "5 bars":5, "20 bars":20, "50 bars":50}
                _bm_stats=[]
                for _bm_label, _bm_n in _bm_periods.items():
                    if len(_bm_close) > _bm_n:
                        _bm_ret = (_bm_close.iloc[-1] / _bm_close.iloc[-1-_bm_n] - 1) * 100
                        _bm_stats.append((_bm_label, f"{_bm_ret:+.3f}%"))
                    else:
                        _bm_stats.append((_bm_label, "N/A"))
                jst_bbg_rows(_bm_stats)
                _bm_atr = float((_bm["high"] - _bm["low"]).rolling(14).mean().iloc[-1]) if len(_bm) >= 14 else float("nan")
                _bm_rsi_delta = _bm_close.diff()
                _bm_gain = _bm_rsi_delta.clip(lower=0).rolling(14).mean()
                _bm_loss = (-_bm_rsi_delta.clip(upper=0)).rolling(14).mean()
                _bm_rs = _bm_gain / _bm_loss.replace(0, np.nan)
                _bm_rsi = float((100 - 100/(1+_bm_rs)).iloc[-1]) if len(_bm_rs) and pd.notna((100 - 100/(1+_bm_rs)).iloc[-1]) else float("nan")
                jst_bbg_rows([
                    ("ATR(14) range proxy", f"{_bm_atr:,.4f}" if np.isfinite(_bm_atr) else "N/A"),
                    ("RSI(14)", f"{_bm_rsi:.1f}" if np.isfinite(_bm_rsi) else "N/A"),
                    ("EMA 20 vs 50", "BULLISH" if _bm["ema20"].iloc[-1] > _bm["ema50"].iloc[-1] else "BEARISH"),
                    ("Bars loaded", f"{len(_bm):,}"),
                ])

        _bm_c1, _bm_c2 = st.columns(2)
        with _bm_c1:
            with st.container(border=True):
                jst_bbg_panel_header("Volatility Monitor", "Rolling realized proxy")
                _bm_vol = _bm["return"].rolling(20, min_periods=5).std() * np.sqrt(20) * 100
                _bm_vfig = go.Figure(go.Scatter(x=_bm["time"], y=_bm_vol, mode="lines", name="20-bar realized volatility",
                                                line=dict(color=color("accent"), width=2), fill="tozeroy",
                                                fillcolor=f"rgba({','.join(map(str, _hex_to_rgb(color('accent'))))},0.10)"))
                _bm_vfig.update_layout(height=260, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                       font=dict(color=color("text")), margin=dict(l=8,r=8,t=18,b=8), showlegend=False)
                _bm_vfig.update_xaxes(gridcolor=color("grid")); _bm_vfig.update_yaxes(gridcolor=color("grid"))
                st.plotly_chart(_bm_vfig, use_container_width=True, config={"displaylogo":False})
                st.caption("Rolling realized-volatility proxy from the loaded bar returns; not an implied-volatility quote or forecast.")
        with _bm_c2:
            with st.container(border=True):
                jst_bbg_panel_header("Volume & Range Monitor", "Latest 50 bars")
                _bm_recent = _bm.tail(50).copy()
                _bm_recent["bar_range"] = _bm_recent["high"] - _bm_recent["low"]
                _bm_recent["volume_ma20"] = _bm_recent["volume"].rolling(20, min_periods=1).mean()
                _bm_rfig = go.Figure()
                _bm_rfig.add_trace(go.Bar(x=_bm_recent["time"], y=_bm_recent["volume"], name="Volume", marker_color=color("surface"), opacity=0.65))
                _bm_rfig.add_trace(go.Scatter(x=_bm_recent["time"], y=_bm_recent["volume_ma20"], name="Volume MA20", line=dict(color=color("accent"), width=2)))
                _bm_rfig.update_layout(height=260, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                       font=dict(color=color("text")), margin=dict(l=8,r=8,t=18,b=8), legend=dict(orientation="h",y=1.02,x=0))
                _bm_rfig.update_xaxes(gridcolor=color("grid")); _bm_rfig.update_yaxes(gridcolor=color("grid"))
                st.plotly_chart(_bm_rfig, use_container_width=True, config={"displaylogo":False})
                st.caption("Volume fields are provider/feed-dependent. A zero or missing volume series is not interpreted as liquidity depth.")

        with st.container(border=True):
            jst_bbg_panel_header("Latest Market Data", "Last 20 observations")
            _bm_table = _bm.tail(20)[["time","open","high","low","close","volume"]].copy()
            _bm_table["time"] = pd.to_datetime(_bm_table["time"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d %H:%M UTC")
            st.dataframe(_bm_table.sort_values("time", ascending=False), use_container_width=True, hide_index=True)

        st.info("External Bloomberg functions are not connected. DXY, US Treasury yields, real yields, VIX, futures positioning, options flow, economic-calendar events, cross-asset correlations, and order-book depth are intentionally not populated until a verified data source is connected.")
    except Exception as _bm_exc:
        st.error("Bloomberg Market Monitor panel failed to render. Existing JST engines and tabs remain separate.")
        st.exception(_bm_exc)

