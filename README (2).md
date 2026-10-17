# JUMP STREET TRADING

Streamlit quantitative research dashboard. Signal-only; no broker order execution and no Telegram integration.

## Existing dashboard
- Multi-asset watchlist with clearly labelled synthetic demo fallback.
- Candlestick chart, EMA overlays, risk/reward planner and global color customization.
- Quant Math and Advanced Quant 3D research modules.
- Machine Learning Lab with chronological holdout diagnostics.
- Rule-based research signal candidates.

## Macro & News Intelligence
- DXY request via Twelve Data if your account/provider supports the symbol.
- US 2Y nominal yield (DGS2), US 10Y nominal yield (DGS10), and US 10Y real yield (DFII10) via FRED. These are daily series, not tick-level M1 data.
- Cross-asset correlation heatmap and rolling correlation from uploaded synchronized multi-asset price CSV.
- Medium/high-impact calendar CSV and dashboard alert windows (T-60, T-15, release).

## Optional Streamlit Secrets
```toml
TWELVE_DATA_API_KEY = "YOUR_TWELVE_DATA_KEY"
FRED_API_KEY = "YOUR_FRED_API_KEY"
```

## Economic calendar CSV
Required: `event,datetime_utc,currency,impact`
Optional: `forecast,previous,actual,country,source`
Use ISO-8601 UTC timestamps, e.g. `2026-10-20T12:30:00Z`. Upload in the Macro & News Intelligence tab. A global calendar provider has not been automatically connected in this build because provider APIs and schemas differ.

## Correlation CSV
Wide format: one timestamp column and two or more synchronized numeric price columns, e.g. `time,XAUUSD,DXY,US10Y,EURUSD,BTCUSD`. Use real historical data and do not mix synthetic data with live data.

## Limitations
- Synthetic demo data is not live.
- DXY requires provider support.
- FRED yield series are daily.
- Correlation is calculated only from uploaded multi-asset data.
- Calendar alert windows update when Streamlit reruns; optional dashboard auto-refresh can rerun while the page is open. This is not a persistent background scheduler and is not guaranteed while the app is closed.
- No Telegram, broker execution, or guaranteed trading predictions.
