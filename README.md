JUMP STREET TRADING — Streamlit Prototype
Included
- Candlestick chart with Entry / Stop Loss / Take Profit overlays
- Risk/reward presets and custom reward multiple
- Quant Math 3D data-derived surface
- 3D illustrative running P&L path for simulated paper position
- Color pickers for background, panels, candles, 3D surface, grid, TP/SL
- CSV upload and optional Twelve Data API source
- Setup journal CSV export
- No broker order execution
Deploy to Streamlit Community Cloud
1. Upload app.py, requirements.txt, and this README to a GitHub repository.
2. In Streamlit Community Cloud, deploy the app with main file path app.py.
3. Optional Twelve Data feed: add this to Streamlit app Secrets:
   TWELVE_DATA_API_KEY = "your_key_here"
4. Select Twelve Data API in the sidebar.
CSV format
Recommended columns: time, open, high, low, close, volume.
A close or price column is required. If OHLC columns are missing, the app uses close as a placeholder for those columns, so proper OHLC is strongly preferred.
Important limitations
- Demo data is synthetic, not live market data.
- Running P&L is simulated and based on configured contract size and lot. It is not a broker's live P&L.
- Auto-refresh reruns the app; it does not guarantee a new market tick. Refresh interval and API quotas depend on data source.
- Risk USD calculations are estimates using contract size × lot and price distance. Confirm broker-specific tick value, contract specifications, quote currency, spread, commission, and slippage.
- The 3D quant surface is a visualization of normalized price, volatility, and momentum; it is not a trained predictive model.
- Walk-forward, out-of-sample, full expected-value backtesting, and comprehensive regime/stress testing are not implemented in this starter build.
- Signal-only: no orders are placed.
