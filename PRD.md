\# PRD: Local Body-Only Stock Monitor



\## Goal

A local Python script that monitors stocks from `tickers.txt` and sends "Strong Buy" alerts to my phone via `ntfy.sh`.



\## Technical Constraints

\- \*\*Timeframe:\*\* 15-minute candles.

\- \*\*Execution:\*\* Runs every 1 hour (at :02 past the hour) via local task scheduler.

\- \*\*Wickless Analysis:\*\* Candlestick patterns must only consider the REAL BODY (Open to Close). Ignore shadows/wicks.

\- \*\*Logic Sequence (The "Triad"):\*\*

&nbsp; 1. \*\*VWAP:\*\* Current Price must be > Daily Anchored VWAP.

&nbsp; 2. \*\*Order Flow:\*\* Volume of the last closed candle must be > 2x the 10-period average.

&nbsp; 3. \*\*Visual Pattern:\*\* GLM-4.6V must identify a 'Bullish Engulfing' or 'Hammer' based on body size/position.



\## Tech Stack

\- \*\*Data:\*\* `yfinance`

\- \*\*Charting:\*\* `mplfinance` (configured to hide wicks).

\- \*\*Brain:\*\* z.ai API (GLM-4.7 for reasoning, GLM-4.6V for vision).

\- \*\*Alerts:\*\* `ntfy.sh` (POST request).

