\# System Architecture



\## 1. Input Module

\- Read `tickers.txt` (local file).

\- Load environment variables from `.env`.



\## 2. Analysis Engine (Python)

\- \*\*Data Fetch:\*\* `yf.download(interval="15m", period="1d")`.

\- \*\*VWAP Calculation:\*\* $\\sum(\\text{Typical Price} \\times \\text{Volume}) / \\sum(\\text{Volume})$.

\- \*\*Visual Generator:\*\* - Filter for the last 20 candles of 15m data.

&nbsp; - Render a chart where `show\_nontrading=False` and `wick=None` (No wicks).

&nbsp; - Save as `current\_chart.png`.



\## 3. LLM Integration (z.ai)

\- \*\*Endpoint:\*\* `https://api.z.ai/api/paas/v4/chat/completions`.

\- \*\*API Call 1 (Vision):\*\* Send `current\_chart.png` to `glm-4.6v`.

\- \*\*API Call 2 (Logic):\*\* Send GLM-4.7 a JSON packet containing:

&nbsp; - {Price, VWAP, VolumeDelta, Vision\_Summary}.

&nbsp; - Request: "Is this a clear buy? Reply 'SIGNAL: BUY' or 'HOLD'."



\## 4. Output Module

\- Simple HTTP POST to `ntfy.sh/\[TOPIC]`.

