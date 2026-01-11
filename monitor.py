"""
Local Body-Only Stock Monitor
Monitors stocks and sends "Strong Buy" alerts via ntfy.sh

Technical Constraints:
- 15-minute candles
- Wickless analysis (body-only patterns)
- Triad Logic: VWAP + Order Flow + Visual Pattern
"""

import os
import base64
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List

# Set matplotlib backend for headless environments (GitHub Actions, servers)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import yfinance as yf
import pandas as pd
import numpy as np
import mplfinance as mpf
import requests
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Config:
    """Configuration and environment variables."""

    ZAI_API_URL = "https://api.z.ai/api/paas/v4/chat/completions"
    NTFY_BASE_URL = "https://ntfy.sh"

    def __init__(self):
        # Try to load .env file (will be ignored if not found)
        load_dotenv()

        # Read environment variables (from .env file OR system env vars)
        self.zai_api_key = os.getenv("ZAI_API_KEY")
        self.ntfy_topic = os.getenv("NTFY_TOPIC")
        self.tickers_file = os.getenv("TICKERS_FILE", "tickers.txt")

        # Model names - configurable via env vars (defaults to glm-4-plus which supports multimodal)
        # Check if env vars are set, otherwise use default
        env_vision = os.getenv("MODEL_VISION")
        env_logic = os.getenv("MODEL_LOGIC")

        self.model_vision = env_vision if env_vision else "glm-4-plus"
        self.model_logic = env_logic if env_logic else "glm-4-plus"

        # Debug logging
        logger.info(f"Config: MODEL_VISION={self.model_vision}, MODEL_LOGIC={self.model_logic}")

        if not self.zai_api_key:
            raise ValueError("ZAI_API_KEY not found in environment variables")
        if not self.ntfy_topic:
            raise ValueError("NTFY_TOPIC not found in environment variables")


class InputModule:
    """Handles input from tickers file and environment."""

    def __init__(self, config: Config):
        self.config = config
        self.tickers: List[str] = []

    def load_tickers(self) -> List[str]:
        """Read tickers from tickers.txt file."""
        try:
            tickers_path = Path(self.config.tickers_file)
            if not tickers_path.exists():
                logger.error(f"Tickers file not found: {self.config.tickers_file}")
                return []

            with open(tickers_path, 'r') as f:
                self.tickers = [
                    line.strip().upper()
                    for line in f
                    if line.strip() and not line.startswith('#')
                ]

            logger.info(f"Loaded {len(self.tickers)} tickers: {self.tickers}")
            return self.tickers
        except Exception as e:
            logger.error(f"Error loading tickers: {e}")
            return []


class AnalysisEngine:
    """Core analysis engine for stock data."""

    def __init__(self, config: Config):
        self.config = config
        self.data_cache: Dict[str, pd.DataFrame] = {}

    def fetch_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Fetch 15-minute candle data for the past day.

        Args:
            ticker: Stock ticker symbol

        Returns:
            DataFrame with OHLCV data or None if fetch fails
        """
        try:
            logger.info(f"Fetching data for {ticker}...")
            ticker_obj = yf.Ticker(ticker)

            # Fetch 15-minute data for 1 day
            data = ticker_obj.history(period="1d", interval="15m", prepost=False)

            if data.empty:
                logger.warning(f"No data returned for {ticker}")
                return None

            self.data_cache[ticker] = data
            logger.info(f"Fetched {len(data)} candles for {ticker}")
            return data

        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return None

    def calculate_vwap(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Daily Anchored VWAP.

        VWAP = Sum(Typical Price * Volume) / Sum(Volume)
        Typical Price = (High + Low + Close) / 3
        """
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        vwap = (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()
        return vwap

    def check_volume_condition(self, data: pd.DataFrame) -> tuple[bool, float, float]:
        """
        Check if volume of last candle > 2x the 10-period average.

        Returns:
            Tuple of (condition_met, last_volume, avg_volume)
        """
        if len(data) < 11:
            return False, 0.0, 0.0

        last_volume = data['Volume'].iloc[-1]
        avg_volume = data['Volume'].iloc[-11:-1].mean()

        condition_met = last_volume > (2 * avg_volume)
        return condition_met, last_volume, avg_volume

    def check_vwap_condition(self, data: pd.DataFrame) -> tuple[bool, float, float]:
        """
        Check if current price is above VWAP.

        Returns:
            Tuple of (condition_met, current_price, vwap)
        """
        vwap_series = self.calculate_vwap(data)
        current_price = data['Close'].iloc[-1]
        current_vwap = vwap_series.iloc[-1]

        condition_met = current_price > current_vwap
        return condition_met, current_price, current_vwap

    def generate_chart(self, ticker: str, data: pd.DataFrame) -> Optional[str]:
        """
        Generate a wickless candlestick chart.

        Args:
            ticker: Stock ticker symbol
            data: OHLCV DataFrame

        Returns:
            Path to saved chart image or None if generation fails
        """
        try:
            # Filter for last 20 candles
            chart_data = data.tail(20).copy()

            # Ensure data has the proper format for mplfinance
            chart_data.index.name = 'Date'

            # Chart filename
            chart_path = f"{ticker.lower()}_chart.png"

            # Custom matplotlib style to hide wicks
            mc = mpf.make_marketcolors(
                up='g', down='r',
                edge='inherit',
                wick={'up': 'none', 'down': 'none'},  # Hide wicks
                volume='in'
            )

            s = mpf.make_mpf_style(marketcolors=mc, gridstyle='--', gridcolor='gray')

            # Configure plot to hide wicks
            kwargs = dict(
                type='candle',
                style=s,
                title=f'{ticker} - 15Min Chart (Wickless)',
                ylabel='Price',
                volume=True,
                savefig=chart_path,
                figscale=1.2,
                warn_too_much_data=1000,
                # Wick is hidden via marketcolors above
            )

            mpf.plot(chart_data, **kwargs)

            logger.info(f"Chart saved: {chart_path}")
            return chart_path

        except Exception as e:
            logger.error(f"Error generating chart for {ticker}: {e}")
            return None

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def generate_annotated_chart(
        self,
        ticker: str,
        data: pd.DataFrame,
        pattern_info: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Generate chart with pattern annotation using matplotlib.

        Args:
            ticker: Stock symbol
            data: OHLCV DataFrame
            pattern_info: Optional dict with 'pattern_detected', 'signal', 'reasoning'

        Returns:
            Path to annotated chart image
        """
        try:
            # Get last 20 candles
            chart_data = data.tail(20).copy()
            chart_data.index.name = 'Date'

            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(
                2, 1,
                figsize=(12, 8),
                gridspec_kw={'height_ratios': [3, 1]},
                sharex=True
            )

            # Plot candlestick chart
            self._plot_candles(ax1, chart_data)

            # Plot volume
            self._plot_volume(ax2, chart_data)

            # Add pattern annotation if provided
            if pattern_info:
                self._add_pattern_annotation(ax1, chart_data, pattern_info, ticker)

            # Add title and labels
            fig.suptitle(
                f'{ticker} - 15Min Chart (Wickless Analysis)',
                fontsize=14,
                fontweight='bold'
            )
            ax1.set_ylabel('Price ($)', fontsize=10)
            ax2.set_ylabel('Volume', fontsize=10)
            ax2.set_xlabel('Time', fontsize=10)

            # Adjust layout and save
            plt.tight_layout()
            chart_path = f"{ticker.lower()}_annotated.png"
            plt.savefig(chart_path, dpi=100, bbox_inches='tight')
            plt.close()

            logger.info(f"Annotated chart saved: {chart_path}")
            return chart_path

        except Exception as e:
            logger.error(f"Error generating annotated chart: {e}")
            # Fall back to regular chart
            return self.generate_chart(ticker, data)

    def _plot_candles(self, ax, data: pd.DataFrame) -> None:
        """
        Plot wickless candlestick chart on given axes.

        Args:
            ax: Matplotlib axes
            data: OHLCV DataFrame
        """
        # Calculate colors based on open/close
        colors = np.where(data['Close'] >= data['Open'], 'g', 'r')

        # Plot body rectangles (wickless)
        for i, (idx, row) in enumerate(data.iterrows()):
            open_price = row['Open']
            close_price = row['Close']
            color = colors[i]

            # Body height (use min height of 0.01 for flat candles)
            body_height = max(abs(close_price - open_price), 0.01)

            # Draw rectangle for body
            rect = mpatches.Rectangle(
                (i - 0.35, min(open_price, close_price)),
                0.7,  # width
                body_height,
                facecolor=color,
                edgecolor=color,
                alpha=0.8
            )
            ax.add_patch(rect)

        # Set x-axis limits and labels
        ax.set_xlim(-0.5, len(data) - 0.5)
        ax.set_xticks(range(0, len(data), max(1, len(data) // 10)))
        ax.set_xticklabels(
            [idx.strftime('%H:%M') for idx in data.index[::max(1, len(data) // 10)]],
            rotation=45
        )

        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylabel('Price ($)', fontsize=10)

    def _plot_volume(self, ax, data: pd.DataFrame) -> None:
        """
        Plot volume bars on given axes.

        Args:
            ax: Matplotlib axes
            data: OHLCV DataFrame
        """
        # Calculate colors based on price movement
        colors = np.where(data['Close'] >= data['Open'], 'g', 'r')

        # Plot volume bars
        ax.bar(range(len(data)), data['Volume'], color=colors, alpha=0.6, width=0.7)

        # Format volume labels
        ax.set_xlim(-0.5, len(data) - 0.5)
        ax.set_ylabel('Volume', fontsize=10)

        # Format y-axis to show volumes in millions/thousands
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.1f}K')
        )

        ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    def _add_pattern_annotation(
        self,
        ax,
        data: pd.DataFrame,
        pattern_info: Dict[str, Any],
        ticker: str
    ) -> None:
        """
        Add annotation box for detected pattern.

        Args:
            ax: Matplotlib axes
            data: OHLCV DataFrame
            pattern_info: Dict with pattern information
            ticker: Stock symbol
        """
        try:
            signal = pattern_info.get('signal', 'NEUTRAL')
            pattern = pattern_info.get('pattern_detected', 'None')

            # Position annotation at the last candle
            x_pos = len(data) - 1
            last_candle = data.iloc[-1]
            y_pos = last_candle['High']

            # Build annotation text
            annotation_text = (
                f"Pattern: {pattern}\n"
                f"Signal: {signal}\n"
            )

            # Add reasoning if available (truncate if too long)
            reasoning = pattern_info.get('reasoning', '')
            if reasoning:
                reasoning_short = reasoning[:60] + '...' if len(reasoning) > 60 else reasoning
                annotation_text += f"Note: {reasoning_short}"

            # Choose annotation color based on signal
            if signal == 'BULLISH':
                box_color = 'lightgreen'
                arrow_color = 'green'
            elif signal == 'BEARISH':
                box_color = 'lightcoral'
                arrow_color = 'red'
            else:
                box_color = 'lightyellow'
                arrow_color = 'orange'

            # Add annotation with arrow
            ax.annotate(
                annotation_text,
                xy=(x_pos, last_candle['Close']),
                xytext=(max(0, x_pos - 4), y_pos * 1.02),
                fontsize=8,
                bbox=dict(
                    boxstyle='round,pad=0.5',
                    facecolor=box_color,
                    alpha=0.8,
                    edgecolor=arrow_color,
                    linewidth=2
                ),
                arrowprops=dict(
                    arrowstyle='->',
                    color=arrow_color,
                    lw=2,
                    connectionstyle='arc3,rad=0.3'
                ),
                verticalalignment='top'
            )

        except Exception as e:
            logger.warning(f"Error adding annotation: {e}")


class LLMApiClient:
    """Client for z.ai API (vision and logic models)."""

    def __init__(self, config: Config):
        self.config = config
        self.api_url = config.ZAI_API_URL
        self.api_key = config.zai_api_key
        self.model_vision = config.model_vision
        self.model_logic = config.model_logic

    def _make_request(self, messages: List[Dict[str, Any]], model: str) -> Optional[str]:
        """
        Make a request to z.ai API.

        Args:
            messages: List of message dictionaries
            model: Model to use (glm-4.7 for thinking, glm-4v for vision)

        Returns:
            Response content or None if request fails
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 1000
        }

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                logger.info(f"{model} response received")
                return content
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            logger.error(f"Error making API request: {e}")
            return None

    def analyze_chart_vision(self, ticker: str, image_base64: str) -> Optional[str]:
        """
        Send chart to GLM-4-Plus for visual pattern analysis.

        Analyzes for Bullish Engulfing or Hammer patterns (body-only).
        Uses Zhipu AI's multimodal format.
        """
        # For Zhipu AI API, use their specific multimodal format
        messages = [
            {
                "role": "user",
                "content": f"""Analyze this {ticker} 15-minute candlestick chart for BODY-ONLY bullish patterns.

IMPORTANT: This is a wickless chart - focus ONLY on the real body (open to close relationship).

Look for these bullish patterns:
1. Bullish Engulfing: A smaller bearish/red body followed by a larger bullish/green body that completely engulfs the previous body
2. Hammer (bullish reversal): Small body at top with previous downward movement, followed by reversal

For the LAST completed candle, analyze:
- Body size (close relative to open)
- Body position relative to prior candles
- Body color pattern sequence

Reply in this exact JSON format:
{{
    "pattern_detected": "Bullish Engulfing" or "Hammer" or "None",
    "confidence": "high" or "medium" or "low",
    "reasoning": "brief explanation based on body analysis",
    "signal": "BULLISH" or "NEUTRAL" or "BEARISH"
}}

Only return the JSON, nothing else."""
            }
        ]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # For Zhipu AI, send the image as a separate parameter in the payload
        payload = {
            "model": self.model_vision,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 1000,
            # Add image directly for Zhipu AI multimodal format
            "image": f"data:image/png;base64,{image_base64}"
        }

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                logger.info(f"{self.model_vision} vision response received")
                return content
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            logger.error(f"Error making vision API request: {e}")
            return None

    def analyze_logic(self, ticker: str, analysis_data: Dict[str, Any]) -> Optional[str]:
        """
        Send quantitative data to logic model for final decision.
        """
        messages = [
            {
                "role": "system",
                "content": """You are a conservative trading signal analyzer. Your task is to evaluate if a stock presents a "Strong Buy" opportunity based on the Triad Logic:

TRIAD EVALUATION:
1. VWAP Condition: Price > VWAP (bullish momentum)
2. Volume Condition: Last Volume > 2x 10-period Average (strong order flow)
3. Visual Pattern: Bullish body-only pattern detected by vision model

RESPONSE FORMAT:
Reply with EXACTLY one of these:
- "SIGNAL: BUY" - if ALL THREE conditions are met
- "HOLD" - if any condition is not met

Do not provide additional explanation. Just the signal."""
            },
            {
                "role": "user",
                "content": f"""Analyze {ticker} based on the following data:

{{
    "ticker": "{ticker}",
    "current_price": {analysis_data.get('current_price', 0)},
    "vwap": {analysis_data.get('vwap', 0)},
    "vwap_condition_met": {analysis_data.get('vwap_condition_met', False)},
    "last_volume": {analysis_data.get('last_volume', 0)},
    "avg_volume_10": {analysis_data.get('avg_volume', 0)},
    "volume_condition_met": {analysis_data.get('volume_condition_met', False)},
    "vision_analysis": {analysis_data.get('vision_analysis', {})}
}}

Evaluate all three conditions and provide your signal."""
            }
        ]

        return self._make_request(messages, model=self.model_logic)


class AlertModule:
    """Handles sending alerts via ntfy.sh."""

    def __init__(self, config: Config):
        self.config = config
        self.base_url = config.NTFY_BASE_URL
        self.topic = config.ntfy_topic

    def send_alert(self, ticker: str, message: str, priority: str = "high") -> bool:
        """
        Send alert via ntfy.sh.

        Args:
            ticker: Stock ticker symbol
            message: Alert message
            priority: Message priority (default: high)

        Returns:
            True if alert sent successfully, False otherwise
        """
        url = f"{self.base_url}/{self.topic}"

        headers = {
            "Title": f"{ticker} Strong Buy Signal",
            "Priority": priority,
            "Tags": "rocket,chart_up"
        }

        try:
            response = requests.post(url, data=message.encode(), headers=headers, timeout=10)

            if response.status_code in [200, 201]:
                logger.info(f"Alert sent for {ticker}: {message}")
                return True
            else:
                logger.error(f"Failed to send alert: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Error sending alert: {e}")
            return False


class StockMonitor:
    """Main orchestrator for stock monitoring."""

    def __init__(self):
        self.config = Config()
        self.input_module = InputModule(self.config)
        self.analysis_engine = AnalysisEngine(self.config)
        self.llm_client = LLMApiClient(self.config)
        self.alert_module = AlertModule(self.config)

    def analyze_ticker(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Perform complete analysis on a single ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Analysis results dictionary or None if analysis fails
        """
        chart_path = None  # Track chart for cleanup (vision chart only)
        annotated_path = None  # Annotated chart is kept for artifacts

        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Starting analysis for {ticker}")
            logger.info(f"{'='*60}")

            # Step 1: Fetch data
            data = self.analysis_engine.fetch_data(ticker)
            if data is None or len(data) < 11:
                logger.warning(f"Insufficient data for {ticker}")
                return None

            # Step 2: Check VWAP condition
            vwap_met, price, vwap = self.analysis_engine.check_vwap_condition(data)
            logger.info(f"VWAP Check: Price=${price:.2f} vs VWAP=${vwap:.2f} - {'✓ PASS' if vwap_met else '✗ FAIL'}")

            if not vwap_met:
                logger.info(f"{ticker}: Price below VWAP, no signal generated")
                return {"ticker": ticker, "signal": "HOLD", "reason": "Price below VWAP"}

            # Step 3: Check Volume condition
            volume_met, last_vol, avg_vol = self.analysis_engine.check_volume_condition(data)
            logger.info(f"Volume Check: Last={last_vol:,.0f} vs Avg={avg_vol:,.0f} (2x={2*avg_vol:,.0f}) - {'✓ PASS' if volume_met else '✗ FAIL'}")

            if not volume_met:
                logger.info(f"{ticker}: Volume condition not met, no signal generated")
                return {"ticker": ticker, "signal": "HOLD", "reason": "Volume below 2x average"}

            # Step 4: Generate chart for vision analysis
            chart_path = self.analysis_engine.generate_chart(ticker, data)
            if not chart_path:
                logger.error(f"Failed to generate chart for {ticker}")
                return {"ticker": ticker, "signal": "HOLD", "reason": "Chart generation failed"}

            # Encode chart for API
            image_base64 = self.analysis_engine.encode_image(chart_path)

            # Step 5: Vision analysis
            logger.info(f"Running vision analysis with {self.llm_client.model_vision}...")
            vision_result = self.llm_client.analyze_chart_vision(ticker, image_base64)

            if not vision_result:
                logger.error(f"Vision analysis failed for {ticker}")
                return {"ticker": ticker, "signal": "HOLD", "reason": "Vision analysis failed"}

            logger.info(f"Vision Analysis Result: {vision_result}")

            # Step 6: Generate annotated chart (kept for artifacts)
            try:
                # Parse vision result as JSON to extract pattern info
                import json
                pattern_info = {}
                try:
                    # Extract JSON from vision result (may be wrapped in markdown)
                    result_text = vision_result.strip()
                    if result_text.startswith('```json'):
                        result_text = result_text[7:]  # Remove ```json
                    if result_text.startswith('```'):
                        result_text = result_text[3:]  # Remove ```
                    if result_text.endswith('```'):
                        result_text = result_text[:-3]  # Remove trailing ```
                    result_text = result_text.strip()

                    pattern_info = json.loads(result_text)
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse vision result as JSON, using empty pattern info")
                    pattern_info = {}

                # Always generate annotated chart (user requested "always annotate")
                annotated_path = self.analysis_engine.generate_annotated_chart(ticker, data, pattern_info)
                if annotated_path:
                    logger.info(f"Annotated chart saved: {annotated_path}")

            except Exception as e:
                logger.warning(f"Error generating annotated chart: {e}")

            # Step 7: Final logic decision
            analysis_data = {
                "ticker": ticker,
                "current_price": price,
                "vwap": vwap,
                "vwap_condition_met": vwap_met,
                "last_volume": last_vol,
                "avg_volume": avg_vol,
                "volume_condition_met": volume_met,
                "vision_analysis": vision_result
            }

            logger.info(f"Running final logic analysis with {self.llm_client.model_logic}...")
            final_signal = self.llm_client.analyze_logic(ticker, analysis_data)

            if not final_signal:
                logger.error(f"Logic analysis failed for {ticker}")
                return {"ticker": ticker, "signal": "HOLD", "reason": "Logic analysis failed"}

            logger.info(f"Final Signal: {final_signal}")

            # Step 8: Send alert if BUY signal
            result = {
                "ticker": ticker,
                "signal": final_signal,
                "price": price,
                "vwap": vwap,
                "volume_ratio": last_vol / avg_vol if avg_vol > 0 else 0,
                "vision_result": vision_result,
                "annotated_chart": annotated_path
            }

            if "SIGNAL: BUY" in final_signal.upper():
                # Include GitHub Actions URL in alert
                artifact_url = os.getenv('GITHUB_ACTIONS_URL', '')
                alert_message = f"""Strong Buy Signal for {ticker}

Price: ${price:.2f}
VWAP: ${vwap:.2f}
Volume Ratio: {last_vol/avg_vol:.1f}x

Pattern: {vision_result}"""
                if artifact_url:
                    alert_message += f"\n\nView charts: {artifact_url}"

                self.alert_module.send_alert(ticker, alert_message)

            return result

        finally:
            # Clean up only the vision chart, NOT the annotated chart
            # Annotated charts are kept for GitHub Artifacts
            if chart_path and os.path.exists(chart_path):
                try:
                    os.remove(chart_path)
                    logger.info(f"Cleaned up vision chart: {chart_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up chart {chart_path}: {e}")

    def run(self):
        """Main execution loop."""
        # Check if it's a weekday and market hours
        now = datetime.utcnow()
        weekday = now.weekday()  # 0=Monday, 6=Sunday
        hour_utc = now.hour

        # Skip weekends (Saturday=5, Sunday=6 in weekday())
        if weekday >= 5:  # Saturday or Sunday
            logger.info("Weekend - skipping analysis")
            return

        # Market hours check (London + New York)
        # Winter: London 8:00-16:30 UTC, NY 14:30-21:00 UTC → Combined 8:00-21:00 UTC
        # Summer: London 7:00-15:30 UTC, NY 13:30-20:00 UTC → Combined 7:00-20:00 UTC
        # Allow hours 7-21 UTC to cover both markets
        if hour_utc < 7 or hour_utc > 21:
            logger.info(f"Outside market hours (UTC hour {hour_utc}) - skipping analysis")
            return

        logger.info("="*60)
        logger.info("Stock Monitor Started")
        logger.info(f"Execution time: {datetime.now()}")
        logger.info("="*60)

        # Load tickers
        tickers = self.input_module.load_tickers()
        if not tickers:
            logger.error("No tickers loaded. Exiting.")
            return

        results = []

        # Analyze each ticker
        for ticker in tickers:
            try:
                result = self.analyze_ticker(ticker)
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing {ticker}: {e}")
                continue

        # Summary
        logger.info("\n" + "="*60)
        logger.info("Analysis Summary")
        logger.info("="*60)
        for result in results:
            signal_emoji = "🟢" if "BUY" in result.get("signal", "") else "⚪"
            logger.info(f"{signal_emoji} {result['ticker']}: {result['signal']}")

        buy_signals = sum(1 for r in results if "BUY" in r.get("signal", ""))
        logger.info(f"\nTotal analyzed: {len(results)}")
        logger.info(f"Buy signals: {buy_signals}")


def main():
    """Entry point."""
    try:
        monitor = StockMonitor()
        monitor.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()
