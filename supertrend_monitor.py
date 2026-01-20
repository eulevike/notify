"""
Supertrend Indicator Monitor
Monitors stocks on weekly timeframe using Supertrend indicator (ATR period=22, multiplier=3)

This runs separately from the main Triad Logic monitor and runs every 7 days.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
from dotenv import load_dotenv

import yfinance as yf
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Config:
    """Configuration and environment variables."""

    NTFY_BASE_URL = "https://ntfy.sh"

    def __init__(self):
        load_dotenv()

        self.ntfy_topic = os.getenv("NTFY_TOPIC")
        self.tickers_file = os.getenv("TICKERS_FILE", "tickers.txt")
        self.state_file = "supertrend_state.json"

        if not self.ntfy_topic:
            raise ValueError("NTFY_TOPIC not found in environment variables")


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
            "Title": f"Supertrend Signal: {ticker}",
            "Priority": priority,
            "Tags": "chart_with_upwards_trend"
        }

        try:
            response = requests.post(url, data=message, headers=headers, timeout=10)

            if response.status_code in [200, 201]:
                logger.info(f"Alert sent for {ticker}")
                return True
            else:
                logger.error(f"Failed to send alert: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"Error sending alert: {e}")
            return False


class SupertrendIndicator:
    """Calculates Supertrend indicator."""

    def __init__(self, period: int = 22, multiplier: float = 3.0):
        self.period = period
        self.multiplier = multiplier

    def calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range."""
        high = df['High']
        low = df['Low']
        close = df['Close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Supertrend indicator.

        Args:
            df: DataFrame with High, Low, Close columns

        Returns:
            DataFrame with Supertrend columns added
        """
        df = df.copy()

        # Calculate ATR
        atr = self.calculate_atr(df, self.period)

        # Calculate basic bands
        hl2 = (df['High'] + df['Low']) / 2
        upper_band = hl2 + (self.multiplier * atr)
        lower_band = hl2 - (self.multiplier * atr)

        # Initialize Supertrend
        df['supertrend'] = np.nan
        df['supertrend_direction'] = np.nan  # 1 for uptrend (buy), -1 for downtrend (sell)

        # Calculate Supertrend
        current_trend = np.nan
        prev_upper = upper_band.iloc[0]
        prev_lower = lower_band.iloc[0]
        prev_supertrend = np.nan

        for i in range(len(df)):
            if i == 0:
                df.iloc[i, df.columns.get_loc('supertrend')] = upper_band.iloc[i]
                df.iloc[i, df.columns.get_loc('supertrend_direction')] = -1
                prev_supertrend = upper_band.iloc[i]
                current_trend = -1
                continue

            # Check for trend change
            close_prev = df['Close'].iloc[i - 1]
            close_curr = df['Close'].iloc[i]

            if current_trend == -1:  # Currently in downtrend
                if close_curr > prev_upper:
                    # Switch to uptrend
                    df.iloc[i, df.columns.get_loc('supertrend')] = lower_band.iloc[i]
                    df.iloc[i, df.columns.get_loc('supertrend_direction')] = 1
                    current_trend = 1
                    prev_supertrend = lower_band.iloc[i]
                else:
                    # Stay in downtrend
                    new_upper = min(upper_band.iloc[i], prev_upper)
                    df.iloc[i, df.columns.get_loc('supertrend')] = new_upper
                    df.iloc[i, df.columns.get_loc('supertrend_direction')] = -1
                    prev_upper = new_upper
                    prev_supertrend = new_upper
            else:  # Currently in uptrend
                if close_curr < prev_lower:
                    # Switch to downtrend
                    df.iloc[i, df.columns.get_loc('supertrend')] = upper_band.iloc[i]
                    df.iloc[i, df.columns.get_loc('supertrend_direction')] = -1
                    current_trend = -1
                    prev_supertrend = upper_band.iloc[i]
                else:
                    # Stay in uptrend
                    new_lower = max(lower_band.iloc[i], prev_lower)
                    df.iloc[i, df.columns.get_loc('supertrend')] = new_lower
                    df.iloc[i, df.columns.get_loc('supertrend_direction')] = 1
                    prev_lower = new_lower
                    prev_supertrend = new_lower

        return df


class SupertrendMonitor:
    """Main orchestrator for Supertrend monitoring."""

    def __init__(self):
        self.config = Config()
        self.alert_module = AlertModule(self.config)
        self.indicator = SupertrendIndicator(period=22, multiplier=3.0)

    def load_tickers(self) -> List[str]:
        """Read tickers from tickers.txt file."""
        try:
            tickers_path = Path(self.config.tickers_file)
            if not tickers_path.exists():
                logger.error(f"Tickers file not found: {self.config.tickers_file}")
                return []

            with open(tickers_path, 'r') as f:
                tickers = [
                    line.strip().upper()
                    for line in f
                    if line.strip() and not line.startswith('#')
                ]

            logger.info(f"Loaded {len(tickers)} tickers")
            return tickers

        except Exception as e:
            logger.error(f"Error loading tickers: {e}")
            return []

    def load_previous_state(self) -> Dict[str, str]:
        """Load previous Supertrend signals from state file."""
        try:
            state_path = Path(self.config.state_file)
            if state_path.exists():
                with open(state_path, 'r') as f:
                    state = json.load(f)
                logger.info(f"Loaded previous state with {len(state)} ticker signals")
                return state
            return {}
        except Exception as e:
            logger.warning(f"Could not load previous state: {e}")
            return {}

    def save_state(self, state: Dict[str, str]) -> None:
        """Save current Supertrend signals to state file."""
        try:
            with open(self.config.state_file, 'w') as f:
                json.dump(state, f, indent=2)
            logger.info(f"Saved state for {len(state)} tickers")
        except Exception as e:
            logger.error(f"Error saving state: {e}")

    def get_weekly_data(self, ticker: str, period: str = "5y") -> Optional[pd.DataFrame]:
        """
        Fetch weekly data for a ticker.

        Args:
            ticker: Stock ticker symbol
            period: Time period (default 5 years for weekly data)

        Returns:
            DataFrame with weekly OHLCV data or None if error
        """
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval="1wk")

            if df.empty or len(df) < self.indicator.period:
                logger.warning(f"Insufficient data for {ticker}: {len(df)} weeks")
                return None

            return df

        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return None

    def get_current_signal(self, df: pd.DataFrame) -> Tuple[str, float]:
        """
        Get current Supertrend signal from the latest data.

        Args:
            df: DataFrame with Supertrend calculations

        Returns:
            Tuple of (signal: "BUY" or "SELL", supertrend_value)
        """
        latest = df.iloc[-1]
        direction = latest['supertrend_direction']
        supertrend_value = latest['supertrend']
        close = latest['Close']

        if direction == 1:
            # Uptrend - price above supertrend
            return "BUY", supertrend_value
        else:
            # Downtrend - price below supertrend
            return "SELL", supertrend_value

    def analyze_ticker(self, ticker: str, previous_state: Dict[str, str]) -> Optional[Dict]:
        """
        Analyze a single ticker and notify if signal changed.

        Args:
            ticker: Stock ticker symbol
            previous_state: Dictionary of previous signals

        Returns:
            Analysis result dictionary or None if error
        """
        try:
            # Fetch weekly data
            df = self.get_weekly_data(ticker)
            if df is None:
                return None

            # Calculate Supertrend
            df = self.indicator.calculate(df)

            # Get current signal
            signal, st_value = self.get_current_signal(df)
            previous_signal = previous_state.get(ticker)

            latest_close = df.iloc[-1]['Close']
            latest_date = df.index[-1].strftime('%Y-%m-%d')

            result = {
                "ticker": ticker,
                "signal": signal,
                "supertrend_value": round(st_value, 2),
                "close": round(latest_close, 2),
                "latest_date": latest_date,
                "previous_signal": previous_signal,
                "changed": previous_signal != signal if previous_signal else False
            }

            logger.info(f"{ticker}: {signal} @ ${latest_close:.2f} (ST: ${st_value:.2f})")

            # Send notification if signal changed
            if result["changed"]:
                self._send_signal_notification(result)

            return result

        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {e}")
            return None

    def _send_signal_notification(self, result: Dict) -> None:
        """Send notification for signal change."""
        ticker = result["ticker"]
        new_signal = result["signal"]
        old_signal = result["previous_signal"]
        close = result["close"]
        st_value = result["supertrend_value"]
        date = result["latest_date"]

        emoji = "🟢" if new_signal == "BUY" else "🔴"
        old_emoji = "🟢" if old_signal == "BUY" else "🔴"

        message = f"""Supertrend Signal Change: {ticker}

{old_emoji} {old_signal} → {emoji} {new_signal}

Price: ${close:.2f}
Supertrend: ${st_value:.2f}
Date: {date}

Timeframe: Weekly
ATR Period: 22, Multiplier: 3.0
"""

        self.alert_module.send_alert(ticker, message)
        logger.info(f"Signal change notification sent for {ticker}: {old_signal} → {new_signal}")

    def run(self) -> None:
        """Main execution method."""
        logger.info("="*60)
        logger.info("Supertrend Monitor - Weekly Timeframe")
        logger.info("ATR Period: 22, Multiplier: 3.0")
        logger.info("="*60)

        # Load tickers
        tickers = self.load_tickers()
        if not tickers:
            logger.error("No tickers to analyze")
            return

        # Load previous state
        previous_state = self.load_previous_state()

        # Analyze each ticker
        results = []
        new_state = {}

        for ticker in tickers:
            result = self.analyze_ticker(ticker, previous_state)
            if result:
                results.append(result)
                new_state[ticker] = result["signal"]

        # Save new state
        self.save_state(new_state)

        # Summary
        logger.info("="*60)
        logger.info("Analysis Summary")
        logger.info("="*60)

        buy_count = sum(1 for r in results if r["signal"] == "BUY")
        sell_count = sum(1 for r in results if r["signal"] == "SELL")
        changed_count = sum(1 for r in results if r["changed"])

        logger.info(f"Total analyzed: {len(results)}")
        logger.info(f"BUY signals: {buy_count}")
        logger.info(f"SELL signals: {sell_count}")
        logger.info(f"Signal changes: {changed_count}")

        # Show all current signals
        logger.info("\nCurrent Signals:")
        for result in sorted(results, key=lambda x: x['ticker']):
            emoji = "🟢" if result["signal"] == "BUY" else "🔴"
            logger.info(f"  {emoji} {result['ticker']}: {result['signal']} @ ${result['close']:.2f}")

        logger.info("="*60)


if __name__ == "__main__":
    monitor = SupertrendMonitor()
    monitor.run()
