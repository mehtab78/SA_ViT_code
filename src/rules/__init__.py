"""
Rule Engine for Candlestick Pattern Recognition (Semantic Branch)
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import Union, List, Tuple, Dict, Any

@dataclass
class CandleProperties:
    """Properties of a single candlestick."""
    open: float
    high: float
    low: float
    close: float

    @property
    def is_bullish(self) -> bool:
        """Check if candle is bullish (white/green)."""
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        """Check if candle is bearish (black/red)."""
        return self.close < self.open

    @property
    def body_size(self) -> float:
        """Get the absolute size of the candle body."""
        return abs(self.close - self.open)

    @property
    def total_range(self) -> float:
        """Get the total range (high - low)."""
        return self.high - self.low

    @property
    def upper_shadow(self) -> float:
        """Get the upper shadow length."""
        return self.high - max(self.open, self.close)

    @property
    def lower_shadow(self) -> float:
        """Get the lower shadow length."""
        return min(self.open, self.close) - self.low

def create_candle(ohlc: np.ndarray) -> CandleProperties:
    """Create CandleProperties from OHLC array."""
    return CandleProperties(
        open=float(ohlc[0]),
        high=float(ohlc[1]),
        low=float(ohlc[2]),
        close=float(ohlc[3])
    )

class RuleEngine:
    """Rule-Based Pattern Recognition Engine."""
    
    def __init__(self, doji_threshold: float = 0.1, long_body_multiplier: float = 1.5):
        self.doji_threshold = doji_threshold
        self.long_body_multiplier = long_body_multiplier
        self.pattern_names = [
            'Advance Block', 'Doji Star', 'Evening Star'
        ]

    def _heaviside(self, condition: bool) -> float:
        """Heaviside step function."""
        return 1.0 if condition else 0.0

    def _calculate_average_body(self, ohlc_window: np.ndarray, exclude_last_n: int = 3) -> float:
        """Calculate average body size from the window."""
        if len(ohlc_window) <= exclude_last_n:
            return 0.01

        bodies = []
        for i in range(len(ohlc_window) - exclude_last_n):
            candle = create_candle(ohlc_window[i])
            bodies.append(candle.body_size)

        return np.mean(bodies) if bodies else 0.01

    def detect_advance_block(self, c1: CandleProperties, c2: CandleProperties, c3: CandleProperties) -> float:
        """Detect Advance Block pattern (Bearish Reversal)."""
        # All three candles must be bullish
        all_bullish = c1.is_bullish and c2.is_bullish and c3.is_bullish
        if not all_bullish:
            return 0.0

        # Consecutive higher closes
        higher_closes = c1.close < c2.close < c3.close
        
        # Smaller bodies (weakening trend)
        smaller_bodies = c3.body_size < c2.body_size < c1.body_size
        
        # Upper shadows getting longer (selling pressure)
        longer_shadows = c2.upper_shadow > c1.upper_shadow or c3.upper_shadow > c2.upper_shadow

        # Combine conditions
        if higher_closes and (smaller_bodies or longer_shadows):
            return 1.0
        
        return 0.0

    def detect_doji_star(self, c1: CandleProperties, c2: CandleProperties, avg_body: float) -> float:
        """Detect Doji Star pattern (Bearish Reversal)."""
        # C1 is a long bullish candle
        c1_long_bullish = c1.is_bullish and c1.body_size >= avg_body * self.long_body_multiplier
        
        if not c1_long_bullish:
            return 0.0

        # C2 is a doji
        c2_is_doji = c2.body_size < self.doji_threshold * avg_body
        
        if not c2_is_doji:
            return 0.0

        # Gap up - Doji opens above C1's close
        gap_up = c2.open > c1.close

        return 1.0 if gap_up else 0.0

    def detect_evening_star(self, c1: CandleProperties, c2: CandleProperties, c3: CandleProperties, avg_body: float) -> float:
        """Detect Evening Star pattern (Bearish Reversal)."""
        # C1 is a long bullish candle
        c1_long_bullish = c1.is_bullish and c1.body_size >= avg_body * self.long_body_multiplier
        
        if not c1_long_bullish:
            return 0.0

        # C2 gaps up (small body above C1)
        c2_gaps_up = min(c2.open, c2.close) > max(c1.open, c1.close)
        
        if not c2_gaps_up:
            return 0.0

        # C3 is bearish
        c3_bearish = c3.is_bearish
        if not c3_bearish:
            return 0.0

        # C3 closes below C1's midpoint (deep penetration)
        c1_midpoint = (c1.open + c1.close) / 2
        c3_penetrates = c3.close < c1_midpoint

        return 1.0 if c3_penetrates else 0.0

    def calculate_pattern_vector(self, ohlc_window: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Calculate the Pattern Probability Vector P ∈ R^K."""
        if isinstance(ohlc_window, torch.Tensor):
            ohlc_window = ohlc_window.cpu().numpy()

        # Need at least 3 candles for pattern detection
        if len(ohlc_window) < 3:
            return torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

        # Extract last 3 candles
        c1 = create_candle(ohlc_window[-3])
        c2 = create_candle(ohlc_window[-2])
        c3 = create_candle(ohlc_window[-1])

        # Calculate average body for context
        avg_body = self._calculate_average_body(ohlc_window, exclude_last_n=3)

        # Detect patterns
        advance_block = self.detect_advance_block(c1, c2, c3)
        doji_star = self.detect_doji_star(c2, c3, avg_body)
        evening_star = self.detect_evening_star(c1, c2, c3, avg_body)

        return torch.tensor([advance_block, doji_star, evening_star], dtype=torch.float32)

    def analyze_patterns(self, ohlc_window: np.ndarray) -> Dict[str, Any]:
        """Detailed pattern analysis with probabilities and confidence."""
        pattern_vector = self.calculate_pattern_vector(ohlc_window)
        
        # Extract candles for detailed analysis
        c1 = create_candle(ohlc_window[-3])
        c2 = create_candle(ohlc_window[-2])
        c3 = create_candle(ohlc_window[-1])
        
        analysis = {
            'pattern_vector': pattern_vector.numpy(),
            'patterns_detected': {
                name: float(score) 
                for name, score in zip(self.pattern_names, pattern_vector.numpy())
            },
            'candle_properties': {
                'c1': {
                    'is_bullish': c1.is_bullish,
                    'body_size': c1.body_size,
                    'upper_shadow': c1.upper_shadow,
                    'lower_shadow': c1.lower_shadow
                },
                'c2': {
                    'is_bullish': c2.is_bullish,
                    'body_size': c2.body_size,
                    'upper_shadow': c2.upper_shadow,
                    'lower_shadow': c2.lower_shadow
                },
                'c3': {
                    'is_bullish': c3.is_bullish,
                    'body_size': c3.body_size,
                    'upper_shadow': c3.upper_shadow,
                    'lower_shadow': c3.lower_shadow
                }
            }
        }
        
        return analysis