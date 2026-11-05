"""Causal, same-source-conversation H1 features (standard library only)."""
from dataclasses import dataclass
import math

ALPHA = 0.3
H1_NAMES = (
    "h1_output_ewma", "h1_output_mean", "h1_output_std",
    "h1_output_difference", "h1_output_slope", "h1_previous_output_input_ratio",
    "h1_ratio_ewma", "h1_input_multiplier", "h1_observations",
)


@dataclass
class History:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    ewma: float = math.nan
    ratio_ewma: float = math.nan
    last_output: float = math.nan
    last_input: float = math.nan
    last_turn: int = -1
    penultimate_output: float = math.nan
    penultimate_turn: int = -1

    def snapshot(self, current_input):
        """No current target argument: snapshot precedes the observation update."""
        ratio = self.last_output / self.last_input if self.last_input > 0 else math.nan
        multiplier = current_input / self.last_input if self.last_input > 0 else math.nan
        difference = self.last_output - self.penultimate_output if self.count >= 2 else math.nan
        slope = difference / (self.last_turn - self.penultimate_turn) if self.count >= 2 else math.nan
        return [self.ewma, self.mean if self.count else math.nan,
                math.sqrt(max(0.0, self.m2 / self.count)) if self.count else math.nan,
                difference, slope, ratio, self.ratio_ewma, multiplier, float(self.count)]

    def observe(self, input_tokens, output_tokens, turn):
        self.count += 1
        delta = output_tokens - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (output_tokens - self.mean)
        self.ewma = (output_tokens if self.count == 1 else
                     ALPHA * output_tokens + (1 - ALPHA) * self.ewma)
        if input_tokens > 0:
            ratio = output_tokens / input_tokens
            self.ratio_ewma = (ratio if math.isnan(self.ratio_ewma) else
                               ALPHA * ratio + (1 - ALPHA) * self.ratio_ewma)
        self.penultimate_output, self.penultimate_turn = self.last_output, self.last_turn
        self.last_output, self.last_input, self.last_turn = output_tokens, input_tokens, turn


def h1_features(sources, turns, inputs, outputs):
    """Snapshot then update per source; callers supply eligible prepared pairs only.

    Missing/invalid pairs are absent, but their original turn positions remain in
    ``turns``. Slope is the last two valid outputs' difference per original user
    turn. Population std uses all prior valid responses (ddof=0). With no history
    all features are NaN except count=0; one observation has std=0. Zero input
    denominators yield NaN and are skipped only by the ratio EWMA. The first
    valid value initializes each EWMA (adjust=False). No clock time is used.
    """
    histories, result = {}, []
    for source, turn, input_tokens, output_tokens in zip(sources, turns, inputs, outputs, strict=True):
        history = histories.setdefault(source, History())
        if turn < 0 or turn <= history.last_turn:
            raise ValueError("source turns must be nonnegative and strictly increasing")
        if any(not math.isfinite(v) or v < 0 for v in (input_tokens, output_tokens)):
            raise ValueError("eligible input/output lengths must be finite and nonnegative")
        result.append(history.snapshot(input_tokens))
        history.observe(float(input_tokens), float(output_tokens), int(turn))
    return result
