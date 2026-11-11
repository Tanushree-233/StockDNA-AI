"""
Decision Calibration Module for StockDNA-AI (Candidate v3).

Provides probability decision rules, frequency safeguards, and walk-forward
calibration algorithms to convert multi-class model probability outputs
P(SELL), P(HOLD), P(BUY) into discrete trading action signals.

Class contract:
- Index 0: SELL
- Index 1: HOLD
- Index 2: BUY
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score


class DecisionRuleType(str, Enum):
    ARGMAX = "argmax"
    PRIOR_ADJUSTED = "prior_adjusted"
    MARGIN_THRESHOLD = "margin_threshold"
    DIRECT_THRESHOLD = "direct_threshold"


@dataclass
class FrequencySafeguards:
    """
    Safeguards preventing degenerate decision rules that predict BUY or SELL excessively
    or collapse the neutral HOLD state.
    """
    max_buy_rate: float = 0.40   # Maximum allowable fraction of BUY predictions
    max_sell_rate: float = 0.40  # Maximum allowable fraction of SELL predictions
    min_hold_rate: float = 0.30  # Minimum allowable fraction of HOLD predictions

    def is_valid(self, preds: np.ndarray) -> bool:
        """Validates predicted class distribution against safeguard boundaries."""
        n = len(preds)
        if n == 0:
            return False
        counts = np.bincount(preds, minlength=3)
        frac_sell = counts[0] / n
        frac_hold = counts[1] / n
        frac_buy = counts[2] / n

        if frac_buy > self.max_buy_rate:
            return False
        if frac_sell > self.max_sell_rate:
            return False
        if frac_hold < self.min_hold_rate:
            return False
        return True


@dataclass
class CalibratedDecisionPolicy:
    """
    A serialized, executable decision policy mapping (N, 3) probabilities to class signals.
    """
    rule_type: DecisionRuleType
    parameters: Dict[str, Any] = field(default_factory=dict)
    safeguards: Optional[FrequencySafeguards] = None
    cal_macro_f1: float = 0.0
    cal_balanced_accuracy: float = 0.0

    def predict(self, probs: np.ndarray) -> np.ndarray:
        """
        Applies the decision policy to predicted probability matrix.
        
        Args:
            probs: Array of shape (N, 3) with columns [P(SELL), P(HOLD), P(BUY)].
        Returns:
            1D array of shape (N,) with integer predictions in {0, 1, 2}.
        """
        probs = np.asarray(probs, dtype=np.float64)
        if probs.ndim != 2 or probs.shape[1] != 3:
            raise ValueError(f"Expected probability matrix of shape (N, 3), got {probs.shape}")

        # Ensure probabilities are normalized and non-negative
        if np.any(probs < 0):
            raise ValueError("Probabilities cannot be negative")
        
        row_sums = probs.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        p = probs / row_sums

        if self.rule_type == DecisionRuleType.ARGMAX:
            return np.argmax(p, axis=1)

        elif self.rule_type == DecisionRuleType.PRIOR_ADJUSTED:
            priors = np.array(self.parameters.get("priors", [0.25, 0.50, 0.25]), dtype=np.float64)
            priors = np.where(priors <= 0, 1e-6, priors)
            scaled = p / priors
            return np.argmax(scaled, axis=1)

        elif self.rule_type == DecisionRuleType.MARGIN_THRESHOLD:
            # Margin decision rule:
            # BUY if P(BUY) - P(HOLD) >= delta_buy and P(BUY) >= P(SELL)
            # SELL if P(SELL) - P(HOLD) >= delta_sell and P(SELL) > P(BUY)
            # else HOLD
            d_sell = float(self.parameters.get("delta_sell", 0.0))
            d_buy = float(self.parameters.get("delta_buy", 0.0))

            preds = np.ones(len(p), dtype=int)  # Default HOLD
            mask_buy = (p[:, 2] - p[:, 1] >= d_buy) & (p[:, 2] >= p[:, 0])
            mask_sell = (p[:, 0] - p[:, 1] >= d_sell) & (p[:, 0] > p[:, 2])

            # In case of tie, higher margin wins
            conflict = mask_buy & mask_sell
            if np.any(conflict):
                buy_margin = (p[:, 2] - p[:, 1])[conflict]
                sell_margin = (p[:, 0] - p[:, 1])[conflict]
                mask_buy[conflict] = buy_margin >= sell_margin
                mask_sell[conflict] = sell_margin > buy_margin

            preds[mask_buy] = 2
            preds[mask_sell] = 0
            return preds

        elif self.rule_type == DecisionRuleType.DIRECT_THRESHOLD:
            # Direct threshold rule:
            # BUY if P(BUY) >= theta_buy and P(BUY) >= P(SELL)
            # SELL if P(SELL) >= theta_sell and P(SELL) > P(BUY)
            # else HOLD
            t_sell = float(self.parameters.get("theta_sell", 0.333))
            t_buy = float(self.parameters.get("theta_buy", 0.333))

            preds = np.ones(len(p), dtype=int)  # Default HOLD
            mask_buy = (p[:, 2] >= t_buy) & (p[:, 2] >= p[:, 0])
            mask_sell = (p[:, 0] >= t_sell) & (p[:, 0] > p[:, 2])

            conflict = mask_buy & mask_sell
            if np.any(conflict):
                mask_buy[conflict] = p[conflict, 2] >= p[conflict, 0]
                mask_sell[conflict] = p[conflict, 0] > p[conflict, 2]

            preds[mask_buy] = 2
            preds[mask_sell] = 0
            return preds

        else:
            raise ValueError(f"Unknown decision rule type: {self.rule_type}")


def calibrate_margin_thresholds(
    p_cal: np.ndarray,
    y_cal: np.ndarray,
    safeguards: Optional[FrequencySafeguards] = None,
    delta_sell_grid: Optional[np.ndarray] = None,
    delta_buy_grid: Optional[np.ndarray] = None,
) -> CalibratedDecisionPolicy:
    """
    Calibrates margin thresholds (delta_sell, delta_buy) on calibration data.
    Maximizes Macro F1 subject to frequency safeguards.
    """
    if safeguards is None:
        safeguards = FrequencySafeguards()

    if delta_sell_grid is None:
        delta_sell_grid = np.linspace(-0.15, 0.15, 31)
    if delta_buy_grid is None:
        delta_buy_grid = np.linspace(-0.15, 0.15, 31)

    best_deltas = (0.0, 0.0)
    # Start with argmax score as baseline
    default_preds = np.argmax(p_cal, axis=1)
    best_f1 = f1_score(y_cal, default_preds, average="macro", zero_division=0)
    best_bal_acc = balanced_accuracy_score(y_cal, default_preds)

    for d_sell in delta_sell_grid:
        for d_buy in delta_buy_grid:
            cand_policy = CalibratedDecisionPolicy(
                rule_type=DecisionRuleType.MARGIN_THRESHOLD,
                parameters={"delta_sell": float(d_sell), "delta_buy": float(d_buy)},
            )
            cand_preds = cand_policy.predict(p_cal)

            # Check frequency safeguards
            if not safeguards.is_valid(cand_preds):
                continue

            cand_f1 = f1_score(y_cal, cand_preds, average="macro", zero_division=0)
            cand_bal_acc = balanced_accuracy_score(y_cal, cand_preds)

            if cand_f1 > best_f1 or (cand_f1 == best_f1 and cand_bal_acc > best_bal_acc):
                best_f1 = cand_f1
                best_bal_acc = cand_bal_acc
                best_deltas = (float(d_sell), float(d_buy))

    return CalibratedDecisionPolicy(
        rule_type=DecisionRuleType.MARGIN_THRESHOLD,
        parameters={"delta_sell": best_deltas[0], "delta_buy": best_deltas[1]},
        safeguards=safeguards,
        cal_macro_f1=float(best_f1),
        cal_balanced_accuracy=float(best_bal_acc),
    )


def calibrate_direct_thresholds(
    p_cal: np.ndarray,
    y_cal: np.ndarray,
    safeguards: Optional[FrequencySafeguards] = None,
    theta_sell_grid: Optional[np.ndarray] = None,
    theta_buy_grid: Optional[np.ndarray] = None,
) -> CalibratedDecisionPolicy:
    """
    Calibrates absolute probability thresholds (theta_sell, theta_buy) on calibration data.
    Maximizes Macro F1 subject to frequency safeguards.
    """
    if safeguards is None:
        safeguards = FrequencySafeguards()

    if theta_sell_grid is None:
        theta_sell_grid = np.linspace(0.25, 0.45, 21)
    if theta_buy_grid is None:
        theta_buy_grid = np.linspace(0.25, 0.45, 21)

    best_thetas = (0.333, 0.333)
    default_preds = np.argmax(p_cal, axis=1)
    best_f1 = f1_score(y_cal, default_preds, average="macro", zero_division=0)
    best_bal_acc = balanced_accuracy_score(y_cal, default_preds)

    for t_sell in theta_sell_grid:
        for t_buy in theta_buy_grid:
            cand_policy = CalibratedDecisionPolicy(
                rule_type=DecisionRuleType.DIRECT_THRESHOLD,
                parameters={"theta_sell": float(t_sell), "theta_buy": float(t_buy)},
            )
            cand_preds = cand_policy.predict(p_cal)

            if not safeguards.is_valid(cand_preds):
                continue

            cand_f1 = f1_score(y_cal, cand_preds, average="macro", zero_division=0)
            cand_bal_acc = balanced_accuracy_score(y_cal, cand_preds)

            if cand_f1 > best_f1 or (cand_f1 == best_f1 and cand_bal_acc > best_bal_acc):
                best_f1 = cand_f1
                best_bal_acc = cand_bal_acc
                best_thetas = (float(t_sell), float(t_buy))

    return CalibratedDecisionPolicy(
        rule_type=DecisionRuleType.DIRECT_THRESHOLD,
        parameters={"theta_sell": best_thetas[0], "theta_buy": best_thetas[1]},
        safeguards=safeguards,
        cal_macro_f1=float(best_f1),
        cal_balanced_accuracy=float(best_bal_acc),
    )
