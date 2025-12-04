"""
Helper utilities for changepoint detection in streaming applications.
"""
import math
from typing import List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class Changepoint:
    """Record of a detected changepoint"""
    index: int
    """Time index where changepoint occurred"""
    
    prev_run_length: int = 0
    """Estimated length of the segment that just ended (from MAP estimate)"""
    
    cp_prob: float = 0.0
    """Changepoint probability P(r_t=0 | x_1:t)"""
    
    map_run_length: int = 0
    """MAP run length estimate at detection time"""
    
    map_confidence: float = 0.0
    """Confidence in MAP estimate (max posterior probability)"""
    
    observation: float = 0.0
    """Observation value at changepoint"""
    
    metadata: Optional[Any] = None
    """Optional user-provided metadata (e.g., timestamp, label)"""
    
    @property
    def confidence(self) -> float:
        """Alias for cp_prob (backward compatibility)"""
        return self.cp_prob
    
    def __str__(self):
        meta_str = f" ({self.metadata})" if self.metadata else ""
        return (f"Changepoint at t={self.index}{meta_str}: "
                f"previous segment lasted {self.prev_run_length} steps "
                f"(P(CP)={self.cp_prob:.1%}, MAP r={self.map_run_length})")


class OnlineChangeDetector:
    """
    Wrapper for BOCPD optimized for online/streaming detection.
    
    Automatically detects changepoints using:
    1. Changepoint probability threshold (P(r_t=0) >= min_cp_prob)
    2. MAP run length reset heuristic (sharp drop in estimated segment length)
    
    Features:
    - Automatic changepoint detection with configurable sensitivity
    - History tracking
    - Confidence scoring
    - Optional metadata attachment
    - Debouncing to prevent duplicate detections
    
    Example:
        >>> from fast_bocpd import BOCPD, GaussianNIG, ConstantHazard
        >>> from fast_bocpd.utils import OnlineChangeDetector
        >>> 
        >>> bocpd = BOCPD(GaussianNIG(...), ConstantHazard(100))
        >>> detector = OnlineChangeDetector(bocpd)
        >>> 
        >>> # Process streaming data
        >>> for observation in data_stream:
        ...     cp = detector.update(observation)
        ...     if cp:
        ...         print(f"Changepoint detected: {cp}")
    """
    
    @staticmethod
    def _compute_default_min_cp_prob(lambda_: float, bayes_factor: float = 5.0) -> float:
        """
        Compute principled default for min_cp_prob using Bayes factor.
        
        Args:
            lambda_: Expected run length (from ConstantHazard)
            bayes_factor: Evidence threshold (5=moderate, 10=strong, 3=weak)
            
        Returns:
            Minimum changepoint probability threshold
        """
        H = 1.0 / float(lambda_)
        return (bayes_factor * H) / ((1.0 - H) + bayes_factor * H)
    
    def __init__(self, bocpd, min_cp_prob: Optional[float] = None, 
                 reset_r: int = 2, drop_prev_min: Optional[int] = None, 
                 cooldown: Optional[int] = None, bayes_factor: float = 5.0):
        """
        Initialize online detector.
        
        Args:
            bocpd: BOCPD instance
            min_cp_prob: Minimum P(r_t=0) to report changepoint
                        If None, computed from hazard using Bayes factor
                        Lower = more sensitive, more false positives
                        Higher = less sensitive, fewer false positives
            reset_r: MAP run length threshold for "reset" detection (default: 2)
            drop_prev_min: Minimum previous run length for "sharp drop"
                          If None, set to max(10, 0.25*lambda_)
            cooldown: Minimum timesteps between changepoint reports
                     If None, set to max(3, 0.1*lambda_)
            bayes_factor: Evidence threshold for auto-computing min_cp_prob (default: 5.0)
                         5=moderate evidence, 10=strong, 3=weak
                         Only used if min_cp_prob is None and hazard is ConstantHazard
        """
        self.bocpd = bocpd
        self.bayes_factor = bayes_factor
        if not math.isfinite(self.bayes_factor) or self.bayes_factor <= 0.0:
            raise ValueError(f"bayes_factor must be finite and > 0, got {self.bayes_factor}")
        
        # Auto-compute defaults from hazard (duck-typing to avoid imports)
        # If hazard exposes .lambda_ (expected run length), use it for principled defaults
        hazard = getattr(bocpd, "hazard", None)
        lambda_ = getattr(hazard, "lambda_", None)
        
        needs_lambda = (min_cp_prob is None) or (drop_prev_min is None) or (cooldown is None)

        if lambda_ is not None and needs_lambda:
            try:
                lam = float(lambda_)
            except (TypeError, ValueError):
                raise ValueError(f"hazard.lambda_ must be a positive number, got {lambda_!r}")

            if not math.isfinite(lam) or lam <= 0.0:
                raise ValueError(f"hazard.lambda_ must be finite and > 0, got {lam}")

            if min_cp_prob is None:
                min_cp_prob = self._compute_default_min_cp_prob(lam, bayes_factor)
            if drop_prev_min is None:
                drop_prev_min = max(10, int(round(0.25 * lam)))
            if cooldown is None:
                cooldown = max(3, int(round(0.1 * lam)))
        
        # Fallback defaults (softer threshold for unknown hazards)
        self.min_cp_prob = min_cp_prob if min_cp_prob is not None else 0.1
        self.reset_r = reset_r
        self.drop_prev_min = drop_prev_min if drop_prev_min is not None else 10
        self.cooldown = cooldown if cooldown is not None else 5
        
        # Validate probability threshold
        if not (0.0 < self.min_cp_prob < 1.0):
            raise ValueError(f"min_cp_prob must be in (0, 1), got {self.min_cp_prob}")
        
        # Validate other parameters
        if self.cooldown < 0:
            raise ValueError(f"cooldown must be >= 0, got {self.cooldown}")
        if self.reset_r < 0:
            raise ValueError(f"reset_r must be >= 0, got {self.reset_r}")
        if self.drop_prev_min < 0:
            raise ValueError(f"drop_prev_min must be >= 0, got {self.drop_prev_min}")
        
        self._t = 0
        self._prev_map_r = None
        self._prev_cp_prob = 0.0
        self._has_emitted = False  # Track if we've ever emitted a CP
        self._last_cp_t = -self.cooldown  # Start before time 0 so can_emit is True initially
        self._pending_reset = False  # Latch reset detection during cooldown
        self._changepoints: List[Changepoint] = []
        self._map_history: List[int] = []
    
    def update(self, x: float, metadata: Optional[Any] = None) -> Optional[Changepoint]:
        """
        Process new observation and detect changepoints.
        
        Args:
            x: New observation
            metadata: Optional metadata to attach (e.g., timestamp, sample ID)
            
        Returns:
            Changepoint if detected, None otherwise
        """
        # Update BOCPD
        _, cp_prob = self.bocpd.update(x)
        map_r = self.bocpd.get_map_run_length()
        map_conf = self.bocpd.get_map_confidence()
        
        # Track history
        self._map_history.append(map_r)
        
        cp = None
        
        # Check if we're in cooldown
        can_emit = (self._t - self._last_cp_t >= self.cooldown)
        
        # Primary signal: probabilistic changepoint probability
        # (crossing threshold from below, only checked when we can emit)
        detect_cp_prob = False
        if can_emit:
            detect_cp_prob = (cp_prob >= self.min_cp_prob) and (self._prev_cp_prob < self.min_cp_prob)
        
        # Secondary signal: run-length reset heuristic
        # (MAP dropped from long run to very short run)
        detect_reset = False
        if self._prev_map_r is not None:
            detect_reset = (map_r <= self.reset_r) and (self._prev_map_r >= self.drop_prev_min)
            
            # Latch pending reset during cooldown so we don't miss it
            if detect_reset and not can_emit:
                self._pending_reset = True
        
        # Check for pending reset that was latched during cooldown
        # Only trigger if we still look "reset-like" now (prevent false positives)
        if can_emit and self._pending_reset:
            if map_r <= self.reset_r:
                detect_reset = True
            self._pending_reset = False  # Always clear latch when cooldown ends
        
        detect_cp = detect_cp_prob or detect_reset
        
        # Emit changepoint if detected and not in cooldown
        if detect_cp:
            cp = Changepoint(
                index=self._t,
                prev_run_length=self._prev_map_r if self._prev_map_r is not None else 0,
                cp_prob=cp_prob,
                map_run_length=map_r,
                map_confidence=map_conf,
                observation=x,
                metadata=metadata
            )
            self._changepoints.append(cp)
            self._last_cp_t = self._t
            self._has_emitted = True  # Mark that we've emitted at least one CP
            self._pending_reset = False  # Clear latch after emission
        
        self._prev_map_r = map_r
        self._prev_cp_prob = cp_prob
        self._t += 1
        
        return cp
    
    def get_current_run_length(self) -> int:
        """
        Get time since last detected changepoint.
        
        Returns:
            Number of observations since last changepoint detected by this wrapper
            Returns 0 if no changepoint has been detected yet
        """
        if not self._has_emitted:
            return 0
        return self._t - self._last_cp_t
    
    def get_current_map_run_length(self) -> int:
        """
        Get BOCPD's current MAP run length estimate.
        
        Returns:
            Most likely run length (segment length) according to BOCPD posterior
        """
        return 0 if self._prev_map_r is None else self._prev_map_r
    
    def get_changepoints(self) -> List[Changepoint]:
        """Get all detected changepoints"""
        return self._changepoints.copy()
    
    def get_map_history(self) -> List[int]:
        """
        Get complete history of MAP run length estimates.
        
        Returns:
            List where element i is the MAP run length at time i
        """
        return self._map_history.copy()
    
    def get_segments(self) -> List[Tuple[int, int]]:
        """
        Get segments between changepoints as (start, end) indices.
        
        Note: Segment boundaries are based on detection time (first sample after change).
        
        Returns:
            List of (start_idx, end_idx) tuples for each segment
            
        Example:
            >>> segments = detector.get_segments()
            >>> for start, end in segments:
            ...     print(f"Segment from {start} to {end} (length: {end-start})")
        """
        if not self._changepoints:
            return [(0, self._t)]
        
        segments = []
        prev_end = 0
        
        for cp in self._changepoints:
            segments.append((prev_end, cp.index))
            prev_end = cp.index
        
        # Add final segment
        if prev_end < self._t:
            segments.append((prev_end, self._t))
        
        return segments
    
    def reset(self):
        """Reset detector to initial state"""
        self.bocpd.reset()
        self._t = 0
        self._prev_map_r = None
        self._prev_cp_prob = 0.0
        self._has_emitted = False  # Reset emission flag
        self._last_cp_t = -self.cooldown  # Reset to allow immediate detection
        self._pending_reset = False
        self._changepoints = []
        self._map_history = []
    
    def __repr__(self) -> str:
        """String representation for debugging"""
        return (f"OnlineChangeDetector(min_cp_prob={self.min_cp_prob:.4f}, "
                f"reset_r={self.reset_r}, drop_prev_min={self.drop_prev_min}, "
                f"cooldown={self.cooldown}, bayes_factor={self.bayes_factor})")
