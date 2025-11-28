"""
Helper utilities for changepoint detection in streaming applications.
"""
import numpy as np
from typing import List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class Changepoint:
    """Record of a detected changepoint"""
    index: int
    """Time index where changepoint occurred"""
    
    prev_run_length: int = 0
    """Length of the segment that just ended"""
    
    confidence: float = 0.0
    """Confidence in this changepoint (posterior probability at r=0)"""
    
    observation: float = 0.0
    """Observation value at changepoint"""
    
    metadata: Optional[Any] = None
    """Optional user-provided metadata (e.g., timestamp, label)"""
    
    def __str__(self):
        meta_str = f" ({self.metadata})" if self.metadata else ""
        return (f"Changepoint at t={self.index}{meta_str}: "
                f"previous segment lasted {self.prev_run_length} steps "
                f"(confidence: {self.confidence:.1%})")


class OnlineChangeDetector:
    """
    Wrapper for BOCPD optimized for online/streaming detection.
    
    Automatically detects changepoints by monitoring when the most likely
    run length (MAP estimate) jumps to zero.
    
    Features:
    - Automatic changepoint detection using MAP
    - History tracking
    - Confidence scoring
    - Optional metadata attachment
    
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
    
    def __init__(self, bocpd, min_confidence: float = 0.3):
        """
        Initialize online detector.
        
        Args:
            bocpd: BOCPD instance
            min_confidence: Minimum confidence to report changepoint (default: 0.3)
                           Lower values = more sensitive, more false positives
                           Higher values = less sensitive, fewer false positives
        """
        self.bocpd = bocpd
        self.min_confidence = min_confidence
        
        self._t = 0
        self._prev_map_r = None
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
        posterior_r, cp_prob = self.bocpd.update(x)
        map_r = self.bocpd.get_map_run_length()
        confidence = self.bocpd.get_map_confidence()
        
        # Track history
        self._map_history.append(map_r)
        
        # Detect changepoint: MAP jumped to 0
        cp = None
        if self._prev_map_r is not None and map_r == 0 and self._prev_map_r > 0:
            # Only report if confident enough
            if confidence >= self.min_confidence:
                cp = Changepoint(
                    index=self._t,
                    prev_run_length=self._prev_map_r,
                    confidence=confidence,
                    observation=x,
                    metadata=metadata
                )
                self._changepoints.append(cp)
        
        self._prev_map_r = map_r
        self._t += 1
        
        return cp
    
    def get_current_run_length(self) -> int:
        """
        Get current run length (time since last changepoint).
        
        Returns:
            Number of observations since last changepoint
        """
        if self._prev_map_r is None:
            return 0
        return self._prev_map_r
    
    def get_changepoints(self) -> List[Changepoint]:
        """Get all detected changepoints"""
        return self._changepoints.copy()
    
    def get_map_history(self) -> np.ndarray:
        """
        Get complete history of MAP run length estimates.
        
        Returns:
            Array where element i is the MAP run length at time i
        """
        return np.array(self._map_history)
    
    def get_segments(self) -> List[Tuple[int, int]]:
        """
        Get segments between changepoints as (start, end) indices.
        
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
        self._changepoints = []
        self._map_history = []
