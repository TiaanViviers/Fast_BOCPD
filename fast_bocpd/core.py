import math
import numpy as np
from typing import Any, Optional, Tuple, List, Union


def _logsumexp(arr: np.ndarray) -> float:
    m = np.max(arr)
    if m == -math.inf:
        return -math.inf
    return m + math.log(np.sum(np.exp(arr - m)))


def _logsumexp_pair(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    m = a if a > b else b
    return m + math.log(math.exp(a - m) + math.exp(b - m))


class BOCPD:
    """
    Generic BOCPD engine.

    - obs_model: any object with
        * prior_stats()
        * update_stats(stats, x)
        * predictive_logpdf(stats, x)

    - hazard: any object with
        * log_transition_cp(r_prev)
        * log_transition_cont(r_prev)

    max_run_length: integer truncation of run length.
    """
    def __init__(self,
                 obs_model: Any,
                 hazard: Any,
                 max_run_length: int = 200):
        if max_run_length <= 0:
            raise ValueError("max_run_length must be > 0")

        self.obs_model = obs_model
        self.hazard = hazard
        self.max_run_length = max_run_length

        # Will hold log P(r_t = r, x_1:t)
        self.log_joint: np.ndarray = np.full(max_run_length + 1,
                                             -math.inf,
                                             dtype=float)
        # One stats object per run length r
        self.stats: List[Optional[Any]] = [None] * (max_run_length + 1)

        self.reset()

    def reset(self) -> None:
        """
        Reset to prior (as if no data has been seen).
        """
        self.log_joint[:] = -math.inf
        self.log_joint[0] = 0.0   # log P(r_0=0, no data) = 0
        self.stats = [None] * (self.max_run_length + 1)
        self.stats[0] = self.obs_model.prior_stats()

    def update(self, x: Union[float, np.ndarray]) -> Tuple[np.ndarray, float]:
        """
        Process one new observation x_t.

        Returns:
            posterior_r: array of P(r_t = r | x_1:t)
            cp_prob:     posterior_r[0], probability of changepoint at t
        """
        R = self.max_run_length
        new_log_joint = np.full(R + 1, -math.inf, dtype=float)
        new_stats: List[Optional[Any]] = [None] * (R + 1)

        prior_stats = self.obs_model.prior_stats()
        # Set new_stats[0] once - same for all changepoint paths
        new_stats[0] = prior_stats

        for r_prev in range(R + 1):
            lj_prev = self.log_joint[r_prev]
            if lj_prev == -math.inf:
                continue

            stats_prev = self.stats[r_prev]
            if stats_prev is None:
                continue

            # Predictive log likelihood under current run hypothesis
            log_pred = self.obs_model.predictive_logpdf(stats_prev, x)

            # 2) changepoint branch: r_t = 0
            # At changepoint, predict x under the PRIOR (not under r_prev's stats)
            log_pred_cp = self.obs_model.predictive_logpdf(prior_stats, x)
            logp_cp = lj_prev + log_pred_cp + self.hazard.log_transition_cp(r_prev)
            new_log_joint[0] = _logsumexp_pair(new_log_joint[0], logp_cp)

            # 3) continuation branch: r_t = r_prev + 1
            # At continuation, predict x under current run's stats
            r_cont = r_prev + 1
            if r_cont <= R:
                logp_cont = lj_prev + log_pred + self.hazard.log_transition_cont(r_prev)
                new_log_joint[r_cont] = _logsumexp_pair(new_log_joint[r_cont], logp_cont)
                new_stats[r_cont] = self.obs_model.update_stats(stats_prev, x)

        # Normalize to get posterior over run length
        log_Z = _logsumexp(new_log_joint)
        if log_Z == -math.inf:
            posterior_r = np.zeros(R + 1, dtype=float)
        else:
            posterior_r = np.exp(new_log_joint - log_Z)

        # Update internal state
        self.log_joint = new_log_joint
        self.stats = new_stats

        cp_prob = float(posterior_r[0])
        return posterior_r, cp_prob
