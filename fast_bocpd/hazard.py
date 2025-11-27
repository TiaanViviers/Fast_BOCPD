import math


class ConstantHazard:
    """
    Constant hazard model: H = 1 / lambda_.

    At each time step, independent of the current run length r:
        P(changepoint)  = H
        P(continuation) = 1 - H
    """
    def __init__(self, lambda_: float):
        if lambda_ <= 0:
            raise ValueError("lambda_ must be > 0")
        self.lambda_ = float(lambda_)
        self.H = 1.0 / self.lambda_
        if not (0.0 < self.H < 1.0):
            # For extreme lambda_, this could under/overflow numerically.
            raise ValueError("Hazard H = 1/lambda_ must be in (0, 1).")

        self.log_H = math.log(self.H)
        self.log_1mH = math.log(1.0 - self.H)

    def log_transition_cp(self, r_prev: int) -> float:
        """
        log P(r_t = 0 | r_{t-1} = r_prev)

        For a constant hazard, this is always log(H).
        """
        return self.log_H

    def log_transition_cont(self, r_prev: int) -> float:
        """
        log P(r_t = r_prev + 1 | r_{t-1} = r_prev)

        For a constant hazard, this is always log(1 - H).
        """
        return self.log_1mH
