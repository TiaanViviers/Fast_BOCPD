"""
Hazard function parameter wrappers.

The actual implementation is in C, these are just Python wrappers for validation.
"""


class ConstantHazard:
    r"""
    Constant hazard function: :math:`H = 1 / \lambda`.

    At each time step, independent of the current run length r:
        P(changepoint)  = H
        P(continuation) = 1 - H
    
    Parameters
    ----------
    lambda_ : float
        Expected run length (must be > 0)
    """
    def __init__(self, lambda_: float):
        if lambda_ <= 0:
            raise ValueError("lambda_ must be > 0")
        self.lambda_ = float(lambda_)
        self.H = 1.0 / self.lambda_
        if not (0.0 < self.H < 1.0):
            raise ValueError("Hazard H = 1/lambda_ must be in (0, 1)")
