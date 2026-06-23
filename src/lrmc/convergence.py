# generic
from abc import ABC, abstractmethod

# third party
import numpy as np


class ConvergenceMonitor(ABC):
    def __init__(
        self, patience: int = 0, tolerance: float = 1e-6, verbose: int = 1
    ):
        self.patience = patience
        self.tolerance = tolerance
        self.verbose = verbose

        self.n_iter_ = None
        self.old_estimate_ = None

    @abstractmethod
    def estimate_error(self, new_estimate):
        return

    def is_converged(self, new_estimate):
        # initialization
        if self.old_estimate_ is None:
            self.old_estimate_ = new_estimate
            self.n_iter_ = 0

            return False

        # evaluate
        error = self.estimate_error(new_estimate)

        if error < self.tolerance:
            if self.verbose > 0:
                print(f"Converged after {self.n_iter_} iterations")
            return True

        self.n_iter_ += 1
        self.old_estimate_ = new_estimate

        return False


class FactorConvergence(ConvergenceMonitor):
    def __init__(
        self, patience: int = 0, tolerance: float = 1e-6, verbose: int = 1
    ):
        super().__init__(
            patience=patience, tolerance=tolerance, verbose=verbose
        )

    def estimate_error(self, new_estimate):
        diff = np.linalg.norm(new_estimate - self.old_estimate_)
        scale = np.linalg.norm(self.old_estimate_)

        return diff**2 / scale**2
