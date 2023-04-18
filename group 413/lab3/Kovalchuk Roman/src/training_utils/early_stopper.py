import numpy as np


class EarlyStopper:
    def __init__(self, tolerance=7, min_delta=1e-4):
        self.tolerance = tolerance
        self.min_delta = float(min_delta)
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss: float):
        if self.best_score is None:
            self.best_score = val_loss
        elif np.abs(val_loss - self.best_score) < self.min_delta:
            self.counter += 1
            print(f'Early stopping counter: {self.counter} out of {self.tolerance}')
            if self.counter >= self.tolerance:
                print("Early stopping")
                self.early_stop = True

        else:
            self.best_score = val_loss
            self.counter = 0

        return self.early_stop
