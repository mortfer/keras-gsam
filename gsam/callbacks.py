import math
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
import numpy as np

class RhoScheduler(Callback):
    """Rho scheduler which sets rho according to schedule.

  Arguments:
      schedule: a function that takes an epoch index
          (integer, indexed from 0) and current eho
          as inputs and returns a new rho as output (float).
  """

    def __init__(self, schedule, verbose=0):
        super().__init__()
        self.schedule = schedule
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model, "rho"):
            raise ValueError('Model must have a "rho" attribute.')
        rho = K.get_value(self.model.rho)
        scheduled_rho = self.schedule(epoch, rho)
        self.model.rho = scheduled_rho
        if self.verbose > 0:
            print(
                f"\nEpoch {epoch}: RhoScheduler setting rho "
                f"to {scheduled_rho}."
            )

class CosineAnnealingScheduler(object):
    """Cosine annealing scheduler.
    """
    def __init__(self, T_max, eta_max, eta_min=0):
        self.T_max = T_max
        self.eta_max = eta_max
        self.eta_min = eta_min

    def __call__(self, epoch, lr):     
        lr = self.eta_min + (self.eta_max - self.eta_min) * (1 + math.cos(math.pi * epoch / self.T_max)) / 2
        return lr