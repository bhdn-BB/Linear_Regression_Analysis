import dataclasses
import numpy as np


@dataclasses.dataclass()
class AppState:
    ndarray_field = lambda: dataclasses.field(default_factory=lambda: np.array([]))  # noqa: E731
    n_obs: int = 0
    n_feats: int = 0
    data_X: np.ndarray = ndarray_field()
    data_B: np.ndarray = ndarray_field()
    data_Y: np.ndarray = ndarray_field()
    B_hat: np.ndarray = ndarray_field()
    noise: np.ndarray = ndarray_field()
    x_precision: int = 9
    b_precision: int = 9
    b_0: float = 1.0