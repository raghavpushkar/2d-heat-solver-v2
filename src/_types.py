from collections.abc import Callable

import numpy as np

ICFunc = Callable[[np.ndarray, np.ndarray], np.ndarray]
