from typing import Iterable

import numpy as np

State = Iterable[bool]
Transition = tuple[State, State]
TrimmedTransition = tuple[State, bool]

PBN = list[tuple[np.ndarray, np.ndarray]]
