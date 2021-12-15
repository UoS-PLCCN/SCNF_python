from typing import Iterable, List, Tuple

import numpy as np

State = Iterable[bool]
Transition = Tuple[State, State]
TrimmedTransition = Tuple[State, bool]

PBN = List[Tuple[np.ndarray, np.ndarray]]
