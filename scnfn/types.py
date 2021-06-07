from typing import Iterable

State = Iterable[bool]
Transition = tuple[State, State]
TrimmedTransition = tuple[State, bool]
