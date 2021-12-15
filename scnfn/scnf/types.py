from typing import List, Tuple

Literal = str
Probability = float

Disjunction = List[Literal]
CNF = List[Disjunction]
SCNF = List[Tuple[Disjunction, Probability]]
