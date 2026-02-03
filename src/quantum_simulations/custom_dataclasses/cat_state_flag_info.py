from dataclasses import dataclass, field
from typing import Optional

from numpy import array
from numpy._typing import NDArray


@dataclass
class CatStateFlagInfo:
    control_qubit_index: int
    recovery_qubit_num: Optional[int] = None
    flags_outcome: NDArray[int] = field(default_factory=lambda: array([]))
