from dataclasses import dataclass
from enum import Enum, auto


class LogicalGateLabel(Enum):
    H = auto()
    T = auto()
    X = auto()
    Z = auto()


@dataclass
class LogicalOperation:
    gate: LogicalGateLabel
    qubit_index: int
