from dataclasses import dataclass
from typing import List

from cirq import Gate, Operation


@dataclass
class RecoveryGate:
    gate: Gate
    qubit_index: int
    symptom: List[int]


@dataclass
class RecoveryOperation:
    operation: Operation
    symptom: List[int]
