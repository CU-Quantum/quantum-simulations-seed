from abc import ABC, abstractmethod
from typing import Optional

from cirq import Circuit, MeasurementKey, Operation


FAULT_TOLERANT_MEASURER_TAG = 'FAULT_TOLERANT_MEASURER'
MEASURER_WITH_SINGLE_QUBIT_TAG = 'MEASURER_WITH_SINGLE_QUBIT'


class Measurer(ABC):
    def __init__(self,
                 observables: list[list[Operation]],
                 measurement_keys: Optional[list[MeasurementKey]] = None,
                 correction_between_repetitions: bool = True,
                 ):
        self._observables = observables
        self._measurement_keys = measurement_keys or []
        self._correction_between_repetitions = correction_between_repetitions

    @abstractmethod
    def get_measurement_circuit(self) -> Circuit:
        pass
