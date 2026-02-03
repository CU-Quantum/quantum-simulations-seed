from abc import ABC, abstractmethod

from cirq import Circuit, LineQubit, MeasurementKey


class ParityVerifier(ABC):
    def __init__(self, target_qubits: list[LineQubit], measurement_key: MeasurementKey):
        self._target_qubits = target_qubits
        self._measurement_key = measurement_key

    @abstractmethod
    def validate_parity(self) -> Circuit:
        pass
