from abc import ABC, abstractmethod

from cirq import Circuit, LineQubit, Operation


DELAYED_NOISE_TAG = 'DELAYED_NOISE_TAG'


class OperationsApplier(ABC):
    def __init__(self,
                 operations: list[Operation],
                 measurement_qubit: LineQubit,
                 ):
        self._operations = operations
        self._measurement_qubit = measurement_qubit

    def get_application_circuit(self) -> Circuit:
        self._validate()
        if not self._operations:
            return Circuit()
        return self._perform_get_application_circuit()

    @abstractmethod
    def _perform_get_application_circuit(self) -> Circuit:
        pass

    def _validate(self) -> None:
        self._validate_disjoint_qubits()

    def _validate_disjoint_qubits(self) -> None:
        operation_qubits = {qubit for operation in self._operations for qubit in operation.qubits}
        if self._measurement_qubit in operation_qubits:
            raise ValueError(f"The target qubits and measurement qubit must be disjoint. "
                             f"Found duplicate qubit {self._measurement_qubit}.")
