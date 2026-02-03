from abc import ABC, abstractmethod
from functools import cached_property
from typing import Optional

from cirq import Circuit, LineQubit

from quantum_simulations.custom_dataclasses.correction_circuit import CorrectionCircuit
from quantum_simulations.custom_dataclasses.logical_operation import LogicalOperation


class ErrorCorrectingCode(ABC):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance._saved_init_args = (args, kwargs)
        return instance

    def create_new(self, qubits: list[LineQubit] = 0) -> 'ErrorCorrectingCode':
        self._saved_init_args[1]['qubits'] = qubits
        return self.__class__(*self._saved_init_args[0], **self._saved_init_args[1])

    def __init__(self,
                 num_data_qubits: int,
                 num_logical_qubits: int,
                 qubits: Optional[list[LineQubit]] = None,
                 ):
        self._num_data_qubits = num_data_qubits
        self._num_logical_qubits = num_logical_qubits
        self._qubits = qubits

    @abstractmethod
    def encode_logical_qubit(self) -> Circuit:
        pass

    @abstractmethod
    def get_error_correction_circuit(self) -> CorrectionCircuit:
        pass

    @abstractmethod
    def _perform_get_operation_circuit(self, operation: LogicalOperation) -> Optional[Circuit]:
        pass

    def get_operation_circuit(self, operation: LogicalOperation) -> Circuit:
        if not 0 <= operation.qubit_index < self._num_logical_qubits:
            raise ValueError(f"Qubit index must be between 0 and {self._num_logical_qubits - 1}. Was given {operation.qubit_index}.")
        circuit = self._perform_get_operation_circuit(operation=operation)
        if circuit is None:
            raise NotImplementedError(f"Operation {operation.gate.name} is not implemented for code {self.__class__.__name__}.")
        return circuit

    @cached_property
    def data_qubits(self) -> list[LineQubit]:
        if self._qubits:
            if len(self._qubits) != self._num_data_qubits:
                raise ValueError(f"The number of provided qubits ({len(self._qubits)}) must be equal to the specified number of data qubits ({self._num_data_qubits}).")
            return self._qubits
        return LineQubit.range(self._num_data_qubits)

    @property
    def num_logical_qubits(self) -> int:
        return self._num_logical_qubits

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self._saved_init_args == other._saved_init_args
