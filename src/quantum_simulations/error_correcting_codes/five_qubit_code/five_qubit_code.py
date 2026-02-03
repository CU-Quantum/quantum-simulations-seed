from typing import Optional

from cirq import Circuit, LineQubit, Operation, X, Z

from quantum_simulations.custom_dataclasses.check_matrix import CheckMatrix
from quantum_simulations.custom_dataclasses.logical_operation import LogicalGateLabel, LogicalOperation
from quantum_simulations.error_correcting_codes.stabilizer_code.stabilizer_code import StabilizerCode
from quantum_simulations.predefined_check_matrix_values import get_check_matrix_values_5_qubit


class FiveQubitCode(StabilizerCode):
    def __init__(self, qubits: Optional[list[LineQubit]] = None):
        self.check_matrix = CheckMatrix(matrix=get_check_matrix_values_5_qubit())
        super().__init__(check_matrix=self.check_matrix, qubits=qubits)

    def _get_anticommuter_for_generator(self, generator_index: int) -> list[Operation]:
        qubits_indices_to_flip_per_generator = [(0, 2), (0, 1, 2, 3), (0, 1, 3, 4), (1, 4)]
        qubits_indices_to_flip = qubits_indices_to_flip_per_generator[generator_index]
        return [Z(self.data_qubits[qubit_index]) for qubit_index in qubits_indices_to_flip]

    def _perform_get_operation_circuit(self, operation: LogicalOperation) -> Optional[Circuit]:
        if operation.gate == LogicalGateLabel.Z:
            return Circuit(Z(qubit) for qubit in self.data_qubits)
        if operation.gate == LogicalGateLabel.X:
            return Circuit(X(qubit) for qubit in self.data_qubits)
        return None
