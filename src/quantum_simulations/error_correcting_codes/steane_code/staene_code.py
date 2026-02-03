from typing import Optional

from cirq import Circuit, LineQubit, Operation, X, Z

from quantum_simulations.custom_dataclasses.check_matrix import CheckMatrix
from quantum_simulations.custom_dataclasses.logical_operation import LogicalGateLabel, LogicalOperation
from quantum_simulations.error_correcting_codes.stabilizer_code.stabilizer_code import StabilizerCode
from quantum_simulations.predefined_check_matrix_values import get_check_matrix_values_steane


class SteaneCode(StabilizerCode):
    def __init__(self, qubits: Optional[list[LineQubit]] = None):
        self.check_matrix = CheckMatrix(matrix=get_check_matrix_values_steane())
        super().__init__(check_matrix=self.check_matrix, qubits=qubits)

    def _get_anticommuter_for_generator(self, generator_index: int) -> list[Operation]:
        target_qubit_index_per_sector = [3, 1, 0]
        target_qubit_index_per_generator = target_qubit_index_per_sector * 2
        target_qubit = target_qubit_index_per_generator[generator_index]
        is_x_generator = generator_index < len(target_qubit_index_per_sector)
        gate = Z if is_x_generator else X
        return [gate(self.data_qubits[target_qubit])]

    def _perform_get_operation_circuit(self, operation: LogicalOperation) -> Optional[Circuit]:
        if operation.gate == LogicalGateLabel.Z:
            return Circuit(Z(qubit) for qubit in self.data_qubits)
        if operation.gate == LogicalGateLabel.X:
            return Circuit(X(qubit) for qubit in self.data_qubits)
        return None
