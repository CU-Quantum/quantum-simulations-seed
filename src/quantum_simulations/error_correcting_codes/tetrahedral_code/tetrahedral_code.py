from typing import Optional

from cirq import Circuit, LineQubit, Operation, T, X, Z, inverse

from quantum_simulations.custom_dataclasses.check_matrix import CheckMatrix
from quantum_simulations.custom_dataclasses.logical_operation import LogicalGateLabel, LogicalOperation
from quantum_simulations.error_correcting_codes.stabilizer_code.stabilizer_code import StabilizerCode
from quantum_simulations.predefined_check_matrix_values import get_check_matrix_values_tetrahedral


class TetrahedralCode(StabilizerCode):
    def __init__(self, qubits: Optional[list[LineQubit]] = None):
        check_matrix = CheckMatrix(matrix=get_check_matrix_values_tetrahedral())
        super().__init__(check_matrix=check_matrix, qubits=qubits)

    def _get_anticommuter_for_generator(self, generator_index: int) -> list[Operation]:
        return [
            [X(self.data_qubits[0])],
            [X(self.data_qubits[1])],
            [X(self.data_qubits[3])],
            [X(self.data_qubits[7])],
            [X(self.data_qubits[i]) for i in range(3)],
            [X(self.data_qubits[i]) for i in (0, 3, 4)],
            [X(self.data_qubits[i]) for i in (1, 3, 5)],
            [X(self.data_qubits[i]) for i in (1, 7, 9)],
            [X(self.data_qubits[i]) for i in (3, 7, 11)],
            [X(self.data_qubits[i]) for i in (0, 7, 8)],
            [Z(self.data_qubits[0])],
            [Z(self.data_qubits[1])],
            [Z(self.data_qubits[3])],
            [Z(self.data_qubits[7])],
        ][generator_index]

    def _perform_get_operation_circuit(self, operation: LogicalOperation) -> Optional[Circuit]:
        if operation.gate == LogicalGateLabel.X:
            return Circuit(
                [X(qubit) for qubit in self.data_qubits[:7]]
            )
        elif operation.gate == LogicalGateLabel.Z:
            return Circuit(
                [Z(qubit) for qubit in self.data_qubits[:3]]
            )
        elif operation.gate == LogicalGateLabel.T:
            return Circuit(
                [inverse(T(qubit)) for qubit in self.data_qubits]
            )
        return None
