from typing import List

from cirq import Gate, X, Z

from quantum_simulations.custom_dataclasses.check_matrix import CheckMatrix


class CheckMatrixToGates:
    def __init__(self, check_matrix: CheckMatrix) -> None:
        self._check_matrix = check_matrix

    def get_gates(self) -> List[List[List[Gate]]]:
        return [[self._get_operations(row=row, qubit_index=qubit_index) for qubit_index in range(self._check_matrix.num_physical_qubits)]
                for row in self._check_matrix.matrix]

    def _get_operations(self, row: List[bool], qubit_index: int) -> List[Gate]:
        operations = []
        if row[qubit_index]:
            is_negative = row[qubit_index] == -1
            if is_negative:
                operations.append(Z)
            operations.append(X)
            if is_negative:
                operations.append(Z)
        if row[qubit_index + self._check_matrix.num_physical_qubits]:
            is_negative = row[qubit_index + self._check_matrix.num_physical_qubits] == -1
            if is_negative:
                operations.append(X)
            operations.append(Z)
            if is_negative:
                operations.append(X)
        return operations
