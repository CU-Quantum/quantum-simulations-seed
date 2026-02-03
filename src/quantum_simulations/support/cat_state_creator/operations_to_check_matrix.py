from cirq import Operation, X, Y, Z
from numpy import array

from quantum_simulations.custom_dataclasses.check_matrix import CheckMatrix


class OperationsToCheckMatrix:
    def __init__(self, operations_list: list[list[Operation]]):
        self._operations_list = operations_list

    def get_check_matrix(self):
        qubits = {operation.qubits[0] for operations in self._operations_list for operation in operations}
        qubit_order = list(sorted(qubits))
        num_qubits = len(qubit_order)
        check_matrix = []
        for operations in self._operations_list:
            row_x = [0] * num_qubits
            row_z = [0] * num_qubits
            for operation in operations:
                active_qubit = operation.qubits[0]
                qubit_index = qubit_order.index(active_qubit)
                if operation.gate in [X, Y]:
                    row_x[qubit_index] = 1
                if operation.gate in [Z, Y]:
                    row_z[qubit_index] = 1
            check_matrix.append(row_x + row_z)
        return CheckMatrix(matrix=array(check_matrix))

