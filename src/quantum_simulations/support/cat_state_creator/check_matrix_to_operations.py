from cirq import LineQubit, Operation

from quantum_simulations.custom_dataclasses.check_matrix import CheckMatrix
from quantum_simulations.support.cat_state_creator.check_matrix_to_gates import CheckMatrixToGates


class CheckMatrixToOperations:
    def __init__(self, check_matrix: CheckMatrix, qubits: list[LineQubit]):
        self._check_matrix = check_matrix
        self._qubits = qubits

    def get_operations(self) -> list[list[Operation]]:
        generator_gates = CheckMatrixToGates(check_matrix=self._check_matrix).get_gates()
        return [[gate(self._qubits[target_index])
                 for target_index, gates in enumerate(qubit_gates)
                 for gate in gates]
                for qubit_gates in generator_gates]
