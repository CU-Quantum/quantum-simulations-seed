from cirq import LineQubit, X, Y, Z

from quantum_simulations.custom_dataclasses.check_matrix import CheckMatrix
from quantum_simulations.custom_dataclasses.recovery import RecoveryGate, RecoveryOperation
from quantum_simulations.utilities.utilities import binary_array_to_int


class RecoveryFinder:
    def __init__(self, check_matrix: CheckMatrix):
        self._check_matrix = check_matrix

    def find_recovery_operations(self, qubits: list[LineQubit]) -> list[RecoveryOperation]:
        recovery_gates = self.find_recovery_gates()
        return [
            RecoveryOperation(
                operation=recovery_gates.gate(qubits[recovery_gates.qubit_index]),
                symptom=recovery_gates.symptom
            )
            for recovery_gates in recovery_gates
        ]

    def find_recovery_gates(self) -> list[RecoveryGate]:
        possible_errors = [X, Z]
        transposed_check_matrix = self._check_matrix.matrix.transpose()
        x_or_z_recoveries = [RecoveryGate(gate=possible_errors[column_index < self._check_matrix.num_physical_qubits],
                                          qubit_index=column_index % self._check_matrix.num_physical_qubits,
                                          symptom=syndrome.tolist())
                             for column_index, syndrome in enumerate(transposed_check_matrix)]
        y_recoveries = [RecoveryGate(gate=Y,
                                     qubit_index=column_index,
                                     symptom=self._get_y_symptom(column_index=column_index))
                        for column_index in range(self._check_matrix.num_physical_qubits)]
        one_recovery_per_symptom = {binary_array_to_int(recovery.symptom): recovery
                                    for recovery in x_or_z_recoveries + y_recoveries if any(recovery.symptom)}
        return list(one_recovery_per_symptom.values())

    def _get_y_symptom(self, column_index: int) -> list[int]:
        transposed_check_matrix = self._check_matrix.matrix.transpose()
        reciprocal_column = column_index + self._check_matrix.num_physical_qubits
        if any(transposed_check_matrix[column_index]) and any(transposed_check_matrix[reciprocal_column]):
            return (transposed_check_matrix[column_index] ^ transposed_check_matrix[reciprocal_column]).tolist()
        return []
