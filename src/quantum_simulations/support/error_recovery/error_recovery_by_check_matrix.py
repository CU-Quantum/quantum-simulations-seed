from typing import Optional

from cirq import LineQubit

from quantum_simulations.custom_dataclasses.check_matrix import CheckMatrix
from quantum_simulations.custom_dataclasses.correction_circuit import CorrectionCircuit
from quantum_simulations.custom_dataclasses.recovery import RecoveryOperation
from quantum_simulations.support.cat_state_creator.check_matrix_to_operations import CheckMatrixToOperations
from quantum_simulations.support.cat_state_creator.recovery_finder import RecoveryFinder
from quantum_simulations.support.error_recovery.error_recovery_by_stabilizers import ErrorRecoveryByStabilizers
from quantum_simulations.support.recovery_combinations_finder import RecoveryCombinationsFinder


class ErrorRecoveryByCheckMatrix:
    def __init__(self,
                 check_matrix: CheckMatrix,
                 qubits: list[LineQubit],
                 recovery_combinations_finder: Optional[RecoveryCombinationsFinder]):
        self._check_matrix = check_matrix
        self._qubits = qubits
        self._recovery_combinations_finder = recovery_combinations_finder

    def get_error_correction_circuit(self) -> CorrectionCircuit:
        generator_operations = CheckMatrixToOperations(check_matrix=self._check_matrix, qubits=self._qubits).get_operations()
        return ErrorRecoveryByStabilizers(
            stabilizers=generator_operations,
            recoveries=self._recoveries,
        ).get_error_correction_circuit()

    @property
    def _recoveries(self) -> list[RecoveryOperation]:
        recoveries = RecoveryFinder(check_matrix=self._check_matrix).find_recovery_operations(qubits=self._qubits)
        return self._recovery_combinations_finder.find_recovery_operations(single_error_recoveries=recoveries)
