from abc import ABC, abstractmethod
from typing import Optional

from cirq import Circuit, LineQubit, Operation

from quantum_simulations.custom_dataclasses.check_matrix import CheckMatrix
from quantum_simulations.custom_dataclasses.correction_circuit import CorrectionCircuit
from quantum_simulations.custom_dataclasses.logical_operation import LogicalOperation
from quantum_simulations.error_correcting_codes.error_correcting_code.error_correcting_code import ErrorCorrectingCode
from quantum_simulations.support.error_recovery.error_recovery_by_check_matrix import ErrorRecoveryByCheckMatrix
from quantum_simulations.support.recovery_combinations_finder import RecoveryCombinationsFinder
from quantum_simulations.support.state_encoder.state_encoder_by_generator_measurement import \
    StateEncoderByGeneratorMeasurement


class StabilizerCode(ErrorCorrectingCode, ABC):
    def __init__(self,
                 check_matrix: CheckMatrix,
                 recovery_combinations_finder: Optional[RecoveryCombinationsFinder] = None,
                 qubits: Optional[list[LineQubit]] = None):
        self.check_matrix = check_matrix
        self.recovery_combinations_finder = recovery_combinations_finder or RecoveryCombinationsFinder(max_num_errors=1)
        super().__init__(num_data_qubits=self.check_matrix.num_physical_qubits,
                         num_logical_qubits=self.check_matrix.num_logical_qubits,
                         qubits=qubits)

    @abstractmethod
    def _get_anticommuter_for_generator(self, generator_index: int) -> list[Operation]:
        pass

    @abstractmethod
    def _perform_get_operation_circuit(self, operation: LogicalOperation) -> Optional[Circuit]:
        pass

    def encode_logical_qubit(self) -> Circuit:
        phase_corrections = [
            self._get_anticommuter_for_generator(generator_index=generator_index)
            for generator_index in range(len(self.check_matrix.matrix))
        ]
        return StateEncoderByGeneratorMeasurement(
            check_matrix=self.check_matrix,
            phase_corrections=phase_corrections,
            qubits=self.data_qubits,
        ).encode_state()

    def get_error_correction_circuit(self) -> CorrectionCircuit:
        return ErrorRecoveryByCheckMatrix(
            check_matrix=self.check_matrix,
            qubits=self.data_qubits,
            recovery_combinations_finder=self.recovery_combinations_finder
        ).get_error_correction_circuit()
