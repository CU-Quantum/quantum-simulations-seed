from typing import Optional

from cirq import Circuit, I, LineQubit, Operation, X, Z
from numpy import array

from quantum_simulations.custom_dataclasses.check_matrix import CheckMatrix
from quantum_simulations.custom_dataclasses.correction_circuit import CorrectionCircuit
from quantum_simulations.custom_dataclasses.logical_operation import LogicalGateLabel, LogicalOperation
from quantum_simulations.error_correcting_codes.error_correcting_code.error_correcting_code import ErrorCorrectingCode
from quantum_simulations.generalized_shor_code_generators import GeneralizedShorCodeGenerators
from quantum_simulations.support.error_recovery.error_recovery_by_check_matrix import ErrorRecoveryByCheckMatrix
from quantum_simulations.support.recovery_combinations_finder import RecoveryCombinationsFinder
from quantum_simulations.support.state_encoder.state_encoder_by_generator_measurement import \
    StateEncoderByGeneratorMeasurement


class RepetitionCodeOneLogical(ErrorCorrectingCode):
    def __init__(self, num_qubits: int, qubits: Optional[list[LineQubit]] = None):
        self.check_matrix = None
        self.recovery_combinations_finder = RecoveryCombinationsFinder(max_num_errors=1)
        if num_qubits >= 2:
            z_stabilizers = GeneralizedShorCodeGenerators(num_qubits_per_cat=num_qubits, num_cats=1).get_z_generators()
            self.check_matrix = CheckMatrix(matrix=array(z_stabilizers))
        super().__init__(num_data_qubits=num_qubits,
                         num_logical_qubits=1,
                         qubits=qubits)

    def encode_logical_qubit(self) -> Circuit:
        if self.check_matrix is None:
            return self._empty_circuit
        phase_corrections = [
            self._get_anticommuter_for_generator(generator_index=generator_index)
            for generator_index in range(len(self.check_matrix.matrix))
        ]
        return StateEncoderByGeneratorMeasurement(
            check_matrix=self.check_matrix,
            phase_corrections=phase_corrections,
            qubits=self.data_qubits,
        ).encode_state()

    def _get_anticommuter_for_generator(self, generator_index: int) -> list[Operation]:
        return [X(self.data_qubits[qubit_index]) for qubit_index in range(generator_index + 1)]

    def get_error_correction_circuit(self) -> CorrectionCircuit:
        if self.check_matrix is None:
            return CorrectionCircuit(
                syndrome_circuit=self._empty_circuit,
                recovery_circuit=self._empty_circuit,
            )
        return ErrorRecoveryByCheckMatrix(
            check_matrix=self.check_matrix,
            qubits=self.data_qubits,
            recovery_combinations_finder=RecoveryCombinationsFinder(max_num_x_errors=self._num_data_qubits // 2, max_num_z_errors=0)
        ).get_error_correction_circuit()

    def _perform_get_operation_circuit(self, operation: LogicalOperation) -> Optional[Circuit]:
        if operation.gate == LogicalGateLabel.Z:
            return Circuit(Z(self.data_qubits[0]))
        if operation.gate == LogicalGateLabel.X:
            return Circuit(X(qubit) for qubit in self.data_qubits)
        return None

    @property
    def _empty_circuit(self) -> Circuit:
        return Circuit(I(qubit) for qubit in self.data_qubits)
