from typing import Optional

from cirq import Circuit, LineQubit

from quantum_simulations.custom_dataclasses.correction_circuit import CorrectionCircuit
from quantum_simulations.custom_dataclasses.logical_operation import LogicalOperation
from quantum_simulations.error_correcting_codes.error_correcting_code.error_correcting_code import ErrorCorrectingCode
from quantum_simulations.error_correcting_codes.generalized_shor_code.generalized_shor_code import GeneralizedShorCode


class ShorsRepetitionCode(ErrorCorrectingCode):
    def __init__(self, qubits: Optional[list[LineQubit]] = None,):
        self._alias = GeneralizedShorCode(num_cats=3, num_qubits_per_cat=3, qubits=qubits)
        super().__init__(num_data_qubits=len(self._alias.data_qubits),
                         num_logical_qubits=self._alias.num_logical_qubits,
                         qubits=self._alias.data_qubits)

    def encode_logical_qubit(self) -> Circuit:
        return self._alias.encode_logical_qubit()

    def get_operation_circuit(self, operation: LogicalOperation) -> Circuit:
        return self._alias.get_operation_circuit(operation=operation)

    def _perform_get_operation_circuit(self, operation: LogicalOperation) -> None:
        pass

    def get_error_correction_circuit(self) -> CorrectionCircuit:
        return self._alias.get_error_correction_circuit()
