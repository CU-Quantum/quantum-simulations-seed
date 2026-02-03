from cirq import Circuit

from quantum_simulations.error_correcting_codes.stabilizer_standardized_code.stabilizer_standardized_code import \
    StabilizerStandardizedCode
from quantum_simulations.support.state_encoder.state_encoder_gottesman import StateEncoderGottesman


class StabilizerStandardizedCodeGottesmanEncoding(StabilizerStandardizedCode):
    def encode_logical_qubit(self) -> Circuit:
        return StateEncoderGottesman(check_matrix_standardized=self._check_matrix_standardized,
                                     data_qubits=self.data_qubits).get_encoding_circuit()
