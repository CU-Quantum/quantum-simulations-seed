from cirq import Circuit, M, ResetChannel, X

from cirq_experiments.custom_dataclasses.configuration_error_correcing_code import ConfigurationErrorCorrectingCode
from cirq_experiments.support.cat_state_creator.cat_state_creator_basic_nondeterministic.support.parity_verifier import \
    ParityVerifier
from cirq_experiments.support.measurer.measurer import Measurer
from cirq_experiments.globals.error_correcting_code_configuration import ConfigurationErrorCorrectingCodeManager
from cirq_experiments.globals.fresh_ancillas_pool import FreshAncillasPool


class ParityVerifierParallel(ParityVerifier):
    def validate_parity(self) -> Circuit:
        if not self._target_qubits:
            return Circuit()
        with FreshAncillasPool().use_fresh_ancillas(num_ancillas=self._num_target_qubits - 1) as ancillas:
            return Circuit(
                ResetChannel().on_each(*ancillas),
                [
                    [
                        X(verifier_ancilla).controlled_by(self._target_qubits[i]),
                        X(verifier_ancilla).controlled_by(self._target_qubits[i + 1]),
                        M(verifier_ancilla, key=self._measurement_key),
                    ] for i, verifier_ancilla in enumerate(ancillas)
                ],
            )

    @property
    def _num_target_qubits(self) -> int:
        return len(self._target_qubits)

    @property
    def _measurer(self) -> type[Measurer]:
        return self._configuration.measurer_type

    @property
    def _configuration(self) -> ConfigurationErrorCorrectingCode:
        return ConfigurationErrorCorrectingCodeManager().get_configuration()
