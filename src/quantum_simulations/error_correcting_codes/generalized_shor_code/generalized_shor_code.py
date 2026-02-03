from typing import Optional

from cirq import Circuit, CircuitOperation, FrozenCircuit, TaggedOperation, X, Z

from quantum_simulations.custom_dataclasses.correction_circuit import CorrectionCircuit
from quantum_simulations.custom_dataclasses.logical_operation import LogicalGateLabel, LogicalOperation
from quantum_simulations.error_correcting_codes.generalized_shor_code_hadamard.generalized_shor_code_hadamard import \
    GeneralizedShorCodeHadamard
from quantum_simulations.globals.configuration_simulation_manager import ConfigurationSimulationManager
from quantum_simulations.support.cat_state_creator.cat_state_creator import CatStateCreator
from quantum_simulations.support.operations_applier.operations_applier import DELAYED_NOISE_TAG

GENERALIZED_SHOR_ENCODING_TAG = 'GENERALIZED_SHOR_ENCODING'
GENERALIZED_SHOR_CAT_CREATING_TAG = 'GENERALIZED_SHOR_CAT_CREATING'


SYNDROME_TAG = 'SYNDROME'
RECOVERY_TAG = 'RECOVERY'


def syndrome_then_recovery_circuit(correction_circuit: CorrectionCircuit) -> Circuit:
    return Circuit(
        TaggedOperation(
            CircuitOperation(
                FrozenCircuit(correction_circuit.syndrome_circuit),
            ),
            SYNDROME_TAG, DELAYED_NOISE_TAG
        ),
        TaggedOperation(
            CircuitOperation(
                FrozenCircuit(correction_circuit.recovery_circuit),
            ),
            RECOVERY_TAG, DELAYED_NOISE_TAG
        ),
    )


class GeneralizedShorCode(GeneralizedShorCodeHadamard):
    def encode_logical_qubit(self) -> Circuit:
        return Circuit(
            TaggedOperation(
                CircuitOperation(
                    FrozenCircuit(
                        TaggedOperation(
                            CircuitOperation(
                                FrozenCircuit(
                                    self._cat_state_creator_type(qubit_register=subregister).get_cat_state_circuit()
                                    for subregister in self.subregisters
                                )
                            ),
                            GENERALIZED_SHOR_CAT_CREATING_TAG,
                        ),
                        syndrome_then_recovery_circuit(correction_circuit=self.get_error_correction_circuit()),
                    ),
                ),
                GENERALIZED_SHOR_ENCODING_TAG
            ),
        )

    def _perform_get_operation_circuit(self, operation: LogicalOperation) -> Optional[Circuit]:
        if operation.gate == LogicalGateLabel.X:
            return Circuit(
                [Z(self.data_qubits[i * self._num_qubits_per_cat]) for i in range(self._num_cats)]
            )
        elif operation.gate == LogicalGateLabel.Z:
            return Circuit(
                [X(self.data_qubits[i]) for i in range(self._num_qubits_per_cat)]
            )
        return None

    @property
    def _cat_state_creator_type(self) -> type[CatStateCreator]:
        return ConfigurationSimulationManager().get_configuration().cat_state_creator_type
