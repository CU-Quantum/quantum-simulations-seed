from cirq import Circuit, CircuitOperation, H, R, TaggedOperation

from quantum_simulations.support.operations_applier.operations_applier import DELAYED_NOISE_TAG, OperationsApplier


class OperationsApplierUsingSingleQubitHadamardControl(OperationsApplier):
    def _perform_get_application_circuit(self) -> Circuit:
        return Circuit(
            R(self._measurement_qubit),
            H(self._measurement_qubit),
            TaggedOperation(
                CircuitOperation(
                    Circuit(
                        [operation.controlled_by(self._measurement_qubit) for operation in self._operations]
                    ).freeze(),
                ),
                DELAYED_NOISE_TAG
            ),
            H(self._measurement_qubit),
        )
