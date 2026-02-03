from cirq import Circuit, CircuitOperation, FrozenCircuit, M, TaggedOperation

from quantum_simulations.globals.fresh_ancillas_pool import FreshAncillasPool
from quantum_simulations.support.measurer.measurer import MEASURER_WITH_SINGLE_QUBIT_TAG, Measurer
from quantum_simulations.support.operations_applier.operations_applier import DELAYED_NOISE_TAG
from quantum_simulations.support.operations_applier.operations_applier_using_single_qubit_hadamard_control import \
    OperationsApplierUsingSingleQubitHadamardControl


class MeasurerWithSingleQubitSequential(Measurer):
    def get_measurement_circuit(self) -> Circuit:
        if not self._observables:
            return Circuit()
        with FreshAncillasPool().use_fresh_ancillas(num_ancillas=1) as ancilla_qubits:
            measuring_qubit = ancilla_qubits[0]
            operations = [
                [
                    OperationsApplierUsingSingleQubitHadamardControl(
                        operations=operations,
                        measurement_qubit=measuring_qubit,
                    ).get_application_circuit(),
                    M(measuring_qubit, key=measurement_key),
                ] for operations, measurement_key in zip(self._observables, self._measurement_keys)
            ]
            return Circuit(
                TaggedOperation(
                    CircuitOperation(
                        FrozenCircuit(operations),
                    ),
                    MEASURER_WITH_SINGLE_QUBIT_TAG, DELAYED_NOISE_TAG
                )
            )
