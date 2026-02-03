from cirq import Circuit, CircuitOperation, FrozenCircuit, M, OP_TREE, TaggedOperation

from quantum_simulations.conditions.majority_vote import MajorityVote
from quantum_simulations.conditions.multiple_conditions import MultipleConditions
from quantum_simulations.globals.active_encodings_store import ActiveEncodingsStore
from quantum_simulations.globals.configuration_simulation_manager import ConfigurationSimulationManager
from quantum_simulations.globals.fresh_ancillas_pool import FreshAncillasPool
from quantum_simulations.support.measurer.measurer import FAULT_TOLERANT_MEASURER_TAG, Measurer
from quantum_simulations.support.operations_applier.operations_applier_using_cat_state import \
    OperationsApplierUsingCatStateControl


class FaultTolerantMeasurerSequential(Measurer):
    def get_measurement_circuit(self) -> Circuit:
        with FreshAncillasPool().use_fresh_ancillas(num_ancillas=1) as ancilla_qubits:
            measurement_qubit = ancilla_qubits[0]
            appliers = [
                OperationsApplierUsingCatStateControl(operations=operations, measurement_qubit=measurement_qubit)
                for operations in self._observables
            ]
            conditions = [
                MajorityVote(desired_measurement_key=measurement_key)
                for measurement_key in self._measurement_keys
            ]
            one_repetition = [
                [
                    applier.get_application_circuit(),
                    M(measurement_qubit, key=condition.key),
                ] for applier, condition in zip(appliers, conditions)
            ]
            all_repetitions = [one_repetition for _ in range(self._majority_vote_repetitions)]
            if self._correction_between_repetitions:
                all_repetitions = self._add_correction_circuits_between_elements(all_repetitions=all_repetitions)

            return Circuit(
                TaggedOperation(
                    CircuitOperation(
                        FrozenCircuit(all_repetitions),
                        use_repetition_ids=False,
                        repeat_until=MultipleConditions(conditions),
                    ),
                    FAULT_TOLERANT_MEASURER_TAG,
                )
            )

    def _add_correction_circuits_between_elements(self, all_repetitions: list[OP_TREE]) -> OP_TREE:
        with ActiveEncodingsStore(additional_tracked_encodings=[]) as encodings_store:
            interspersed = [encodings_store.get_all_correction_circuits()
                            for _ in range(len(all_repetitions) * 2 - 1)]
            for i, repetition in enumerate(all_repetitions):
                interspersed[i * 2] = repetition
            return interspersed

    @property
    def _majority_vote_repetitions(self) -> int:
        return ConfigurationSimulationManager().get_configuration().majority_vote_repetitions
