from functools import cached_property
from uuid import uuid4

from cirq import Circuit, CircuitOperation, LineQubit, ResetChannel, TaggedOperation, inverse

from quantum_simulations.conditions.verification_is_zero import VerificationIsZero
from quantum_simulations.globals.configuration_simulation_manager import ConfigurationSimulationManager
from quantum_simulations.support.cat_state_creator.cat_state_creator import CatStateCreator
from quantum_simulations.support.cat_state_creator.cat_state_creator_basic_nondeterministic.support.parity_verifier import \
    ParityVerifier
from quantum_simulations.support.cat_state_creator.cat_state_creator_cx_from_first_qubit import \
    CatStateCreatorCxFromFirstQubit
from quantum_simulations.utilities.measurement_key_with_stable_hash import MeasurementKeyWithStableHash


class CatStateCreatorBasicNondeterministic(CatStateCreator):
    def __init__(self, qubit_register: list[LineQubit]):
        super().__init__(qubit_register=qubit_register)

    def get_cat_state_circuit(self) -> Circuit:
        verification_condition = VerificationIsZero(key=MeasurementKeyWithStableHash(f'VERIFICATION_{uuid4().hex}'))
        preparation_circuit = Circuit(
            ResetChannel().on_each(*self._qubit_register),
            self._state_propagation,
            self._parity_verifier(target_qubits=self._qubit_register, measurement_key=verification_condition.key).validate_parity(),
        )
        return Circuit(
            TaggedOperation(
                CircuitOperation(preparation_circuit.freeze(),
                                 use_repetition_ids=False,
                                 repeat_until=verification_condition if len(self._qubit_register) > 1 else None),
            'CAT_STATE_CREATOR_BASIC_NONDETERMINISTIC'
            )
        )

    def decode_state(self) -> Circuit:
        return inverse(self._state_propagation)

    @cached_property
    def _state_propagation(self) -> Circuit:
        return CatStateCreatorCxFromFirstQubit(qubit_register=self._qubit_register).get_cat_state_circuit()

    @property
    def _parity_verifier(self) -> type[ParityVerifier]:
        return ConfigurationSimulationManager().get_configuration().parity_verifier_type
