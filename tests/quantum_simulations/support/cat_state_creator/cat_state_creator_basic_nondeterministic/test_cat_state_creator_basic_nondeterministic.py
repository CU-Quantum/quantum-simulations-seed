import pytest
from cirq import Circuit, CircuitOperation, Gate, I, LineQubit, NoiseModel, OP_TREE, Operation, X

from quantum_simulations.conditions.verification_is_zero import VerificationIsZero
from quantum_simulations.globals.fresh_ancillas_pool import FreshAncillasPool
from quantum_simulations.simulators.circuit_simulator import CircuitSimulatorStateVector
from quantum_simulations.support.cat_state_creator.cat_state_creator_basic_nondeterministic.cat_state_creator_basic_nondeterministic import \
    CatStateCreatorBasicNondeterministic
from quantum_simulations.utilities.utilities import KET_ZERO_STATE_VECTOR, states_are_equal, tensor
from tests.quantum_simulations.utilities_for_tests import get_cat_state_vector


class TestCatStateCreatorBasicNondeterministic:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self._num_qubits = 3
        self._qubits = LineQubit.range(self._num_qubits)
        FreshAncillasPool().set_first_ancilla_num(first_ancilla_num=self._num_qubits)

        preparer = CatStateCreatorBasicNondeterministic(qubit_register=self._qubits)
        self._circuit_constructing_and_verifying_3_qubit_cat_state = preparer.get_cat_state_circuit()

    def test_creates_cat_state(self):
        circuit = self._circuit_constructing_and_verifying_3_qubit_cat_state
        assert self._successfully_created_3_qubit_cat_state(circuit=circuit)
        assert self._number_of_repetitions(circuit=circuit) == 1

    def test_retries_if_invalid(self):
        circuit = self._circuit_constructing_and_verifying_3_qubit_cat_state
        circuit_noisy = circuit.with_noise(BitFlipOnceNoiseModel())
        assert self._successfully_created_3_qubit_cat_state(circuit=circuit_noisy)
        assert self._number_of_repetitions(circuit=circuit) == 2

    def _successfully_created_3_qubit_cat_state(self, circuit: Circuit):
        initial_state = tensor(*[KET_ZERO_STATE_VECTOR] * self._num_qubits)
        circuit_simulator = CircuitSimulatorStateVector()

        state = circuit_simulator.run_simulation(circuit=circuit,
                                                 num_data_qubits=self._num_qubits,
                                                 initial_data_state=initial_state)

        expected_target_qubits_state = get_cat_state_vector(num_qubits=self._num_qubits)
        return states_are_equal(state.state, tensor(expected_target_qubits_state))

    def _number_of_repetitions(self, circuit: Circuit) -> int:
        number_of_measurements_per_repetition = 2
        operation: CircuitOperation = circuit.moments[0].operations[0]
        resolver: VerificationIsZero = operation.untagged.repeat_until
        return resolver._last_num_measurements // number_of_measurements_per_repetition


class BitFlipOnceNoiseModel(NoiseModel):
    def __init__(self):
        super().__init__()
        self._added_noise = False

    def noisy_operation(self, operation: Operation) -> OP_TREE:
        circuit = operation.untagged.circuit.unfreeze()
        circuit.insert(1, BitFlipOnceChannel().on(operation.qubits[1]))
        return CircuitOperation(circuit.freeze(), use_repetition_ids=False, repeat_until=operation.untagged.repeat_until),


class BitFlipOnceChannel(Gate):
    def __init__(self):
        super().__init__()
        self._caused_bit_flip = False

    def _num_qubits_(self) -> int:
        return 1

    def _decompose_(self, qubits):
        target_qubit = qubits[0]
        if self._caused_bit_flip:
            yield I(target_qubit)
        else:
            yield X(target_qubit)
            self._caused_bit_flip = True
