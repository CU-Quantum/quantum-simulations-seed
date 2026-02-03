from functools import cached_property

import numpy as np
import pytest
from cirq import Circuit, CircuitOperation, LineQubit, NoiseModel, OP_TREE, Operation
from cirq.testing import SingleQubitGate

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
        noise_channel = BitFlipNumberOfTimesNoiseModel(num_times_to_flip=0)
        circuit = self._circuit_constructing_and_verifying_3_qubit_cat_state
        circuit_noisy = circuit.with_noise(noise_channel)
        assert self._successfully_created_3_qubit_cat_state(circuit=circuit_noisy)
        assert noise_channel.bit_flip_number_of_times_channel.num_times_ran == 1

    def test_retries_if_invalid(self):
        noise_channel = BitFlipNumberOfTimesNoiseModel(num_times_to_flip=1)
        circuit = self._circuit_constructing_and_verifying_3_qubit_cat_state
        circuit_noisy = circuit.with_noise(noise_channel)
        assert self._successfully_created_3_qubit_cat_state(circuit=circuit_noisy)
        assert noise_channel.bit_flip_number_of_times_channel.num_times_ran == 2

    def _successfully_created_3_qubit_cat_state(self, circuit: Circuit):
        initial_state = tensor(*[KET_ZERO_STATE_VECTOR] * self._num_qubits)
        circuit_simulator = CircuitSimulatorStateVector()

        state = circuit_simulator.run_simulation(circuit=circuit,
                                                 num_data_qubits=self._num_qubits,
                                                 initial_data_state=initial_state)

        expected_target_qubits_state = get_cat_state_vector(num_qubits=self._num_qubits)
        return states_are_equal(state.state, tensor(expected_target_qubits_state))


class BitFlipNumberOfTimesChannel(SingleQubitGate):
    def __init__(self, num_times_to_flip: int):
        super().__init__()
        self.num_times_ran = 0
        self._num_times_to_flip = num_times_to_flip

    def _unitary_(self):
        has_run_enough_times = self.num_times_ran >= self._num_times_to_flip
        self.num_times_ran += 1
        if has_run_enough_times:
            return np.array([[1, 0], [0, 1]])
        else:
            return np.array([[0, 1], [1, 0]])


class BitFlipNumberOfTimesNoiseModel(NoiseModel):
    def __init__(self, num_times_to_flip: int):
        super().__init__()
        self._num_times_to_flip = num_times_to_flip

    def noisy_operation(self, operation: Operation) -> OP_TREE:
        circuit = operation.untagged.circuit.unfreeze()
        circuit.insert(1, self.bit_flip_number_of_times_channel.on(operation.qubits[1]))
        return CircuitOperation(circuit.freeze(), use_repetition_ids=False, repeat_until=operation.untagged.repeat_until)

    @cached_property
    def bit_flip_number_of_times_channel(self) -> BitFlipNumberOfTimesChannel:
        return BitFlipNumberOfTimesChannel(num_times_to_flip=self._num_times_to_flip)
