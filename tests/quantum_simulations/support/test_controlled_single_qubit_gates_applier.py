import pytest
from cirq import CX, Circuit, H, LineQubit, Simulator, X

from quantum_simulations.support.controlled_single_qubit_gates_applier import ControlledSingleQubitGatesApplier
from quantum_simulations.utilities.utilities import KET_ONE_STATE_VECTOR, KET_PLUS_STATE_VECTOR, KET_ZERO_STATE_VECTOR, \
    TYPE_STATE_VECTOR, \
    states_are_equal, tensor


class TestControlledSingleQubitGatesApplier:
    def test_trivial(self):
        applier = ControlledSingleQubitGatesApplier(operations=[], controls=[])
        circuit = applier.get_circuit()
        assert circuit == Circuit()

    @pytest.mark.parametrize('initial_control_state', [KET_ZERO_STATE_VECTOR, KET_ONE_STATE_VECTOR])
    def test_one_x_gate(self, initial_control_state: TYPE_STATE_VECTOR):
        qubits = LineQubit.range(2)
        applier = ControlledSingleQubitGatesApplier(operations=[X(qubits[0])], controls=[qubits[1]])
        circuit = applier.get_circuit()

        initial_state = tensor(KET_ZERO_STATE_VECTOR, initial_control_state)
        simulation = Simulator().simulate(circuit, initial_state=initial_state, qubit_order=qubits)
        expected_state = tensor(initial_control_state, initial_control_state)
        assert states_are_equal(simulation.final_state_vector, expected_state)

    def test_unequal_number_of_gates(self):
        qubits = LineQubit.range(3)
        applier = ControlledSingleQubitGatesApplier(operations=[X(qubit) for qubit in qubits[:1]], controls=qubits[1:])
        with pytest.raises(ValueError, match="^The number of gates \\(1\\) and controls\\(2\\) must be equal\\.$"):
            applier.get_circuit()

    def test_unequal_number_of_controls(self):
        qubits = LineQubit.range(3)
        applier = ControlledSingleQubitGatesApplier(operations=[X(qubit) for qubit in qubits[:2]], controls=qubits[2:])
        with pytest.raises(ValueError, match="^The number of gates \\(2\\) and controls\\(1\\) must be equal\\.$"):
            applier.get_circuit()

    def test_ensures_only_single_qubit_gates(self):
        qubits = LineQubit.range(2)
        applier = ControlledSingleQubitGatesApplier(operations=[CX(*qubits)], controls=qubits[:1])
        with pytest.raises(ValueError, match="^All operations must be single-qubit operations\\. Was given \\{cirq\\.CNOT\\(cirq\\.LineQubit\\(0\\), cirq\\.LineQubit\\(1\\)\\)\\}\\.$"):
            applier.get_circuit()

    @pytest.mark.parametrize('initial_control_state, expected_data_state', [
        (tensor(KET_ZERO_STATE_VECTOR, KET_ZERO_STATE_VECTOR), tensor(KET_ZERO_STATE_VECTOR, KET_ZERO_STATE_VECTOR)),
        (tensor(KET_ONE_STATE_VECTOR, KET_ONE_STATE_VECTOR), tensor(KET_PLUS_STATE_VECTOR, KET_ONE_STATE_VECTOR)),
    ])
    def test_two_gates_h_and_x(self, initial_control_state: TYPE_STATE_VECTOR, expected_data_state: TYPE_STATE_VECTOR):
        qubits = LineQubit.range(4)
        applier = ControlledSingleQubitGatesApplier(operations=[H(qubits[0]), X(qubits[1])], controls=qubits[2:])
        circuit = applier.get_circuit()

        initial_state = tensor(KET_ZERO_STATE_VECTOR, KET_ZERO_STATE_VECTOR, initial_control_state)
        simulation = Simulator().simulate(circuit, initial_state=initial_state, qubit_order=qubits)
        expected_state = tensor(expected_data_state, initial_control_state)
        assert states_are_equal(simulation.final_state_vector, expected_state)
