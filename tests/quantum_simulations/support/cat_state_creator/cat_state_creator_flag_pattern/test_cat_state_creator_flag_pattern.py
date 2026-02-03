from cirq import Circuit, LineQubit, Simulator, StateVectorTrialResult

from quantum_simulations.support.cat_state_creator.cat_state_creator_flag_pattern.cat_state_creator_flag_pattern import \
    CatStateCreatorFlagPattern
from quantum_simulations.utilities.utilities import states_are_equal
from tests.quantum_simulations.support.cat_state_creator.cat_state_creator_flag_pattern.utilities import \
    circuit_results_in_expected_state
from tests.quantum_simulations.utilities_for_tests import get_cat_state_vector


class TestCatStateCreatorFlagPattern:
    def test_trivial(self):
        qubits = LineQubit.range(0)
        circuit = CatStateCreatorFlagPattern(qubit_register=qubits).get_cat_state_circuit()
        assert circuit == Circuit()

    def test_single_qubit(self):
        qubits = LineQubit.range(1)
        circuit = CatStateCreatorFlagPattern(qubit_register=qubits).get_cat_state_circuit()
        assert circuit_results_in_expected_state(circuit=circuit, expected_state=KET_PLUS_STATE_VECTOR)

    def test_three_qubits(self):
        num_qubits = 3
        qubits = LineQubit.range(num_qubits)
        circuit = CatStateCreatorFlagPattern(qubit_register=qubits).get_cat_state_circuit()
        expected_state = get_cat_state_vector(num_qubits=num_qubits)
        simulation: StateVectorTrialResult = Simulator().simulate(circuit)
        assert states_are_equal(simulation.final_state_vector, expected_state)
