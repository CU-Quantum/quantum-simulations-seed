from cirq import Circuit, I, LineQubit, Simulator, X

from quantum_simulations.simulators.circuit_simulator import CircuitSimulatorStateVector
from quantum_simulations.support.cat_state_creator.cat_state_creator_flag_pattern.cat_state_creator_flag_pattern import \
    CatStateCreatorFlagPattern
from quantum_simulations.utilities.utilities import TYPE_STATE_VECTOR, get_num_qubits_in_state, states_are_equal
from tests.quantum_simulations.utilities_for_tests import get_cat_state_vector


def circuit_results_in_expected_state(circuit: Circuit, expected_state: TYPE_STATE_VECTOR) -> bool:
    num_data_qubits = get_num_qubits_in_state(expected_state)
    simulation = CircuitSimulatorStateVector().run_simulation(circuit=circuit, num_data_qubits=num_data_qubits)
    return states_are_equal(simulation.state, expected_state)

def get_circuit_with_x_error_on_first_n_qubits(qubits: list[LineQubit], n: int) -> Circuit:
    control_qubit = qubits[0]
    first_error_qubit = qubits[n - 1]
    circuit = CatStateCreatorFlagPattern(qubit_register=qubits).get_cat_state_circuit()
    faulty_moment = next(i for i, moment in enumerate(circuit.moments) if first_error_qubit in moment.qubits)
    circuit.insert(faulty_moment, X(control_qubit))
    return circuit

def get_cat_state_with_x_error(num_qubits: int, qubit_index_with_error: int) -> TYPE_STATE_VECTOR:
    ideal_state = get_cat_state_vector(num_qubits=num_qubits)
    circuit = Circuit(
        [I(LineQubit(i)) for i in range(num_qubits)],
        X(LineQubit(qubit_index_with_error))
    )
    return Simulator().simulate(circuit, initial_state=ideal_state).final_state_vector
