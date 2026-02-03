import pytest
from cirq import Circuit

from quantum_simulations.custom_dataclasses.logical_operation import LogicalGateLabel, LogicalOperation
from quantum_simulations.error_correcting_codes.stabilizer_standardized_code.stabilizer_standardized_code import \
    StabilizerStandardizedCode
from quantum_simulations.globals.fresh_ancillas_pool import FreshAncillasPool
from quantum_simulations.predefined_check_matrix_values import get_check_matrix_values_4_qubit, \
    get_check_matrix_values_5_qubit
from quantum_simulations.simulators.circuit_simulator import CircuitSimulatorStateVector
from quantum_simulations.utilities.utilities import KET_ONE_STATE_VECTOR, KET_ZERO_STATE_VECTOR, states_are_equal, \
    tensor


class TestStabilizerStandardizedCode:
    def test_logical_x_on_one_out_of_multiple_encoded_qubits(self):
        code = StabilizerStandardizedCode(generators=get_check_matrix_values_4_qubit())
        FreshAncillasPool().set_first_ancilla_num(first_ancilla_num=len(code.data_qubits))
        circuit = Circuit(
            code.encode_logical_qubit(),
            code.get_operation_circuit(operation=LogicalOperation(gate=LogicalGateLabel.X, qubit_index=1))
        )

        initial_data_state = tensor(*[KET_ZERO_STATE_VECTOR] * len(code.data_qubits))
        circuit_simulator = CircuitSimulatorStateVector()
        state_and_measurements = circuit_simulator.run_simulation(circuit=circuit,
                                                                  num_data_qubits=len(code.data_qubits),
                                                                  initial_data_state=initial_data_state)

        expected_state = circuit_simulator.run_simulation(
            circuit=Circuit(
                code.encode_logical_qubit(),
            ),
            num_data_qubits=len(code.data_qubits),
            initial_data_state=tensor(*[KET_ZERO_STATE_VECTOR] * 3, KET_ONE_STATE_VECTOR)
        ).state
        assert states_are_equal(state_and_measurements.state, expected_state)

    def test_qubit_index_must_be_at_most_largest_logical_index(self):
        code = StabilizerStandardizedCode(generators=get_check_matrix_values_4_qubit())
        with pytest.raises(ValueError, match="Qubit index must be between 0 and 1. Was given 2."):
            code.get_operation_circuit(operation=LogicalOperation(gate=LogicalGateLabel.H, qubit_index=2))

        code = StabilizerStandardizedCode(generators=get_check_matrix_values_5_qubit())
        with pytest.raises(ValueError, match="Qubit index must be between 0 and 0. Was given 1."):
            code.get_operation_circuit(operation=LogicalOperation(gate=LogicalGateLabel.H, qubit_index=1))
