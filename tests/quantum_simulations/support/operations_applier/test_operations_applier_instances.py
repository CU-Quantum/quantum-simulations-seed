import pytest
from cirq import Circuit, LineQubit, X

from quantum_simulations.globals.fresh_ancillas_pool import FreshAncillasPool
from quantum_simulations.simulators.circuit_simulator import CircuitSimulatorStateVector
from quantum_simulations.support.operations_applier.operations_applier_using_cat_state import \
    OperationsApplierUsingCatStateControl
from quantum_simulations.support.operations_applier.operations_applier_using_single_qubit_hadamard_control import \
    OperationsApplierUsingSingleQubitHadamardControl
from quantum_simulations.utilities.utilities import KET_MINUS_STATE_VECTOR, KET_ONE_STATE_VECTOR, KET_ZERO_STATE_VECTOR, \
    states_are_equal, tensor


class TestOperationsApplierInstances:
    @pytest.fixture(autouse=True, params=[
        pytest.param(OperationsApplierUsingSingleQubitHadamardControl, id='OperationsApplierUsingSingleQubitControl'),
        pytest.param(OperationsApplierUsingCatStateControl, id='OperationsApplierUsingCatStateControl'),
    ])
    def _set_operation_applier_instance(self, request):
        self._operation_applier_type = request.param

    def test_trivial(self):
        applier = OperationsApplierUsingSingleQubitHadamardControl(operations=[], measurement_qubit=LineQubit(0))
        circuit = applier.get_application_circuit()
        assert circuit == Circuit()

    def test_one_operation(self):
        assert self._operations_result_in_expected_state(num_data_qubits=1)

    def test_two_operations(self):
        assert self._operations_result_in_expected_state(num_data_qubits=2)

    def _operations_result_in_expected_state(self, num_data_qubits: int) -> bool:
        data_qubits = LineQubit.range(num_data_qubits)
        operations = [X(qubit) for qubit in data_qubits]
        measurement_qubit = LineQubit(num_data_qubits)
        num_qubits_with_measure = len(data_qubits) + 1
        FreshAncillasPool().set_first_ancilla_num(first_ancilla_num=num_qubits_with_measure)

        applier = self._operation_applier_type(operations=operations, measurement_qubit=measurement_qubit)
        circuit = applier.get_application_circuit()

        initial_state = tensor(*[KET_MINUS_STATE_VECTOR] * num_data_qubits, KET_ZERO_STATE_VECTOR)
        circuit_simulator = CircuitSimulatorStateVector()
        simulation = circuit_simulator.run_simulation(circuit=circuit,
                                                      num_data_qubits=num_qubits_with_measure,
                                                      initial_data_state=initial_state)

        expected_measurement_result = KET_ONE_STATE_VECTOR if num_data_qubits % 2 else KET_ZERO_STATE_VECTOR
        expected_state = tensor(*[KET_MINUS_STATE_VECTOR] * num_data_qubits, expected_measurement_result)
        return states_are_equal(simulation.state, expected_state)
