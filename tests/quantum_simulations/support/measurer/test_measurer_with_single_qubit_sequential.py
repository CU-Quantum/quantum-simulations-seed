from cirq import Circuit, LineQubit, MeasurementKey, X, Z

from quantum_simulations.globals.fresh_ancillas_pool import FreshAncillasPool
from quantum_simulations.simulators.circuit_simulator import CircuitSimulatorStateVector
from quantum_simulations.support.measurer.measurer_with_single_qubit_sequential import MeasurerWithSingleQubitSequential
from quantum_simulations.utilities.utilities import KET_MINUS_STATE_VECTOR, KET_ONE_STATE_VECTOR, KET_PLUS_STATE_VECTOR, \
    tensor
from tests.quantum_simulations.utilities_for_tests import set_seed


class TestMeasurerWithSingleQubitSequential:
    def test_trivial(self):
        measurer = MeasurerWithSingleQubitSequential(observables=[])
        circuit = measurer.get_measurement_circuit()
        assert circuit == Circuit()

    def test_one_operation_z(self):
        qubits = LineQubit.range(1)
        FreshAncillasPool.set_first_ancilla_num(first_ancilla_num=len(qubits))
        measurement_key = MeasurementKey('TEST')
        measurer = MeasurerWithSingleQubitSequential(observables=[[Z(qubits[0])]], measurement_keys=[measurement_key])
        circuit = measurer.get_measurement_circuit()

        initial_state = KET_PLUS_STATE_VECTOR
        circuit_simulator = CircuitSimulatorStateVector()

        num_trials = 5
        measurements = []
        for trial in range(num_trials):
            set_seed(trial)
            simulation = circuit_simulator.run_simulation(circuit=circuit,
                                                          num_data_qubits=len(qubits),
                                                          initial_data_state=initial_state)
            measurements.extend(simulation.measurements[measurement_key.name])
        assert any(measurements) and not all(measurements)

    def test_multiple_operations_z(self):
        qubits = LineQubit.range(2)
        FreshAncillasPool.set_first_ancilla_num(first_ancilla_num=len(qubits))
        measurement_key = MeasurementKey('TEST')
        measurer = MeasurerWithSingleQubitSequential(observables=[[X(qubits[0]), Z(qubits[1])]],
                                                     measurement_keys=[measurement_key])
        circuit = measurer.get_measurement_circuit()

        initial_state = tensor(KET_MINUS_STATE_VECTOR, KET_ONE_STATE_VECTOR)
        circuit_simulator = CircuitSimulatorStateVector()

        num_trials = 5
        measurements = []
        for trial in range(num_trials):
            set_seed(trial)
            simulation = circuit_simulator.run_simulation(circuit=circuit,
                                                          num_data_qubits=len(qubits),
                                                          initial_data_state=initial_state)
            measurements.extend(simulation.measurements[measurement_key.name])
        assert not any(measurements)
