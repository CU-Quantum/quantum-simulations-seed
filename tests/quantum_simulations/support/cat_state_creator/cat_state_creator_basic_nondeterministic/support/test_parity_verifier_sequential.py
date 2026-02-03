from cirq import Circuit, I, LineQubit, MeasurementKey
from numpy import array, sqrt

from quantum_simulations.custom_dataclasses.state_and_measurements import StateAndMeasurements
from quantum_simulations.globals.fresh_ancillas_pool import FreshAncillasPool
from quantum_simulations.simulators.circuit_simulator import CircuitSimulatorStateVector
from quantum_simulations.support.cat_state_creator.cat_state_creator_basic_nondeterministic.support.parity_verifier_sequential import \
    ParityVerifierSequential
from quantum_simulations.utilities.utilities import KET_ONE_STATE_VECTOR, KET_ZERO_STATE_VECTOR, tensor
from tests.quantum_simulations.utilities_for_tests import get_cat_state_vector


class TestParityVerifierSequential:
    def test_trivial(self):
        verifier = ParityVerifierSequential(target_qubits=[], measurement_key=MeasurementKey('arbitrary'))
        circuit = verifier.validate_parity()
        assert circuit == Circuit()

    def test_valid_cat_state_one_qubit(self):
        qubits = LineQubit.range(1)
        FreshAncillasPool().set_first_ancilla_num(len(qubits))
        verifier = ParityVerifierSequential(target_qubits=qubits, measurement_key=MeasurementKey('arbitrary'))
        circuit = Circuit(
            I(qubits[0]),
            verifier.validate_parity()
        )

        one_qubit_cat_state = get_cat_state_vector(num_qubits=1)
        circuit_simulator = CircuitSimulatorStateVector()
        state = circuit_simulator.run_simulation(circuit=circuit,
                                                 num_data_qubits=len(qubits),
                                                 initial_data_state=one_qubit_cat_state)
        assert state == StateAndMeasurements(
            state=one_qubit_cat_state,
            measurements={}
        )

    def test_valid_cat_state_two_qubit(self):
        qubits = LineQubit.range(2)
        FreshAncillasPool().set_first_ancilla_num(len(qubits))
        measurement_key = MeasurementKey('arbitrary')
        verifier = ParityVerifierSequential(target_qubits=qubits, measurement_key=MeasurementKey('arbitrary'))
        circuit = Circuit(
            [I(qubit) for qubit in qubits],
            verifier.validate_parity()
        )

        two_qubit_cat_state = get_cat_state_vector(num_qubits=len(qubits))
        circuit_simulator = CircuitSimulatorStateVector()
        state = circuit_simulator.run_simulation(circuit=circuit,
                                                 num_data_qubits=len(qubits),
                                                 initial_data_state=two_qubit_cat_state)
        assert state == StateAndMeasurements(
            state=two_qubit_cat_state,
            measurements={measurement_key.name: array([0])}
        )

    def test_invalid_cat_state_two_qubit(self):
        qubits = LineQubit.range(2)
        FreshAncillasPool().set_first_ancilla_num(len(qubits))
        measurement_key = MeasurementKey('arbitrary')
        verifier = ParityVerifierSequential(target_qubits=qubits, measurement_key=measurement_key)
        circuit = verifier.validate_parity()

        invalid_cat_state = tensor(KET_ZERO_STATE_VECTOR, KET_ONE_STATE_VECTOR)
        circuit_simulator = CircuitSimulatorStateVector()
        state = circuit_simulator.run_simulation(circuit=circuit,
                                                 num_data_qubits=len(qubits),
                                                 initial_data_state=invalid_cat_state)
        assert state == StateAndMeasurements(
            state=invalid_cat_state,
            measurements={measurement_key.name: array([1])}
        )

    def test_valid_cat_state_three_qubit(self):
        qubits = LineQubit.range(3)
        FreshAncillasPool().set_first_ancilla_num(len(qubits))
        measurement_key = MeasurementKey('arbitrary')
        verifier = ParityVerifierSequential(target_qubits=qubits, measurement_key=measurement_key)
        circuit = verifier.validate_parity()

        three_qubit_cat_state = get_cat_state_vector(num_qubits=3)
        circuit_simulator = CircuitSimulatorStateVector()
        state = circuit_simulator.run_simulation(circuit=circuit,
                                                 num_data_qubits=len(qubits),
                                                 initial_data_state=three_qubit_cat_state)
        assert state == StateAndMeasurements(
            state=three_qubit_cat_state,
            measurements={measurement_key.name: array([0])}
        )

    def test_invalid_cat_state_three_qubit(self):
        qubits = LineQubit.range(3)
        FreshAncillasPool().set_first_ancilla_num(len(qubits))
        measurement_key = MeasurementKey('arbitrary')
        verifier = ParityVerifierSequential(target_qubits=qubits, measurement_key=measurement_key)
        circuit = verifier.validate_parity()

        invalid_three_qubit_cat_state = (
                    (1 / sqrt(2)) * (tensor(KET_ZERO_STATE_VECTOR, KET_ZERO_STATE_VECTOR, KET_ONE_STATE_VECTOR)
                                     + tensor(KET_ONE_STATE_VECTOR, KET_ONE_STATE_VECTOR, KET_ZERO_STATE_VECTOR)))
        circuit_simulator = CircuitSimulatorStateVector()
        state = circuit_simulator.run_simulation(circuit=circuit,
                                                 num_data_qubits=len(qubits),
                                                 initial_data_state=invalid_three_qubit_cat_state)
        assert state == StateAndMeasurements(
            state=invalid_three_qubit_cat_state,
            measurements={measurement_key.name: array([1])}
        )
