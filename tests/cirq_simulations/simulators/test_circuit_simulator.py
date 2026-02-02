from cirq import Circuit, LineQubit, M, X
from numpy import array

from cirq_simulations.custom_dataclasses.state_and_measurements import StateAndMeasurements
from cirq_simulations.simulators.circuit_simulator import CircuitSimulatorStateVector
from cirq_simulations.utilities.utilities import KET_ZERO_STATE_VECTOR, tensor


class TestCircuitSimulator:
    def test_multiple_measurements_on_same_qubit_only_uses_last_one(self):
        initial_state = KET_ZERO_STATE_VECTOR
        circuit_simulator = CircuitSimulatorStateVector()

        num_qubits = 1
        qubit = LineQubit(num_qubits)
        circuit = Circuit(
            M(qubit),
            X(qubit),
            M(qubit),
            X(qubit),
        )

        result = circuit_simulator.run_simulation(circuit=circuit,
                                                  num_data_qubits=num_qubits,
                                                  initial_data_state=initial_state)
        assert result == StateAndMeasurements(
            state=initial_state,
            measurements={'q(1)': array([1])}
        )

    def test_measurements_on_different_qubits(self):
        initial_state = tensor(KET_ZERO_STATE_VECTOR, KET_ZERO_STATE_VECTOR)
        circuit_simulator = CircuitSimulatorStateVector()

        qubits = LineQubit.range(2)
        circuit = Circuit(
            X(qubits[1]),
            M(qubits[0]),
            M(qubits[1]),
            X(qubits[1]),
        )

        result = circuit_simulator.run_simulation(circuit=circuit,
                                                                num_data_qubits=len(qubits),
                                                                initial_data_state=initial_state)
        assert result == StateAndMeasurements(
            state=initial_state,
            measurements={'q(0)': array([0]), 'q(1)': [1]}
        )
