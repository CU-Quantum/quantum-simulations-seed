from abc import ABC, abstractmethod
from typing import Optional

from cirq import Circuit, \
    LineQubit, \
    NOISE_MODEL_LIKE, \
    ResetChannel, Simulator, \
    StateVectorTrialResult

from quantum_simulations.custom_dataclasses.state_and_measurements import StateAndMeasurements
from quantum_simulations.globals.configuration_simulation_manager import ConfigurationSimulationManager
from quantum_simulations.utilities.utilities import KET_ZERO_STATE_VECTOR, \
    TYPE_STATE_VECTOR, TYPE_STATE_VECTOR_OR_DENSITY_MATRIX, tensor, \
    trace_out_ancillas_in_zero_state


class CircuitSimulator(ABC):
    @property
    @abstractmethod
    def zero_state(self) -> TYPE_STATE_VECTOR_OR_DENSITY_MATRIX:
        pass

    @abstractmethod
    def _get_simulation_result(self,
                               circuit: Circuit,
                               qubits: list[LineQubit],
                               initial_state: Optional[TYPE_STATE_VECTOR_OR_DENSITY_MATRIX] = None,
                               noise_model: Optional[NOISE_MODEL_LIKE] = None,
                               ) -> StateAndMeasurements:
        pass

    def run_simulation(self,
                       circuit: Circuit,
                       num_data_qubits: int,
                       initial_data_state: Optional[TYPE_STATE_VECTOR_OR_DENSITY_MATRIX] = None,
                       noise_model: Optional[NOISE_MODEL_LIKE] = None,
                       ) -> StateAndMeasurements:
        if initial_data_state is None:
            initial_data_state = tensor(*[self.zero_state] * num_data_qubits)
        qubits = LineQubit.range(self.get_max_qubit_index(circuit=circuit) + 1)
        ancilla_qubits = LineQubit.range(num_data_qubits, len(qubits))
        num_ancillas = len(ancilla_qubits)
        initial_state = tensor(initial_data_state, *[self.zero_state] * num_ancillas)

        circuit_with_reset_ancillas = Circuit(
            circuit,
            ResetChannel().on_each(*ancilla_qubits),
        )
        simulation = self._get_simulation_result(circuit=circuit_with_reset_ancillas,
                                                 qubits=qubits,
                                                 initial_state=initial_state,
                                                 noise_model=noise_model)
        data_state = trace_out_ancillas_in_zero_state(state=simulation.state, num_ancillas=num_ancillas)

        return StateAndMeasurements(
            state=data_state,
            measurements=simulation.measurements,
        )

    def get_max_qubit_index(self, circuit: Circuit) -> int:
        all_qubits = list(circuit.all_qubits())
        return max(all_qubits).x if all_qubits else -1

    @property
    def _seed(self) -> int:
        return ConfigurationSimulationManager().get_configuration().seed


class CircuitSimulatorStateVector(CircuitSimulator):
    @property
    def zero_state(self) -> TYPE_STATE_VECTOR:
        return KET_ZERO_STATE_VECTOR

    def _get_simulation_result(self,
                               circuit: Circuit,
                               qubits: list[LineQubit],
                               initial_state: Optional[TYPE_STATE_VECTOR] = None,
                               noise_model: Optional[NOISE_MODEL_LIKE] = None,
                               ) -> StateAndMeasurements:
        simulator = Simulator(noise=noise_model, seed=self._seed)
        simulation: StateVectorTrialResult = simulator.simulate(circuit, qubit_order=qubits, initial_state=initial_state)
        return StateAndMeasurements(
            state=simulation.final_state_vector,
            measurements=dict(simulation.measurements),
        )
