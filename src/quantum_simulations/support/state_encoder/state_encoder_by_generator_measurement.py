from uuid import uuid4

from cirq import Circuit, FrozenCircuit, LineQubit, Operation

from quantum_simulations.custom_dataclasses.check_matrix import CheckMatrix
from quantum_simulations.globals.configuration_simulation_manager import ConfigurationSimulationManager
from quantum_simulations.support.cat_state_creator.check_matrix_to_operations import CheckMatrixToOperations
from quantum_simulations.support.measurer.measurer import Measurer
from quantum_simulations.support.state_encoder.state_encoder import StateEncoder
from quantum_simulations.utilities.measurement_key_with_stable_hash import MeasurementKeyWithStableHash


class StateEncoderByGeneratorMeasurement(StateEncoder):
    def __init__(self,
                 check_matrix: CheckMatrix,
                 phase_corrections: list[list[Operation]],
                 qubits: list[LineQubit],
                 ):
        self._check_matrix = check_matrix
        self._phase_corrections = phase_corrections
        self._qubits = qubits

    def encode_state(self) -> Circuit:
        measurement_keys = [MeasurementKeyWithStableHash(f'STATE_ENCODER_{i}_{uuid4()}') for i in range(len(self._check_matrix.matrix))]
        generators = CheckMatrixToOperations(check_matrix=self._check_matrix, qubits=self._qubits).get_operations()
        return Circuit(
            self._measurer_type(
                observables=generators,
                measurement_keys=measurement_keys,
                correction_between_repetitions=False,
            ).get_measurement_circuit(),
            FrozenCircuit(  # FrozenCircuit to avoid overlapping operations with noise
                [
                    [
                        phase_correction.with_classical_controls(measurement_key)
                        for phase_correction in self._phase_corrections[generator_index]
                    ]
                    for generator_index, measurement_key in enumerate(measurement_keys)
                ]
            )
        )

    @property
    def _measurer_type(self) -> type[Measurer]:
        configuration = ConfigurationSimulationManager().get_configuration()
        return configuration.measurer_type
