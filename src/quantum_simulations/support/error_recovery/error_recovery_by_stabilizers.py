from uuid import uuid4

from cirq import Circuit, Operation

from quantum_simulations.custom_dataclasses.correction_circuit import CorrectionCircuit
from quantum_simulations.custom_dataclasses.recovery import RecoveryOperation
from quantum_simulations.globals.configuration_simulation_manager import ConfigurationSimulationManager
from quantum_simulations.support.measurer.measurer import Measurer
from quantum_simulations.utilities.measurement_key_with_stable_hash import MeasurementKeyWithStableHash
from quantum_simulations.utilities.utilities import get_sympy_conditions_all_equal


class ErrorRecoveryByStabilizers:
    def __init__(self, stabilizers: list[list[Operation]], recoveries: list[RecoveryOperation]):
        self._stabilizers = stabilizers
        self._recoveries = recoveries

    def get_error_correction_circuit(self) -> CorrectionCircuit:
        measurement_keys = [MeasurementKeyWithStableHash(f'ERROR_RECOVERY_{uuid4().hex}') for _ in range(len(self._stabilizers))]

        syndrome_operations = self._measurer_type(
            observables=self._stabilizers,
            measurement_keys=measurement_keys,
            correction_between_repetitions=False
        ).get_measurement_circuit()

        recovery_conditions = [
            get_sympy_conditions_all_equal(measurement_keys, recovery.symptom)
            for recovery in self._recoveries
        ]
        recovery_operations = [
            recovery.operation.with_classical_controls(*conditions)
            for recovery, conditions in zip(self._recoveries, recovery_conditions)
        ]

        return CorrectionCircuit(
            syndrome_circuit=Circuit(syndrome_operations),
            recovery_circuit=Circuit(recovery_operations),
        )

    @property
    def _measurer_type(self) -> type[Measurer]:
        return ConfigurationSimulationManager().get_configuration().measurer_type
