from abc import ABC, abstractmethod

from cirq import LineQubit, MeasurementKey, Operation

from quantum_simulations.custom_dataclasses.cat_state_flag_info import CatStateFlagInfo


class FlagMeasurer(ABC):
    def __init__(self,
                 qubit_register: list[LineQubit],
                 parity_check_infos: list[CatStateFlagInfo],
                 measurement_keys: list[MeasurementKey]):
        self._qubit_register = qubit_register
        self._parity_check_infos = parity_check_infos
        self._measurement_keys = measurement_keys

    def __post_init__(self):
        if len(self._measurement_keys) != self._num_measurements:
            raise ValueError(f"The number of measurement keys ({len(self._measurement_keys)}) must be equal to the number of flags ({self._num_measurements}).")

    @abstractmethod
    def measure_flags(self) -> list[list[Operation]]:
        pass

    @property
    def _num_measurements(self) -> int:
        return len(self._parity_check_infos[0].flags_outcome)
