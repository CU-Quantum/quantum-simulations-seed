from collections import defaultdict
from dataclasses import dataclass, field

from numpy import array
from numpy._typing import NDArray
from numpy.ma.core import allequal

from quantum_simulations.utilities.utilities import TYPE_STATE_VECTOR_OR_DENSITY_MATRIX, states_are_equal


@dataclass
class Measurements:
    measurements: dict[str, NDArray[int]] = field(default_factory=lambda: defaultdict(list))

    @property
    def measurements_per_shot(self) -> NDArray[NDArray[int]]:
        return array(list(self.logical_qubit_measurements.values())).transpose()[0][0]

    @property
    def logical_qubit_measurements(self) -> dict[str, NDArray[int]]:
        return {k: v for k, v in self.measurements.items() if k.isdigit()}

    def __eq__(self, other):
        return list(self.measurements.keys()) == list(other.measurements.keys()) \
            and all(allequal(v, other.measurements[k]) for k, v in self.measurements.items())


@dataclass
class StateAndMeasurements(Measurements):
    state: TYPE_STATE_VECTOR_OR_DENSITY_MATRIX = field(default_factory=lambda: array([[]]))

    def __eq__(self, other):
        return super().__eq__(other) \
            and states_are_equal(self.state, other.state)
