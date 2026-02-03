from abc import ABC, abstractmethod

from cirq import density_matrix_from_state_vector

from quantum_simulations.utilities.utilities import TYPE_DENSITY_MATRIX, TYPE_STATE_VECTOR


class ExpectedStates(ABC):
    @abstractmethod
    def get_logical_zero_state_vector(self) -> TYPE_STATE_VECTOR:
        pass

    @abstractmethod
    def get_logical_one_state_vector(self) -> TYPE_STATE_VECTOR:
        pass

    def get_logical_zero_density_matrix(self) -> TYPE_DENSITY_MATRIX:
        return density_matrix_from_state_vector(self.get_logical_zero_state_vector())

    def get_logical_one_density_matrix(self) -> TYPE_DENSITY_MATRIX:
        return density_matrix_from_state_vector(self.get_logical_one_state_vector())
