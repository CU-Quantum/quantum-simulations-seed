from typing import Optional

import numpy as np

from quantum_simulations.utilities.utilities import KET_ONE_STATE_VECTOR, KET_ZERO_STATE_VECTOR, TYPE_STATE_VECTOR, \
    int_to_binary_array, tensor
from tests.quantum_simulations.error_correcting_codes.expected_states.expected_states import ExpectedStates


class ExpectedStatesGeneralizedShorHadamard(ExpectedStates):
    num_qubits_per_cat = 4
    num_cats = 5

    def __init__(self, num_cats: Optional[int] = None, num_qubits_per_cat: Optional[int] = None):
        if num_cats:
            self.num_cats = num_cats
        if num_qubits_per_cat:
            self.num_qubits_per_cat = num_qubits_per_cat

    def get_logical_zero_state_vector(self) -> TYPE_STATE_VECTOR:
        cat_values_with_even_weight = [cat_values for cat_values in self._all_cat_values if not sum(cat_values) % 2]
        return self._get_logical_state_vector(cat_values=cat_values_with_even_weight)

    def get_logical_one_state_vector(self) -> TYPE_STATE_VECTOR:
        cat_values_with_odd_weight = [cat_values for cat_values in self._all_cat_values if sum(cat_values) % 2]
        return self._get_logical_state_vector(cat_values=cat_values_with_odd_weight)

    def _get_logical_state_vector(self, cat_values: list[list[int]]) -> TYPE_STATE_VECTOR:
        basis_states_by_value = [
            [tensor(*[KET_ONE_STATE_VECTOR if cat_value else KET_ZERO_STATE_VECTOR] * self.num_qubits_per_cat)
             for cat_value in cat_values]
            for cat_values in cat_values
        ]
        basis_states = [tensor(*states) for states in basis_states_by_value]
        return (1 / np.sqrt(2) ** np.log2(len(basis_states))) * np.sum(basis_states, axis=0)

    @property
    def _all_cat_values(self) -> list[list[int]]:
        return [int_to_binary_array(i, num_elements=self.num_cats) for i in range(2 ** self.num_cats)]
