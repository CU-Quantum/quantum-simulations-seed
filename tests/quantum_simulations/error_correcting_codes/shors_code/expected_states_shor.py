from numpy import sqrt

from quantum_simulations.utilities.utilities import KET_ONE_STATE_VECTOR, KET_ZERO_STATE_VECTOR, TYPE_STATE_VECTOR, \
    tensor
from tests.quantum_simulations.error_correcting_codes.expected_states.expected_states import ExpectedStates
from tests.quantum_simulations.utilities_for_tests import get_cat_state_vector


class ExpectedStatesShor(ExpectedStates):
    def get_logical_zero_state_vector(self) -> TYPE_STATE_VECTOR:
        GHZ_3 = get_cat_state_vector(num_qubits=3)
        return tensor(*[GHZ_3] * 3)

    def get_logical_one_state_vector(self) -> TYPE_STATE_VECTOR:
        GHZ_3_MINUS = (1 / sqrt(2)) * (tensor(KET_ZERO_STATE_VECTOR, KET_ZERO_STATE_VECTOR, KET_ZERO_STATE_VECTOR)
                                       - tensor(KET_ONE_STATE_VECTOR, KET_ONE_STATE_VECTOR, KET_ONE_STATE_VECTOR))
        return tensor(GHZ_3_MINUS, GHZ_3_MINUS, GHZ_3_MINUS)
