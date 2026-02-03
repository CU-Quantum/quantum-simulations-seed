from quantum_simulations.utilities.utilities import KET_ONE_STATE_VECTOR, KET_ZERO_STATE_VECTOR, TYPE_DENSITY_MATRIX, \
    tensor
from tests.quantum_simulations.error_correcting_codes.expected_states.expected_states import ExpectedStates


class ExpectedStatesRepetition(ExpectedStates):
    arbitrary_num_qubits = 4

    def get_logical_zero_state_vector(self) -> TYPE_DENSITY_MATRIX:
        return tensor(*[KET_ZERO_STATE_VECTOR] * self.arbitrary_num_qubits)

    def get_logical_one_state_vector(self) -> TYPE_DENSITY_MATRIX:
        return tensor(*[KET_ONE_STATE_VECTOR] * self.arbitrary_num_qubits)
