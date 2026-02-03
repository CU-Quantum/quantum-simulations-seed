from cirq import Circuit, I, LineQubit, Simulator, Z

from quantum_simulations.utilities.utilities import TYPE_DENSITY_MATRIX, TYPE_STATE_VECTOR, tensor
from tests.quantum_simulations.error_correcting_codes.expected_states.expected_states import ExpectedStates
from tests.quantum_simulations.utilities_for_tests import get_cat_state_vector


class ExpectedStatesGeneralizedShor(ExpectedStates):
    arbitrary_num_qubits_per_cat = 4
    arbitrary_num_cats = 5

    def get_logical_zero_state_vector(self) -> TYPE_STATE_VECTOR:
        return tensor(*[get_cat_state_vector(num_qubits=self.arbitrary_num_qubits_per_cat)] * self.arbitrary_num_cats)

    def get_logical_one_state_vector(self) -> TYPE_DENSITY_MATRIX:
        one_cat_state = get_cat_state_vector(num_qubits=self.arbitrary_num_qubits_per_cat)
        one_cat_state_flipped_sign = Simulator().simulate(Circuit(
            [I(LineQubit(i)) for i in range(self.arbitrary_num_qubits_per_cat)],
            Z(LineQubit(0)),
        ), initial_state=one_cat_state).final_state_vector
        return tensor(*[one_cat_state_flipped_sign] * self.arbitrary_num_cats)
