from quantum_simulations.utilities.utilities import KET_ONE_STATE_VECTOR, KET_ZERO_STATE_VECTOR, TYPE_STATE_VECTOR, \
    tensor
from tests.quantum_simulations.error_correcting_codes.expected_states.expected_states import ExpectedStates


class ExpectedStatesFiveQubit(ExpectedStates):
    def get_logical_zero_state_vector(self) -> TYPE_STATE_VECTOR:
        return (1/4) * (
                tensor(*[KET_ZERO_STATE_VECTOR] * 5)
                + tensor(KET_ONE_STATE_VECTOR,
                        KET_ZERO_STATE_VECTOR,
                        KET_ZERO_STATE_VECTOR,
                        KET_ONE_STATE_VECTOR,
                        KET_ZERO_STATE_VECTOR)
                + tensor(KET_ZERO_STATE_VECTOR,
                        KET_ONE_STATE_VECTOR,
                        KET_ZERO_STATE_VECTOR,
                        KET_ZERO_STATE_VECTOR,
                        KET_ONE_STATE_VECTOR)
                + tensor(KET_ONE_STATE_VECTOR,
                        KET_ZERO_STATE_VECTOR,
                        KET_ONE_STATE_VECTOR,
                        KET_ZERO_STATE_VECTOR,
                        KET_ZERO_STATE_VECTOR)
                + tensor(KET_ZERO_STATE_VECTOR,
                        KET_ONE_STATE_VECTOR,
                        KET_ZERO_STATE_VECTOR,
                        KET_ONE_STATE_VECTOR,
                        KET_ZERO_STATE_VECTOR)
                - tensor(KET_ONE_STATE_VECTOR,
                        KET_ONE_STATE_VECTOR,
                        KET_ZERO_STATE_VECTOR,
                        KET_ONE_STATE_VECTOR,
                        KET_ONE_STATE_VECTOR)
                - tensor(KET_ZERO_STATE_VECTOR,
                        KET_ZERO_STATE_VECTOR,
                        KET_ONE_STATE_VECTOR,
                        KET_ONE_STATE_VECTOR,
                        KET_ZERO_STATE_VECTOR)
                - tensor(KET_ONE_STATE_VECTOR,
                        KET_ONE_STATE_VECTOR,
                        KET_ZERO_STATE_VECTOR,
                        KET_ZERO_STATE_VECTOR,
                        KET_ZERO_STATE_VECTOR)
                - tensor(KET_ONE_STATE_VECTOR,
                        KET_ONE_STATE_VECTOR,
                        KET_ONE_STATE_VECTOR,
                        KET_ZERO_STATE_VECTOR,
                        KET_ONE_STATE_VECTOR)
                - tensor(KET_ZERO_STATE_VECTOR,
                        KET_ZERO_STATE_VECTOR,
                        KET_ZERO_STATE_VECTOR,
                        KET_ONE_STATE_VECTOR,
                        KET_ONE_STATE_VECTOR)
                - tensor(KET_ONE_STATE_VECTOR,
                        KET_ONE_STATE_VECTOR,
                        KET_ONE_STATE_VECTOR,
                        KET_ONE_STATE_VECTOR,
                        KET_ZERO_STATE_VECTOR)
                - tensor(KET_ZERO_STATE_VECTOR,
                        KET_ONE_STATE_VECTOR,
                        KET_ONE_STATE_VECTOR,
                        KET_ONE_STATE_VECTOR,
                        KET_ONE_STATE_VECTOR)
                - tensor(KET_ONE_STATE_VECTOR,
                        KET_ZERO_STATE_VECTOR,
                        KET_ZERO_STATE_VECTOR,
                        KET_ZERO_STATE_VECTOR,
                        KET_ONE_STATE_VECTOR)
                - tensor(KET_ZERO_STATE_VECTOR,
                        KET_ONE_STATE_VECTOR,
                        KET_ONE_STATE_VECTOR,
                        KET_ZERO_STATE_VECTOR,
                        KET_ZERO_STATE_VECTOR)
                - tensor(KET_ONE_STATE_VECTOR,
                        KET_ZERO_STATE_VECTOR,
                        KET_ONE_STATE_VECTOR,
                        KET_ONE_STATE_VECTOR,
                        KET_ONE_STATE_VECTOR)
                + tensor(KET_ZERO_STATE_VECTOR,
                        KET_ZERO_STATE_VECTOR,
                        KET_ONE_STATE_VECTOR,
                        KET_ZERO_STATE_VECTOR,
                        KET_ONE_STATE_VECTOR)
        )

    def get_logical_one_state_vector(self) -> TYPE_STATE_VECTOR:
        return (1 / 4) * (
                tensor(*[KET_ONE_STATE_VECTOR] * 5)
                + tensor(KET_ZERO_STATE_VECTOR,
                       KET_ONE_STATE_VECTOR,
                       KET_ONE_STATE_VECTOR,
                       KET_ZERO_STATE_VECTOR,
                       KET_ONE_STATE_VECTOR)
                + tensor(KET_ONE_STATE_VECTOR,
                       KET_ZERO_STATE_VECTOR,
                       KET_ONE_STATE_VECTOR,
                       KET_ONE_STATE_VECTOR,
                       KET_ZERO_STATE_VECTOR)
                + tensor(KET_ZERO_STATE_VECTOR,
                       KET_ONE_STATE_VECTOR,
                       KET_ZERO_STATE_VECTOR,
                       KET_ONE_STATE_VECTOR,
                       KET_ONE_STATE_VECTOR)
                + tensor(KET_ONE_STATE_VECTOR,
                       KET_ZERO_STATE_VECTOR,
                       KET_ONE_STATE_VECTOR,
                       KET_ZERO_STATE_VECTOR,
                       KET_ONE_STATE_VECTOR)
                - tensor(KET_ZERO_STATE_VECTOR,
                       KET_ZERO_STATE_VECTOR,
                       KET_ONE_STATE_VECTOR,
                       KET_ZERO_STATE_VECTOR,
                       KET_ZERO_STATE_VECTOR)
                - tensor(KET_ONE_STATE_VECTOR,
                       KET_ONE_STATE_VECTOR,
                       KET_ZERO_STATE_VECTOR,
                       KET_ZERO_STATE_VECTOR,
                       KET_ONE_STATE_VECTOR)
                - tensor(KET_ZERO_STATE_VECTOR,
                       KET_ZERO_STATE_VECTOR,
                       KET_ONE_STATE_VECTOR,
                       KET_ONE_STATE_VECTOR,
                       KET_ONE_STATE_VECTOR)
                - tensor(KET_ZERO_STATE_VECTOR,
                       KET_ZERO_STATE_VECTOR,
                       KET_ZERO_STATE_VECTOR,
                       KET_ONE_STATE_VECTOR,
                       KET_ZERO_STATE_VECTOR)
                - tensor(KET_ONE_STATE_VECTOR,
                       KET_ONE_STATE_VECTOR,
                       KET_ONE_STATE_VECTOR,
                       KET_ZERO_STATE_VECTOR,
                       KET_ZERO_STATE_VECTOR)
                - tensor(KET_ZERO_STATE_VECTOR,
                       KET_ZERO_STATE_VECTOR,
                       KET_ZERO_STATE_VECTOR,
                       KET_ZERO_STATE_VECTOR,
                       KET_ONE_STATE_VECTOR)
                - tensor(KET_ONE_STATE_VECTOR,
                       KET_ZERO_STATE_VECTOR,
                       KET_ZERO_STATE_VECTOR,
                       KET_ZERO_STATE_VECTOR,
                       KET_ZERO_STATE_VECTOR)
                - tensor(KET_ZERO_STATE_VECTOR,
                       KET_ONE_STATE_VECTOR,
                       KET_ONE_STATE_VECTOR,
                       KET_ONE_STATE_VECTOR,
                       KET_ZERO_STATE_VECTOR)
                - tensor(KET_ONE_STATE_VECTOR,
                       KET_ZERO_STATE_VECTOR,
                       KET_ZERO_STATE_VECTOR,
                       KET_ONE_STATE_VECTOR,
                       KET_ONE_STATE_VECTOR)
                - tensor(KET_ZERO_STATE_VECTOR,
                       KET_ONE_STATE_VECTOR,
                       KET_ZERO_STATE_VECTOR,
                       KET_ZERO_STATE_VECTOR,
                       KET_ZERO_STATE_VECTOR)
                + tensor(KET_ONE_STATE_VECTOR,
                       KET_ONE_STATE_VECTOR,
                       KET_ZERO_STATE_VECTOR,
                       KET_ONE_STATE_VECTOR,
                       KET_ZERO_STATE_VECTOR)
        )
