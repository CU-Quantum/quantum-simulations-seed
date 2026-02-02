from numpy import array, exp, pi

from quantum_simulations.utilities.utilities import KET_ONE_STATE_VECTOR, KET_PLUS_STATE_VECTOR, KET_ZERO_STATE_VECTOR, \
    states_are_equal, tensor


class TestStatesAreEqual:
    def test_identical_states(self):
        state1 = KET_ZERO_STATE_VECTOR
        state2 = KET_ZERO_STATE_VECTOR
        assert states_are_equal(state1, state2)

    def test_different_states(self):
        state1 = KET_ZERO_STATE_VECTOR
        state2 = KET_ONE_STATE_VECTOR
        result = states_are_equal(state1, state2)
        assert not result

    def test_global_phase_difference(self):
        state1 = KET_ZERO_STATE_VECTOR
        state2 = exp(1j * pi / 4) * KET_ZERO_STATE_VECTOR
        result = states_are_equal(state1, state2)
        assert result

    def test_global_phase_difference_complex_state(self):
        state1 = KET_PLUS_STATE_VECTOR
        state2 = exp(1j * pi / 2) * KET_PLUS_STATE_VECTOR
        result = states_are_equal(state1, state2)
        assert result

    def test_nan_looks_like_global_phase(self):
        state1 = array([1, -1], dtype=complex)
        state2 = array([0, 1], dtype=complex)
        result = states_are_equal(state1, state2)
        assert not result

    def test_multi_qubit_states_equal(self):
        state1 = tensor(KET_ZERO_STATE_VECTOR, KET_ZERO_STATE_VECTOR)
        state2 = tensor(KET_ZERO_STATE_VECTOR, KET_ZERO_STATE_VECTOR)
        result = states_are_equal(state1, state2)
        assert result

    def test_multi_qubit_states_with_phase(self):
        state1 = tensor(KET_ZERO_STATE_VECTOR, KET_ONE_STATE_VECTOR)
        state2 = exp(1j * pi / 3) * tensor(KET_ZERO_STATE_VECTOR, KET_ONE_STATE_VECTOR)  # 60-degree phase shift
        result = states_are_equal(state1, state2)
        assert result

    def test_multi_qubit_states_different(self):
        state1 = tensor(KET_ZERO_STATE_VECTOR, KET_ZERO_STATE_VECTOR)
        state2 = tensor(KET_ZERO_STATE_VECTOR, KET_ONE_STATE_VECTOR)
        result = states_are_equal(state1, state2)
        assert not result

    def test_states_with_zeros_and_phase(self):
        state1 = array([1.0, 0.0, 1.0, 0.0], dtype=complex)
        state2 = exp(1j * pi / 4) * array([1.0, 0.0, 1.0, 0.0], dtype=complex)
        result = states_are_equal(state1, state2)
        assert result
