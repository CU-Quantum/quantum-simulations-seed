import pytest
from cirq import LineQubit

from quantum_simulations.globals.fresh_ancillas_pool import FreshAncillasPool
from quantum_simulations.support.cat_state_creator.cat_state_creator_flag_pattern.cat_state_creator_flag_pattern import \
    CatStateCreatorFlagPattern
from tests.quantum_simulations.support.cat_state_creator.cat_state_creator_flag_pattern.utilities import \
    circuit_results_in_expected_state, get_cat_state_with_x_error, get_circuit_with_x_error_on_first_n_qubits
from tests.quantum_simulations.utilities_for_tests import get_cat_state_vector


class TestCatStateCreatorFlagPattern12Qubits:
    _num_qubits = 12
    _qubits = LineQubit.range(_num_qubits)

    @pytest.fixture(autouse=True, scope='class')
    def _setup(self):
        FreshAncillasPool().set_first_ancilla_num(first_ancilla_num=self._num_qubits)

    def test_no_errors(self):
        qubits = LineQubit.range(self._num_qubits)
        circuit = CatStateCreatorFlagPattern(qubit_register=qubits).get_cat_state_circuit()

        expected_state = get_cat_state_vector(num_qubits=self._num_qubits)
        assert circuit_results_in_expected_state(circuit=circuit, expected_state=expected_state)

    @pytest.mark.parametrize('num_qubits_with_error', [3, 6, 9, 12])
    def test_x_error_on_multiple_of_3(self, num_qubits_with_error: int):
        first_error_qubit_num = num_qubits_with_error
        circuit = get_circuit_with_x_error_on_first_n_qubits(qubits=self._qubits, n=first_error_qubit_num)
        expected_state = get_cat_state_vector(num_qubits=self._num_qubits)
        return circuit_results_in_expected_state(circuit=circuit, expected_state=expected_state)

    def test_x_error_on_one_less_than_multiple_of_3_creates_error_on_multiple_of_3(self):
        multiple_of_3 = 3
        circuit = get_circuit_with_x_error_on_first_n_qubits(qubits=self._qubits, n=multiple_of_3 - 1)
        expected_state = get_cat_state_with_x_error(num_qubits=self._num_qubits, qubit_index_with_error=multiple_of_3 - 1)
        assert circuit_results_in_expected_state(circuit=circuit, expected_state=expected_state)

    def test_x_error_on_one_more_than_multiple_of_3(self):
        multiple_of_3 = 3
        circuit = get_circuit_with_x_error_on_first_n_qubits(qubits=self._qubits, n=multiple_of_3 + 1)
        expected_state = get_cat_state_with_x_error(num_qubits=self._num_qubits, qubit_index_with_error=multiple_of_3)
        assert circuit_results_in_expected_state(circuit=circuit, expected_state=expected_state)
