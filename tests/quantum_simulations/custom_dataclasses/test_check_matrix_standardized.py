from re import escape

import pytest
from numpy import array

from quantum_simulations.custom_dataclasses.check_matrix import TYPE_CHECK_MATRIX
from quantum_simulations.custom_dataclasses.check_matrix_standardized import CheckMatrixStandardized, \
    CheckMatrixSubmatrices
from tests.quantum_simulations.error_correcting_codes.predefined_check_matrix_values_standardized import \
    get_check_matrix_values_4_qubit_standardized, get_check_matrix_values_steane_standardized


class TestStabilizersStandardizer:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self._valid_standardized_check_matrix = array([
            [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0],
        ])

    def test_identity_in_pauli_x_portion(self):
        matrix_without_correct_identity_in_pauli_x_portion = self._valid_standardized_check_matrix
        matrix_without_correct_identity_in_pauli_x_portion[:2, :2] = [[0, 1], [1, 0]]
        with pytest.raises(ValueError, match=escape("The first (r)x(r) submatrix must be the identity.")):
            CheckMatrixStandardized(matrix=matrix_without_correct_identity_in_pauli_x_portion)

    def test_zeros_in_pauli_x_portion(self):
        rank_of_pauli_x_portion = 3

        matrix_with_ones_in_pauli_x_portion_below_rank = self._valid_standardized_check_matrix
        matrix_with_ones_in_pauli_x_portion_below_rank[rank_of_pauli_x_portion] = matrix_with_ones_in_pauli_x_portion_below_rank[0]
        with pytest.raises(ValueError, match="All rows in the pauli_x portion below the rank must be 0."):
            CheckMatrixStandardized(matrix=matrix_with_ones_in_pauli_x_portion_below_rank)

    def test_zeros_in_pauli_z_portion(self):
        num_logical_qubits = 1
        matrix_with_ones_in_c_one_submatrix = self._valid_standardized_check_matrix
        matrix_with_ones_in_c_one_submatrix[0, -num_logical_qubits - 1] = 1
        with pytest.raises(ValueError, match=escape("The (r)x(n-k-r) submatrix beginning at index [0, n+r] must be 0.")):
            CheckMatrixStandardized(matrix=matrix_with_ones_in_c_one_submatrix)

    def test_identity_in_pauli_z_portion(self):
        num_logical_qubits = 1
        rank_of_pauli_x_portion = 3

        matrix_without_correct_identity_in_pauli_z_portion = self._valid_standardized_check_matrix
        matrix_without_correct_identity_in_pauli_z_portion[rank_of_pauli_x_portion + 1, -num_logical_qubits - 1] = 1
        with pytest.raises(ValueError, match=escape("The (n-k-r)x(n-k-r) submatrix beginning at index [r, n+r] must be the identity.")):
            CheckMatrixStandardized(matrix=matrix_without_correct_identity_in_pauli_z_portion)

    def test_can_retrieve_submatrices(self):
        matrix = CheckMatrixStandardized(matrix=self._valid_standardized_check_matrix)
        submatrices = matrix.submatrices
        assert submatrices == CheckMatrixSubmatrices(
            a1=array([[0, 1, 1], [1, 0, 1], [1, 1, 1]]),
            a2=array([[1], [1], [0]]),
            b=array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
            c=array([[0], [0], [0]]),
            d=array([[1, 0, 1], [0, 1, 1], [1, 1, 1]]),
            e=array([[1], [1], [0]]),
        )

    def test_correct_submatrices_for_4_qubit(self):
        standardized_4_qubit_generators = get_check_matrix_values_4_qubit_standardized()
        matrix = CheckMatrixStandardized(matrix=standardized_4_qubit_generators)
        submatrices = matrix.submatrices
        assert submatrices == CheckMatrixSubmatrices(
            a1=array([[], []]),
            a2=array([[0, 1], [1, 0]]),
            b=array([[0, 1], [1, 1]]),
            c=array([[1, 0], [1, 1]]),
            d=array([]),
            e=array([]),
        )

    @pytest.mark.parametrize(['matrix_values', 'expected_xs'], [
        (get_check_matrix_values_steane_standardized(), [[0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]]),
        (get_check_matrix_values_4_qubit_standardized(), [[0, 0, 1, 0, 1, 1, 0, 0], [0, 0, 0, 1, 0, 1, 0, 0]]),
    ])
    def test_determines_logical_x_operators(self, matrix_values: TYPE_CHECK_MATRIX, expected_xs: TYPE_CHECK_MATRIX):
        matrix = CheckMatrixStandardized(matrix=matrix_values)
        logical_xs = matrix.logical_xs
        assert logical_xs.tolist() == expected_xs

    @pytest.mark.parametrize(['matrix_values', 'expected_zs'], [
        (get_check_matrix_values_steane_standardized(), [[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1]]),
        (get_check_matrix_values_4_qubit_standardized(), [[0, 0, 0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 1, 0, 0, 1]]),
    ])
    def test_determines_logical_z_operators(self, matrix_values: TYPE_CHECK_MATRIX, expected_zs: TYPE_CHECK_MATRIX):
        matrix = CheckMatrixStandardized(matrix=matrix_values)
        logical_zs = matrix.logical_zs
        assert logical_zs.tolist() == expected_zs

