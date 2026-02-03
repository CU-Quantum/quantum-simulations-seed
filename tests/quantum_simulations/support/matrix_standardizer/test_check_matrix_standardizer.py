import copy
from re import escape

import pytest
from numpy import array

from quantum_simulations.custom_dataclasses.check_matrix import CheckMatrix
from quantum_simulations.custom_dataclasses.check_matrix_standardized import CheckMatrixStandardized
from quantum_simulations.predefined_check_matrix_values import get_check_matrix_values_4_qubit, \
    get_check_matrix_values_8_qubit, get_check_matrix_values_shors, get_check_matrix_values_steane
from quantum_simulations.support.matrix_standardizer.check_matrix_standardizer import CheckMatrixStandardizer
from tests.quantum_simulations.error_correcting_codes.predefined_check_matrix_values_standardized import \
    get_check_matrix_values_4_qubit_standardized, get_check_matrix_values_8_qubit_standardized, \
    get_check_matrix_values_shors_standardized, get_check_matrix_values_steane_standardized


class TestCheckMatrixStandardizer:
    def test_empty(self):
        with pytest.raises(ValueError, match=escape("The number of rows must be at most than half the number of columns. Shape (1, 0) was provided.")):
            CheckMatrixStandardizer(check_matrix=CheckMatrix(matrix=array([[]])))

    def test_steane(self):
        standardizer = CheckMatrixStandardizer(check_matrix=CheckMatrix(matrix=get_check_matrix_values_steane()))
        standardized_check = standardizer.get_standardized_matrix()
        assert standardized_check == CheckMatrixStandardized(
            matrix=get_check_matrix_values_steane_standardized(),
            qubit_order=[0,1,3,2,4,6,5],
        )

    def test_4_qubit(self):
        standardizer = CheckMatrixStandardizer(check_matrix=CheckMatrix(matrix=get_check_matrix_values_4_qubit()))
        standardized_check = standardizer.get_standardized_matrix()
        assert standardized_check == CheckMatrixStandardized(
            matrix=get_check_matrix_values_4_qubit_standardized(),
            qubit_order=[0, 1, 2, 3],
        )

    def test_8_qubit(self):
        standardizer = CheckMatrixStandardizer(check_matrix=CheckMatrix(matrix=get_check_matrix_values_8_qubit()))
        standardized_check = standardizer.get_standardized_matrix()
        assert standardized_check == CheckMatrixStandardized(
            matrix=get_check_matrix_values_8_qubit_standardized(),
            qubit_order=[0, 1, 2, 4, 3, 5, 6, 7],
        )

    def test_does_not_modify_original_check_matrix(self):
        generators = get_check_matrix_values_4_qubit()
        original_check_matrix = CheckMatrix(matrix=copy.deepcopy(generators))
        CheckMatrixStandardizer(check_matrix=original_check_matrix).get_standardized_matrix()
        assert original_check_matrix.matrix.tolist() == generators.tolist()

    def test_shors(self):
        generators = get_check_matrix_values_shors()
        standardizer = CheckMatrixStandardizer(check_matrix=CheckMatrix(matrix=generators))
        standardized_check = standardizer.get_standardized_matrix()
        assert standardized_check == CheckMatrixStandardized(
            matrix=get_check_matrix_values_shors_standardized(),
            qubit_order=[0, 3, 2, 1, 4, 5, 6, 7, 8],
        )
