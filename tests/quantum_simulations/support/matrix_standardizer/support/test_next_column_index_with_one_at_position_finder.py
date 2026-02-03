import pytest
from numpy import array

from quantum_simulations.support.matrix_standardizer.support.next_column_index_with_one_at_position_finder import \
    NextColumnIndexWithOneAtPositionFinder


class TestNextColumnWithOneAtPositionFinder:
    def test_trivial(self):
        finder = NextColumnIndexWithOneAtPositionFinder(matrix=array([[1, 1]]), row_index=0, column_index=0)
        index = finder.get_column_index()
        assert index == 1

    def test_column_is_not_in_same_row_pauli_x(self):
        finder = NextColumnIndexWithOneAtPositionFinder(matrix=array([[0, 0], [0, 1]]), row_index=0, column_index=0)
        index = finder.get_column_index()
        assert index == 1

    def test_column_does_not_exist(self):
        with pytest.raises(IndexError, match="Could not find column after column index 0 having value 1 at row 0."):
            finder = NextColumnIndexWithOneAtPositionFinder(matrix=array([[0, 0]]), row_index=0, column_index=0)
            finder.get_column_index()

    def test_ensure_column_is_after_given_column_index(self):
        finder = NextColumnIndexWithOneAtPositionFinder(matrix=array([[1, 1, 1]]), row_index=0, column_index=1)
        index = finder.get_column_index()
        assert index == 2

    def test_not_first_row(self):
        finder = NextColumnIndexWithOneAtPositionFinder(matrix=array([[1, 1, 1], [0, 0, 1]]), row_index=1, column_index=1)
        index = finder.get_column_index()
        assert index == 2
