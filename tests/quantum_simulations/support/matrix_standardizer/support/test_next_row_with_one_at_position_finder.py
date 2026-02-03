from numpy import array

from quantum_simulations.support.matrix_standardizer.support.next_row_index_with_one_at_position_finder import \
    NextRowIndexWithOneAtPositionFinder


class TestNextRowWithOneAtPositionFinder:
    def test_trivial(self):
        finder = NextRowIndexWithOneAtPositionFinder(matrix=array([[1], [1]]), row_index=0, column_index=0)
        index = finder.get_row_index()
        assert index == 1

    def test_row_does_not_exist(self):
        finder = NextRowIndexWithOneAtPositionFinder(matrix=array([[0, 1], [0, 0]]), row_index=0, column_index=0)
        index = finder.get_row_index()
        assert index is None

    def test_ensure_nonzero_row_and_column(self):
        finder = NextRowIndexWithOneAtPositionFinder(matrix=array([[1, 0], [0, 0], [1, 1]]), row_index=1, column_index=1)
        index = finder.get_row_index()
        assert index == 2
