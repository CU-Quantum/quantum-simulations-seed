from numpy import argmax
from numpy._typing import NDArray


class NextColumnIndexWithOneAtPositionFinder:
    def __init__(self, matrix: NDArray[NDArray[bool]], row_index: int, column_index: int):
        self._matrix = matrix
        self._starting_row_index = row_index
        self._column_index = column_index

    def get_column_index(self) -> int:
        for row_index in range(self._starting_row_index, len(self._matrix)):
            start_index = self._column_index + 1
            index = start_index + argmax(self._matrix[row_index, start_index:])
            if self._matrix[row_index, index]:
                return index
        raise IndexError(f"Could not find column after column index {self._column_index} having value 1 at row {self._starting_row_index}.")
