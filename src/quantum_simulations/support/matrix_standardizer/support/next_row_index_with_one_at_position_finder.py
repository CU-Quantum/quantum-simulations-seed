from typing import Optional

from numpy import array
from numpy._typing import NDArray


class NextRowIndexWithOneAtPositionFinder:
    def __init__(self, matrix: NDArray[NDArray[bool]], row_index: int, column_index: int):
        self._matrix = matrix
        self._row_index = row_index
        self._column_index = column_index

    def get_row_index(self) -> Optional[int]:
        start_index = self._row_index + 1
        return next((start_index + i for i, row in enumerate(self._matrix[start_index:]) if row[self._column_index]), None)
