import copy

from quantum_simulations.custom_dataclasses.check_matrix import CheckMatrix
from quantum_simulations.custom_dataclasses.check_matrix_standardized import CheckMatrixStandardized
from quantum_simulations.support.matrix_standardizer.support.next_column_index_with_one_at_position_finder import \
    NextColumnIndexWithOneAtPositionFinder
from quantum_simulations.support.matrix_standardizer.support.next_row_index_with_one_at_position_finder import \
    NextRowIndexWithOneAtPositionFinder


class CheckMatrixStandardizer:
    def __init__(self, check_matrix: CheckMatrix):
        self._new_check_matrix = CheckMatrix(matrix=copy.deepcopy(check_matrix.matrix))

    def get_standardized_matrix(self) -> CheckMatrixStandardized:
        self._set_first_identity_matrix()
        self._set_second_identity_matrix()
        return CheckMatrixStandardized(
            matrix=self._new_check_matrix.matrix,
            qubit_order=self._new_check_matrix.qubit_order,
        )

    def _set_first_identity_matrix(self) -> None:
        for i in range(self._new_check_matrix.rank_of_pauli_x_portion):
            self._set_element_equal_to_one(row_index=i, column_index=i)
            self._set_rest_of_column_to_zero(row_index_in_identity_form=i, column_index=i)

    def _set_second_identity_matrix(self) -> None:
        for i in range(self._new_check_matrix.num_physical_qubits - self._new_check_matrix.num_logical_qubits - self._new_check_matrix.rank_of_pauli_x_portion):
            row_index = self._new_check_matrix.rank_of_pauli_x_portion + i
            column_index = self._new_check_matrix.num_physical_qubits + self._new_check_matrix.rank_of_pauli_x_portion + i
            self._set_element_equal_to_one(row_index=row_index, column_index=column_index)
            self._set_rest_of_column_to_zero(row_index_in_identity_form=row_index, column_index=column_index)

    def _set_element_equal_to_one(self, row_index: int, column_index: int) -> None:
            while not self._new_check_matrix.matrix[row_index, column_index]:
                row_index_with_one_in_column = NextRowIndexWithOneAtPositionFinder(matrix=self._new_check_matrix.matrix,
                                                                                   row_index=row_index,
                                                                                   column_index=column_index).get_row_index()
                if row_index_with_one_in_column is not None:
                    self._new_check_matrix.add_rows(row_index=row_index_with_one_in_column, target_row_index=row_index)
                else:
                    is_pauli_z_section = int(column_index >= self._new_check_matrix.num_physical_qubits)
                    start_column = is_pauli_z_section * self._new_check_matrix.num_physical_qubits
                    matrix_separating_x_and_z = self._new_check_matrix.matrix[:, start_column:start_column + self._new_check_matrix.num_physical_qubits]
                    column_index_with_one_in_row = NextColumnIndexWithOneAtPositionFinder(matrix=matrix_separating_x_and_z,
                                                                                          row_index=row_index,
                                                                                          column_index=column_index % self._new_check_matrix.num_physical_qubits
                                                                                          ).get_column_index()
                    self._new_check_matrix.swap_qubits(qubit_indices=(column_index, start_column + column_index_with_one_in_row))

    def _set_rest_of_column_to_zero(self, row_index_in_identity_form: int, column_index: int) -> None:
        for i in range(len(self._new_check_matrix.matrix)):
            if i != row_index_in_identity_form and self._new_check_matrix.matrix[i, column_index]:
                self._new_check_matrix.add_rows(row_index=row_index_in_identity_form, target_row_index=i)
