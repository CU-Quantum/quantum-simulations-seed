from cirq import LineQubit, X, Y, Z
from numpy import array

from quantum_simulations.custom_dataclasses.check_matrix import CheckMatrix
from quantum_simulations.support.cat_state_creator.operations_to_check_matrix import OperationsToCheckMatrix


class TestOperationsToCheckMatrix:
    def test_x(self):
        operations = [[X(LineQubit(0))]]
        check_matrix = OperationsToCheckMatrix(operations_list=operations).get_check_matrix()
        assert check_matrix == CheckMatrix(matrix=array([[1, 0]]), qubit_order=[0])

    def test_y(self):
        operations = [[Y(LineQubit(0))]]
        check_matrix = OperationsToCheckMatrix(operations_list=operations).get_check_matrix()
        assert check_matrix == CheckMatrix(matrix=array([[1, 1]]), qubit_order=[0])

    def test_z(self):
        operations = [[Z(LineQubit(0))]]
        check_matrix = OperationsToCheckMatrix(operations_list=operations).get_check_matrix()
        assert check_matrix == CheckMatrix(matrix=array([[0, 1]]), qubit_order=[0])

    def test_multiple_gates(self):
        operations = [[X(LineQubit(0)), X(LineQubit(1))]]
        check_matrix = OperationsToCheckMatrix(operations_list=operations).get_check_matrix()
        assert check_matrix == CheckMatrix(matrix=array([[1, 1, 0, 0]]), qubit_order=[0, 1])

    def test_unordered_qubits(self):
        operations = [[X(LineQubit(1)), Z(LineQubit(0))]]
        check_matrix = OperationsToCheckMatrix(operations_list=operations).get_check_matrix()
        assert check_matrix == CheckMatrix(matrix=array([[0, 1, 1, 0]]), qubit_order=[0, 1])

    def test_non_sequential_qubits(self):
        operations = [[X(LineQubit(0)), Z(LineQubit(2))]]
        check_matrix = OperationsToCheckMatrix(operations_list=operations).get_check_matrix()
        assert check_matrix == CheckMatrix(matrix=array([[1, 0, 0, 1]]), qubit_order=[0, 1])

    def test_multiple_stabilizers(self):
        operations = [[X(LineQubit(0)), X(LineQubit(1))], [Z(LineQubit(0))]]
        check_matrix = OperationsToCheckMatrix(operations_list=operations).get_check_matrix()
        assert check_matrix == CheckMatrix(matrix=array([[1, 1, 0, 0], [0, 0, 1, 0]]), qubit_order=[0, 1])

    def test_no_zero_qubit(self):
        operations = [[X(LineQubit(1))]]
        check_matrix = OperationsToCheckMatrix(operations_list=operations).get_check_matrix()
        assert check_matrix == CheckMatrix(matrix=array([[1, 0]]), qubit_order=[0])
