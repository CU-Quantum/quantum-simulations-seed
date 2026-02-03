import pytest
from cirq import LineQubit, X, Y, Z
from numpy import array

from quantum_simulations.custom_dataclasses.check_matrix import CheckMatrix
from quantum_simulations.custom_dataclasses.recovery import RecoveryGate, RecoveryOperation
from quantum_simulations.support.cat_state_creator.recovery_finder import RecoveryFinder
from quantum_simulations.support.recovery_combinations_finder import RecoveryCombinationsFinder


class TestRecoveryCombinationsFinder:
    def test_max_zero_errors(self):
        check_matrix = CheckMatrix(matrix=array([[1, 0]]))
        recoveries = RecoveryFinder(check_matrix=check_matrix).find_recovery_gates()
        combos = RecoveryCombinationsFinder(max_num_errors=0).find_recovery_gates(single_error_recoveries=recoveries)
        assert combos == []

    def test_max_one_errors(self):
        check_matrix = CheckMatrix(matrix=array([[1, 0]]))
        recoveries = RecoveryFinder(check_matrix=check_matrix).find_recovery_gates()
        combos = RecoveryCombinationsFinder(max_num_errors=1).find_recovery_gates(single_error_recoveries=recoveries)
        assert combos == [
            RecoveryGate(
                gate=Z,
                qubit_index=0,
                symptom=[1]
            )
        ]

    def test_gates_max_two_errors(self):
        check_matrix = CheckMatrix(matrix=array([[1, 0, 0, 0], [0, 0, 0, 1]]))
        recoveries = RecoveryFinder(check_matrix=check_matrix).find_recovery_gates()
        combos = RecoveryCombinationsFinder(max_num_errors=2).find_recovery_gates(single_error_recoveries=recoveries)
        assert combos == [
            RecoveryGate(
                gate=Z,
                qubit_index=0,
                symptom=[1, 0]
            ),
            RecoveryGate(
                gate=X,
                qubit_index=1,
                symptom=[0, 1]
            ),
            RecoveryGate(
                gate=Z,
                qubit_index=0,
                symptom=[1, 1]
            ),
            RecoveryGate(
                gate=X,
                qubit_index=1,
                symptom=[1, 1]
            ),
        ]

    def test_operations_max_two_errors(self):
        check_matrix = CheckMatrix(matrix=array([[1, 0, 0, 0], [0, 0, 0, 1]]))
        num_qubits = 2
        qubits = LineQubit.range(num_qubits)
        recoveries = RecoveryFinder(check_matrix=check_matrix).find_recovery_operations(qubits=qubits)
        combos = RecoveryCombinationsFinder(max_num_errors=2).find_recovery_operations(single_error_recoveries=recoveries)
        assert combos == [
            RecoveryOperation(
                operation=Z(qubits[0]),
                symptom=[1, 0]
            ),
            RecoveryOperation(
                operation=X(qubits[1]),
                symptom=[0, 1]
            ),
            RecoveryOperation(
                operation=Z(qubits[0]),
                symptom=[1, 1]
            ),
            RecoveryOperation(
                operation=X(qubits[1]),
                symptom=[1, 1]
            ),
        ]

    def test_different_max_x_z(self):
        check_matrix = CheckMatrix(matrix=array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ]))
        recoveries = RecoveryFinder(check_matrix=check_matrix).find_recovery_gates()
        combos = RecoveryCombinationsFinder(max_num_x_errors=2, max_num_z_errors=1).find_recovery_gates(single_error_recoveries=recoveries)
        assert combos == [
            RecoveryGate(
                gate=Z,
                qubit_index=0,
                symptom=[1, 0, 0, 0]
            ),
            RecoveryGate(
                gate=Z,
                qubit_index=1,
                symptom=[0, 1, 0, 0]
            ),
            RecoveryGate(
                gate=X,
                qubit_index=2,
                symptom=[0, 0, 1, 0]
            ),
            RecoveryGate(
                gate=X,
                qubit_index=3,
                symptom=[0, 0, 0, 1]
            ),

            RecoveryGate(
                gate=Z,
                qubit_index=0,
                symptom=[1, 0, 1, 0]
            ),
            RecoveryGate(
                gate=X,
                qubit_index=2,
                symptom=[1, 0, 1, 0]
            ),

            RecoveryGate(
                gate=Z,
                qubit_index=0,
                symptom=[1, 0, 0, 1]
            ),
            RecoveryGate(
                gate=X,
                qubit_index=3,
                symptom=[1, 0, 0, 1]
            ),

            RecoveryGate(
                gate=Z,
                qubit_index=1,
                symptom=[0, 1, 1, 0]
            ),
            RecoveryGate(
                gate=X,
                qubit_index=2,
                symptom=[0, 1, 1, 0]
            ),

            RecoveryGate(
                gate=Z,
                qubit_index=1,
                symptom=[0, 1, 0, 1]
            ),
            RecoveryGate(
                gate=X,
                qubit_index=3,
                symptom=[0, 1, 0, 1]
            ),

            RecoveryGate(
                gate=X,
                qubit_index=2,
                symptom=[0, 0, 1, 1]
            ),
            RecoveryGate(
                gate=X,
                qubit_index=3,
                symptom=[0, 0, 1, 1]
            ),

            RecoveryGate(
                gate=Z,
                qubit_index=0,
                symptom=[1, 0, 1, 1]
            ),
            RecoveryGate(
                gate=X,
                qubit_index=2,
                symptom=[1, 0, 1, 1]
            ),
            RecoveryGate(
                gate=X,
                qubit_index=3,
                symptom=[1, 0, 1, 1]
            ),

            RecoveryGate(
                gate=Z,
                qubit_index=1,
                symptom=[0, 1, 1, 1]
            ),
            RecoveryGate(
                gate=X,
                qubit_index=2,
                symptom=[0, 1, 1, 1]
            ),
            RecoveryGate(
                gate=X,
                qubit_index=3,
                symptom=[0, 1, 1, 1]
            ),
        ]

    def test_different_max_x_z_with_y_errors(self):
        check_matrix = CheckMatrix(matrix=array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 0]]))
        recoveries = RecoveryFinder(check_matrix=check_matrix).find_recovery_gates()
        combos = RecoveryCombinationsFinder(max_num_x_errors=2, max_num_z_errors=1).find_recovery_gates(single_error_recoveries=recoveries)
        assert combos == [
            RecoveryGate(
                gate=Z,
                qubit_index=0,
                symptom=[1, 0, 0]
            ),
            RecoveryGate(
                gate=Z,
                qubit_index=1,
                symptom=[0, 0, 1]
            ),
            RecoveryGate(
                gate=X,
                qubit_index=0,
                symptom=[0, 1, 0]
            ),
            RecoveryGate(
                gate=Y,
                qubit_index=0,
                symptom=[1, 1, 0]
            ),

            RecoveryGate(
                gate=Z,
                qubit_index=1,
                symptom=[0, 1, 1]
            ),
            RecoveryGate(
                gate=X,
                qubit_index=0,
                symptom=[0, 1, 1]
            ),
        ]

    @pytest.mark.parametrize('params', [
        pytest.param((None, None, None), id='empty'),
        pytest.param((None, None, 1), id='no-x'),
        pytest.param((None, 1, None), id='no-z'),
    ])
    def test_invalid_args(self, params: tuple[int | None, int | None, int | None] | None):
        arbitrary_recoveries = [RecoveryGate(gate=Z, qubit_index=0, symptom=[1])]
        recovery_finder = RecoveryCombinationsFinder(max_num_errors=params[0], max_num_x_errors=params[1], max_num_z_errors=params[2])
        with pytest.raises(ValueError, match='^Either "max_num_errors" must be provided or both "max_num_x_errors" and "max_num_z_errors" must be provided.$'):
            recovery_finder.find_recovery_gates(single_error_recoveries=arbitrary_recoveries)
