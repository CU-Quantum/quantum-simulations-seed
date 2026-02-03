from typing import Optional

import pytest
from cirq import Circuit, I, LineQubit

from quantum_simulations.custom_dataclasses.correction_circuit import CorrectionCircuit
from quantum_simulations.custom_dataclasses.logical_operation import LogicalGateLabel, LogicalOperation
from quantum_simulations.error_correcting_codes.error_correcting_code.error_correcting_code import ErrorCorrectingCode


class TestErrorCorrectingCode:
    def test_can_get_data_qubits(self):
        expected_num_qubits = 2
        code = CodeStub(num_data_qubits=expected_num_qubits)
        assert code.data_qubits == LineQubit.range(expected_num_qubits)

    def test_cannot_apply_unimplemented_operation(self):
        code = CodeStub()
        with pytest.raises(NotImplementedError, match="Operation X is not implemented for code CodeStub."):
            code.get_operation_circuit(LogicalOperation(gate=LogicalGateLabel.X, qubit_index=0))

    def test_apply_implemented_operation(self):
        code = CodeStub()
        circuit = code.get_operation_circuit(LogicalOperation(gate=LogicalGateLabel.Z, qubit_index=0))
        assert circuit == Circuit(I(LineQubit(0)))

    def test_qubit_index_must_be_at_least_zero(self):
        code = CodeStub(num_logical_qubits=2)
        with pytest.raises(ValueError, match="Qubit index must be between 0 and 1. Was given -1."):
            code.get_operation_circuit(operation=LogicalOperation(gate=LogicalGateLabel.Z, qubit_index=-1))
        code.get_operation_circuit(operation=LogicalOperation(gate=LogicalGateLabel.Z, qubit_index=0))

    def test_qubit_index_must_be_at_most_largest_logical_index(self):
        code = CodeStub(num_data_qubits=2, num_logical_qubits=2)
        with pytest.raises(ValueError, match="Qubit index must be between 0 and 1. Was given 2."):
            code.get_operation_circuit(operation=LogicalOperation(gate=LogicalGateLabel.Z, qubit_index=2))
        code.get_operation_circuit(operation=LogicalOperation(gate=LogicalGateLabel.Z, qubit_index=1))

        code = CodeStub(num_data_qubits=1, num_logical_qubits=1)
        with pytest.raises(ValueError, match="Qubit index must be between 0 and 0. Was given 1."):
            code.get_operation_circuit(operation=LogicalOperation(gate=LogicalGateLabel.Z, qubit_index=1))
        code.get_operation_circuit(operation=LogicalOperation(gate=LogicalGateLabel.Z, qubit_index=0))

    def test_can_retrieve_num_logical_qubits(self):
        expected_num_qubits = 2
        code = CodeStub(num_logical_qubits=expected_num_qubits)
        assert code.num_logical_qubits == expected_num_qubits

    def test_can_create_new(self):
        num_qubits = 1
        code_original = CodeStub(num_logical_qubits=num_qubits)
        new_code = code_original.create_new(qubits=[LineQubit(num_qubits)])
        circuit1, circuit2 = (code.get_operation_circuit(LogicalOperation(gate=LogicalGateLabel.Z, qubit_index=0))
                              for code in (code_original, new_code))
        assert circuit1 == Circuit(I(LineQubit(0))) and circuit2 == Circuit(I(LineQubit(1)))

    def test_validates_num_provided_qubits(self):
        with pytest.raises(ValueError, match="The number of provided qubits \\(1\\) must be equal to the specified number of data qubits \\(2\\)."):
            _ = CodeStub(num_data_qubits=2, qubits=[LineQubit(0)]).data_qubits


class CodeStub(ErrorCorrectingCode):
    def __init__(self,
                 num_data_qubits: int = 1,
                 num_logical_qubits: int = 1,
                 qubits: Optional[list[LineQubit]] = None,
                 ):
        super().__init__(num_data_qubits=num_data_qubits,
                         num_logical_qubits=num_logical_qubits,
                         qubits=qubits)

    def encode_logical_qubit(self) -> Circuit:
        pass

    def get_error_correction_circuit(self) -> CorrectionCircuit:
        pass

    def _perform_get_operation_circuit(self, operation: LogicalOperation) -> Circuit:
        if operation.gate == LogicalGateLabel.Z:
            return Circuit(I(self.data_qubits[operation.qubit_index]))
        return None
