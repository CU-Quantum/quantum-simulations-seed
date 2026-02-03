import pytest
from cirq import Circuit, I, LineQubit, X

from quantum_simulations.support.operations_applier.operations_applier import OperationsApplier


class OperationsApplierStub(OperationsApplier):
    def _perform_get_application_circuit(self) -> Circuit:
        return Circuit(I(self._measurement_qubit))


class TestOperationsApplier:
    def test_validate_disjoint_qubits_failure(self):
        operations = [X(LineQubit(0))]
        measurement_qubit = LineQubit(0)
        with pytest.raises(ValueError, match='^The target qubits and measurement qubit must be disjoint\\. '
                                             'Found duplicate qubit q\\(0\\)\\.$'):
            applier = OperationsApplierStub(operations=operations, measurement_qubit=measurement_qubit)
            applier.get_application_circuit()

    def test_validate_disjoint_qubits_multiple_operations(self):
        operations = [X(LineQubit(0)), X(LineQubit(1)), X(LineQubit(2))]
        measurement_qubit = LineQubit(1)
        with pytest.raises(ValueError, match='^The target qubits and measurement qubit must be disjoint\\. '
                                             'Found duplicate qubit q\\(1\\)\\.$'):
            applier = OperationsApplierStub(operations=operations, measurement_qubit=measurement_qubit)
            applier._validate()

    def test_get_application_circuit_calls_perform_get_application_circuit(self):
        operations = [X(LineQubit(0))]
        measurement_qubit = LineQubit(1)

        applier = OperationsApplierStub(operations=operations, measurement_qubit=measurement_qubit)
        assert applier.get_application_circuit() == Circuit(I(LineQubit(1)))
