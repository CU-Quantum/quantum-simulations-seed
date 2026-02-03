from cirq import Circuit, LineQubit, Operation


class ControlledSingleQubitGatesApplier:
    def __init__(self, operations: list[Operation], controls: list[LineQubit]):
        self._operations = operations
        self._controls = controls

    def get_circuit(self) -> Circuit:
        self._validate_inputs()
        return Circuit(
            self._operations[i].controlled_by(self._controls[i])
            for i in range(len(self._operations))
        )

    def _validate_inputs(self) -> None:
        if len(self._operations) != len(self._controls):
            raise ValueError(
                f"The number of gates ({len(self._operations)}) and controls({len(self._controls)}) must be equal.")
        multiqubit_operations = {operation for operation in self._operations if len(operation.qubits) != 1}
        if len(multiqubit_operations):
            raise ValueError(f"All operations must be single-qubit operations. Was given {multiqubit_operations}.")
