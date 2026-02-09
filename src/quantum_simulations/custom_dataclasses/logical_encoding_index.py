from dataclasses import dataclass
from typing import Optional

from cirq import Circuit

from quantum_simulations.custom_dataclasses.logical_operation import LogicalGateLabel, LogicalOperation
from quantum_simulations.error_correcting_codes.error_correcting_code.error_correcting_code import ErrorCorrectingCode


@dataclass
class LogicalEncodingIndex:
    encoding: ErrorCorrectingCode
    qubit_index_relative: int
    qubit_index_logical: Optional[int] = None

    def get_observable(self, gate: LogicalGateLabel) -> Circuit:
        return self.encoding.get_operation_circuit(operation=LogicalOperation(gate=gate, qubit_index=self.qubit_index_relative))
