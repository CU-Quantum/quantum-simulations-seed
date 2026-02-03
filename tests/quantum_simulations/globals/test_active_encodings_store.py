from typing import Optional

from cirq import Circuit, I, LineQubit, TaggedOperation

from quantum_simulations.custom_dataclasses.correction_circuit import CorrectionCircuit
from quantum_simulations.custom_dataclasses.logical_operation import LogicalOperation
from quantum_simulations.error_correcting_codes.error_correcting_code.error_correcting_code import ErrorCorrectingCode
from quantum_simulations.globals.active_encodings_store import ActiveEncodingsStore

CODE_STUB_TAG = 'CODE_STUB_TAG'


class CodeStub(ErrorCorrectingCode):
    def __init__(self, qubits: Optional[list[LineQubit]] = None,):
        super().__init__(num_data_qubits=1,
                         num_logical_qubits=1,
                         qubits=qubits)

    def encode_logical_qubit(self) -> Circuit:
        pass

    def get_error_correction_circuit(self) -> CorrectionCircuit:
        return CorrectionCircuit(
            syndrome_circuit=Circuit(TaggedOperation(I(self.data_qubits[0]), f'CODE_STUB_TAG_{self.data_qubits[0]}'))
        )

    def _perform_get_operation_circuit(self, operation: LogicalOperation) -> Optional[Circuit]:
        pass


class TestActiveEncodingsStore:
    def test_no_tracked_encodings(self):
        with ActiveEncodingsStore(additional_tracked_encodings=[]) as encodings_store:
            circuit = encodings_store.get_all_correction_circuits()
            assert circuit == Circuit()

    def test_register_encoding(self):
        qubits = LineQubit.range(1)
        code = CodeStub(qubits=qubits)
        with ActiveEncodingsStore(additional_tracked_encodings=[code]) as encodings_store:
            circuit = encodings_store.get_all_correction_circuits()
            assert 'CODE_STUB_TAG_q(0)' in str(circuit)

    def test_register_multiple_encodings(self):
        qubits = LineQubit.range(2)
        code = CodeStub(qubits=qubits[:1])
        code2 = CodeStub(qubits=qubits[1:])
        with ActiveEncodingsStore(additional_tracked_encodings=[code, code2]) as encodings_store:
            circuit = encodings_store.get_all_correction_circuits()
            assert 'CODE_STUB_TAG_q(0)' in str(circuit) and 'CODE_STUB_TAG_q(1)' in str(circuit)

    def test_register_using_multiple_stores(self):
        qubits = LineQubit.range(2)
        codes = [CodeStub(qubits=[qubit]) for qubit in qubits]
        with ActiveEncodingsStore(additional_tracked_encodings=codes[:1]) as encodings_store:
            circuit = encodings_store.get_all_correction_circuits()
            assert 'CODE_STUB_TAG_q(0)' in str(circuit) and 'CODE_STUB_TAG_q(1)' not in str(circuit)

            with ActiveEncodingsStore(additional_tracked_encodings=codes[1:]) as encodings_store2:
                circuit = encodings_store2.get_all_correction_circuits()
                assert 'CODE_STUB_TAG_q(0)' in str(circuit) and 'CODE_STUB_TAG_q(1)' in str(circuit)

            circuit = encodings_store.get_all_correction_circuits()
            assert 'CODE_STUB_TAG_q(0)' in str(circuit) and 'CODE_STUB_TAG_q(1)' not in str(circuit)
