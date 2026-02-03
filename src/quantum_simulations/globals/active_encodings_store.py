from typing import Optional
from uuid import uuid4

from cirq import Circuit, CircuitOperation, FrozenCircuit, OP_TREE, TaggedOperation

from quantum_simulations.custom_dataclasses.correction_circuit import CorrectionCircuit
from quantum_simulations.error_correcting_codes.error_correcting_code.error_correcting_code import ErrorCorrectingCode
from quantum_simulations.support.operations_applier.operations_applier import DELAYED_NOISE_TAG

CORRECTION_ROUND_TAG = 'CORRECTION_ROUND'
CORRECTION_ROUND_SYNDROMES_TAG = 'CORRECTION_ROUND_SYNDROMES'
CORRECTION_ROUND_RECOVERIES_TAG = 'CORRECTION_ROUND_RECOVERIES'
ENCODING_TAG = 'ENCODING'


class ActiveEncodingsStore:
    _tracked_encodings: dict[str, list[ErrorCorrectingCode]] = {}

    def __init__(self, additional_tracked_encodings: list[ErrorCorrectingCode]):
        self._id = uuid4().hex
        self._tracked_encodings[self._id] = additional_tracked_encodings

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self._tracked_encodings[self._id]

    def get_all_correction_circuits(self, additional_correction_circuits: Optional[list[CorrectionCircuit]] = None) -> OP_TREE:
        correction_circuits = [encoding.get_error_correction_circuit()
                               for encodings in self._tracked_encodings.values()
                               for encoding in encodings]
        if additional_correction_circuits:
            correction_circuits.extend(additional_correction_circuits)
        if not correction_circuits:
            return Circuit()

        syndrome_circuits = [
            TaggedOperation(
                CircuitOperation(
                    FrozenCircuit(correction_circuit.syndrome_circuit),
                ),
                CORRECTION_ROUND_SYNDROMES_TAG, f'{ENCODING_TAG}_{i}'
            )
            for i, correction_circuit in enumerate(correction_circuits)
        ]
        recovery_circuits = [
            TaggedOperation(
                CircuitOperation(
                    FrozenCircuit(correction_circuit.recovery_circuit),
                ),
                CORRECTION_ROUND_RECOVERIES_TAG, f'{ENCODING_TAG}_{i}'
            )
            for i, correction_circuit in enumerate(correction_circuits)
        ]
        return Circuit(
            TaggedOperation(
                CircuitOperation(
                    FrozenCircuit(
                        TaggedOperation(
                            CircuitOperation(
                                FrozenCircuit(syndrome_circuits),
                            ),
                            CORRECTION_ROUND_SYNDROMES_TAG, DELAYED_NOISE_TAG
                        ),
                        TaggedOperation(
                            CircuitOperation(
                                FrozenCircuit(recovery_circuits),
                            ),
                            CORRECTION_ROUND_RECOVERIES_TAG, DELAYED_NOISE_TAG
                        )
                    )
                ),
                CORRECTION_ROUND_TAG,
            )
        )
