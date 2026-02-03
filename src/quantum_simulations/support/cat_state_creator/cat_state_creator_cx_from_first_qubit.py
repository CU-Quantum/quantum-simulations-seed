from cirq import Circuit, CircuitOperation, FrozenCircuit, H, LineQubit, TaggedOperation, inverse

from quantum_simulations.support.cat_state_creator.cat_state_creator import CatStateCreator
from quantum_simulations.support.operations_applier.operations_applier import DELAYED_NOISE_TAG
from quantum_simulations.utilities.utilities import cx_sequentially_further_qubits_from_first

CAT_STATE_CREATOR_CX_FROM_FIRST_QUBIT_TAG = 'CAT_STATE_CREATOR_CX_FROM_FIRST_QUBIT'


class CatStateCreatorCxFromFirstQubit(CatStateCreator):
    def __init__(self, qubit_register: list[LineQubit]):
        super().__init__(qubit_register=qubit_register)
        self._qubit_register = qubit_register

    def get_cat_state_circuit(self) -> Circuit:
        if not self._qubit_register:
            return Circuit()
        return Circuit(
            TaggedOperation(
                CircuitOperation(
                    FrozenCircuit(
                        H(self._qubit_register[0]),
                        cx_sequentially_further_qubits_from_first(qubits=self._qubit_register),
                    )
                ),
                CAT_STATE_CREATOR_CX_FROM_FIRST_QUBIT_TAG, DELAYED_NOISE_TAG
            )
        )

    def decode_state(self) -> Circuit:
        return inverse(self.get_cat_state_circuit())
