from cirq import M, Operation, ResetChannel, X

from quantum_simulations.globals.fresh_ancillas_pool import FreshAncillasPool
from quantum_simulations.support.cat_state_creator.cat_state_creator_flag_pattern.support.flag_measurer.flag_measurer import \
    FlagMeasurer


class FlagMeasurerParallel(FlagMeasurer):
    def measure_flags(self) -> list[list[Operation]]:
        with FreshAncillasPool().use_fresh_ancillas(num_ancillas=self._num_measurements) as ancilla_qubits:
            return [
                ResetChannel().on_each(*ancilla_qubits),
                X(ancilla_qubits[0]).controlled_by(self._qubit_register[self._parity_check_infos[0].control_qubit_index]),
                [
                    [
                        X(flag_qubit).controlled_by(self._qubit_register[parity_check_info.control_qubit_index])
                        for previous_parity_check_index, parity_check_info in enumerate(self._parity_check_infos[1:-1])
                        if parity_check_info.flags_outcome[flag_index]
                           != self._parity_check_infos[previous_parity_check_index].flags_outcome[flag_index]
                    ] + ([M(flag_qubit, key=measurement_key)] if flag_index < self._num_measurements - 1 else [])
                    for flag_index, (flag_qubit, measurement_key) in enumerate(zip(ancilla_qubits, self._measurement_keys))
                ],
                X(ancilla_qubits[-1]).controlled_by(self._qubit_register[self._parity_check_infos[-1].control_qubit_index]),
                M(ancilla_qubits[-1], key=self._measurement_keys[-1]),
            ]
