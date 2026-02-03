from cirq import LineQubit, M, Operation, R, X

from quantum_simulations.globals.fresh_ancillas_pool import FreshAncillasPool
from quantum_simulations.support.cat_state_creator.cat_state_creator_flag_pattern.support.flag_measurer.flag_measurer import \
    FlagMeasurer


class FlagMeasurerSequential(FlagMeasurer):
    def measure_flags(self) -> list[list[Operation]]:
        with FreshAncillasPool().use_fresh_ancillas(num_ancillas=1) as ancilla_qubits:
            ancilla = ancilla_qubits[0]
            return [
                R(ancilla),
                X(ancilla).controlled_by(self._qubit_register[self._parity_check_infos[0].control_qubit_index]),
                [
                    [
                        X(ancilla).controlled_by(self._qubit_register[parity_check_info.control_qubit_index])
                        for previous_parity_check_index, parity_check_info in enumerate(self._parity_check_infos[1:-1])
                        if parity_check_info.flags_outcome[flag_index]
                           != self._parity_check_infos[previous_parity_check_index].flags_outcome[flag_index]
                    ] + (self._get_measurement_and_reset(ancilla=ancilla, flag_index=flag_index) if flag_index < self._num_measurements - 1 else [])
                    for flag_index in range(self._num_measurements)
                ],
                X(ancilla).controlled_by(self._qubit_register[self._parity_check_infos[-1].control_qubit_index]),
                M(ancilla, key=self._measurement_keys[-1]),
            ]

    def _get_measurement_and_reset(self, ancilla: LineQubit, flag_index: int) -> list[Operation]:
        return [
            M(ancilla, key=self._measurement_keys[flag_index]),
            R(ancilla),
        ]
