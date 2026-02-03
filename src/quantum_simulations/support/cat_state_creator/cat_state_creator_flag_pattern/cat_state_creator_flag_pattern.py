from functools import cached_property
from uuid import uuid4

import numpy as np
from cirq import Circuit, H, LineQubit, MeasurementKey, Operation, X, inverse
from numpy._typing import NDArray
from numpy.ma.extras import average

from quantum_simulations.conditions.flag_index_limit import FlagIndexLimit
from quantum_simulations.custom_dataclasses.cat_state_flag_info import CatStateFlagInfo
from quantum_simulations.support.cat_state_creator.cat_state_creator import CatStateCreator
from quantum_simulations.support.cat_state_creator.cat_state_creator_flag_pattern.support.flag_measurer.flag_measurer import \
    FlagMeasurer
from quantum_simulations.support.cat_state_creator.cat_state_creator_flag_pattern.support.flag_measurer.flag_measurer_parallel import \
    FlagMeasurerParallel
from quantum_simulations.support.cat_state_creator.cat_state_creator_flag_pattern.support.flag_sequnce_generator import \
    FlagSequenceGenerator
from quantum_simulations.utilities.measurement_key_with_stable_hash import MeasurementKeyWithStableHash
from quantum_simulations.utilities.utilities import cx_sequentially_closer_qubits_from_first


class CatStateCreatorFlagPattern(CatStateCreator):
    """
    Idea comes from https://quantum-journal.org/papers/q-2023-10-24-1154/
    Note that you apparently cannot use this for syndrome measurement.
    """
    def __init__(self, qubit_register: list[LineQubit], flag_measurer_type: type[FlagMeasurer] = FlagMeasurerParallel):
        super().__init__(qubit_register=qubit_register)
        self._flag_measurer_type = flag_measurer_type

    def get_cat_state_circuit(self) -> Circuit:
        if not self._num_data_qubits:
            return Circuit()
        if self._num_data_qubits <= 3:
            return Circuit(
                self._create_cat_state(),
            )
        return Circuit(
            self._create_cat_state(),
            self._measure_flags(),
            self._recover_from_errors(),
        )

    def decode_state(self) -> Circuit:
        if not self._num_data_qubits:
            return Circuit()
        return Circuit(
            inverse(self._create_cat_state()),
        )

    def _create_cat_state(self) -> list[list[Operation]]:
        return [
            [H(self._control_qubit)],
            cx_sequentially_closer_qubits_from_first(qubits=self._qubit_register),
        ]

    def _measure_flags(self) -> list[list[Operation]]:
        return self._flag_measurer_type(qubit_register=self._qubit_register,
                                        parity_check_infos=self._parity_check_infos,
                                        measurement_keys=self._measurement_keys,
                                        ).measure_flags()

    def _recover_from_errors(self) -> list[list[Operation]]:
        return [
            [X(qubit).with_classical_controls(FlagIndexLimit(measurement_keys=tuple(self._measurement_keys),
                                                             parity_check_index=parity_check_index - 1,
                                                             flag_sequence=self._flag_sequence)
                                              )
             for qubit in self._qubit_register[self._parity_check_infos[parity_check_index - 1].recovery_qubit_num:self._parity_check_infos[parity_check_index].recovery_qubit_num]]
            for parity_check_index in range(2, len(self._parity_check_infos) - 1)
        ]

    @cached_property
    def _measurement_keys(self) -> list[MeasurementKey]:
        return [MeasurementKeyWithStableHash(f"CAT_STATE_FLAG_PATTERN_{uuid4().hex}") for _ in range(self._num_measurements)]

    @cached_property
    def _parity_check_infos(self) -> list[CatStateFlagInfo]:
        perfect_num_data_qubits = 3 * (2 ** self._num_measurements - 2 * self._num_measurements + 2)
        num_data_qubits_less_than_perfect = perfect_num_data_qubits - self._num_data_qubits
        initial_flag = CatStateFlagInfo(control_qubit_index=0,
                                        flags_outcome=self._flag_sequence[0])
        last_flag = CatStateFlagInfo(control_qubit_index=perfect_num_data_qubits - 1)
        parity_check_data = ([initial_flag]
                           + [CatStateFlagInfo(control_qubit_index=last_seq_num * 3 + 1,
                                               recovery_qubit_num=3 * last_seq_num,
                                               flags_outcome=flags_outcome)
                              for last_seq_num, flags_outcome in enumerate(self._flag_sequence[1:])]
                           + [last_flag])
        for i in range(num_data_qubits_less_than_perfect):
            measurement_num_to_move = next(parity_check_index for parity_check_index in range(len(parity_check_data) - 2, 1, -1)
                 if parity_check_data[parity_check_index].control_qubit_index - 1 > parity_check_data[parity_check_index - 1].control_qubit_index)
            for j in range(measurement_num_to_move, len(parity_check_data)):
                parity_check_data[j].control_qubit_index -= 1
                parity_check_data[j].recovery_qubit_num = int(np.floor(average([parity_check_data[j - 1].control_qubit_index, parity_check_data[j].control_qubit_index]))) + 1
        return parity_check_data

    @cached_property
    def _flag_sequence(self) -> NDArray[NDArray[int]]:
        return FlagSequenceGenerator(num_flags=self._num_measurements).get_flag_sequence()

    @property
    def _control_qubit(self) -> LineQubit:
        return self._qubit_register[0]

    @cached_property
    def _num_measurements(self) -> int:
        arbitrary_measurement_limit = 100
        return next(m for m in range(2, arbitrary_measurement_limit) if self._num_data_qubits <= 3 * (2 ** m - 2 * m + 2))

    @property
    def _num_data_qubits(self) -> int:
        return len(self._qubit_register)
