from dataclasses import dataclass
from typing import Tuple

import numpy as np
from cirq import ClassicalDataStoreReader, MeasurementKey, dataclass_json_dict, json_cirq_type
from numpy._typing import NDArray

from quantum_simulations.conditions.custom_condition import CustomCondition


@dataclass(frozen=True)
class FlagIndexLimit(CustomCondition):
    measurement_keys: tuple[MeasurementKey, ...]  # tuple instead of list so it can be hashed/serialized
    parity_check_index: int
    flag_sequence: NDArray[int]

    @property
    def keys(self) -> Tuple[MeasurementKey, ...]:
        return self.measurement_keys

    def replace_key(self, current: MeasurementKey, replacement: MeasurementKey):
        if current in self.keys:
            new_keys = set(self.measurement_keys)
            new_keys.remove(current)
            new_keys.add(replacement)
            return FlagIndexLimit(tuple(new_keys), self.parity_check_index, self.flag_sequence)
        return self

    def __str__(self):
        return str(self.keys)

    def __repr__(self):
        return f'{json_cirq_type(type(self))}({self.keys!r}, {self.parity_check_index}, {self.flag_sequence})'

    def resolve(self, classical_data: ClassicalDataStoreReader) -> bool:
        if not all(key in classical_data.keys() for key in self.keys):
            raise ValueError(f'At least one measurement key from "{self.keys}" is missing when checking flags')
        if not len(self.flag_sequence):
            raise ValueError(f'No flag sequence was given for key "{self.keys}"')

        measurements = [classical_data.records[key][-1][0] for key in self.keys]
        flag_nums_found = np.where(np.all(self.flag_sequence == measurements, axis=1))[0]
        return self.parity_check_index <= flag_nums_found[0] if len(flag_nums_found) else False

    def _json_dict_(self):
        return dataclass_json_dict(self)

    @classmethod
    def _from_json_dict_(cls, measurement_keys, parity_check_index, flag_sequence, **kwargs):
        return cls(measurement_keys=measurement_keys, parity_check_index=parity_check_index, flag_sequence=flag_sequence)

    @property
    def qasm(self):
        raise ValueError('QASM is defined only for SympyConditions of type key == constant.')
