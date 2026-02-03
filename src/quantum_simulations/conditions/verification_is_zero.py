from cirq import ClassicalDataDictionaryStore, MeasurementKey, json_cirq_type, obj_to_dict_helper

from quantum_simulations.conditions.custom_condition import CustomCondition


class VerificationIsZero(CustomCondition):
    def __init__(self, key: MeasurementKey):
        self.key = key
        self._last_num_measurements = 0

    @property
    def keys(self):
        return (self.key,)

    def replace_key(self, current: MeasurementKey, replacement: MeasurementKey):
        return VerificationIsZero(replacement) if self.key == current else self

    def __str__(self):
        return str(self.key)

    def __repr__(self):
        return f'{json_cirq_type(type(self))}({self.key!r})'

    def resolve(self, classical_data: ClassicalDataDictionaryStore) -> bool:
        if self.key not in classical_data.keys():
            raise ValueError(f'Measurement key {self.key} missing when verifying all zeros.')
        num_measurements = len(classical_data.records[self.key])
        all_zero = all(classical_data.get_int(self.key, i) == 0 for i in range(self._last_num_measurements, num_measurements))
        self._last_num_measurements = num_measurements
        return all_zero

    def _json_dict_(self):
        return obj_to_dict_helper(self, ['key'])

    @classmethod
    def _from_json_dict_(cls, key: MeasurementKey, **kwargs):
        return cls(key=key)

    @property
    def qasm(self):
        raise ValueError('QASM is defined only for SympyConditions of type key == constant.')
