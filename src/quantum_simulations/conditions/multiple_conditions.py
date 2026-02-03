from cirq import ClassicalDataDictionaryStore, Condition, MeasurementKey, json_cirq_type, obj_to_dict_helper

from quantum_simulations.conditions.custom_condition import CustomCondition


class MultipleConditions(CustomCondition):
    def __init__(self, conditions: list[Condition]):
        self.conditions = conditions

    @property
    def keys(self):
        return tuple(key for condition in self.conditions for key in condition.keys)

    def replace_key(self, current: MeasurementKey, replacement: MeasurementKey):
        for condition in self.conditions:
            if current in condition.keys:
                condition.replace_key(current, replacement)
        return self

    def __str__(self):
        return str(self.keys)

    def __repr__(self):
        return f'{json_cirq_type(type(self))}({self.conditions})'

    def resolve(self, classical_data: ClassicalDataDictionaryStore) -> bool:
        return all(condition.resolve(classical_data=classical_data) for condition in self.conditions)

    def _json_dict_(self):
        return obj_to_dict_helper(self, ['conditions'])

    @classmethod
    def _from_json_dict_(cls, conditions, **kwargs):
        return cls(conditions=conditions)

    @property
    def qasm(self):
        raise ValueError('QASM is defined only for SympyConditions of type key == constant.')
