import numpy as np
import pytest
from cirq import ClassicalDataDictionaryStore, MeasurementKey, LineQubit
from numpy import array

from quantum_simulations.conditions.flag_index_limit import FlagIndexLimit


class TestFlagIndexLimit:
    def test_missing_key(self):
        store = ClassicalDataDictionaryStore()
        key = MeasurementKey('test_key')
        condition = FlagIndexLimit(measurement_keys=(key,), parity_check_index=0, flag_sequence=array([]))

        with pytest.raises(ValueError, match='^At least one measurement key from "\\(cirq.MeasurementKey\\(name=\'test_key\'\\),\\)" is missing when checking flags$'):
            condition.resolve(classical_data=store)

    def test_empty_flag_sequence(self):
        store = ClassicalDataDictionaryStore()
        key = MeasurementKey('test_key')
        store.record_measurement(key=key, measurement=(0,), qubits=(LineQubit(0),))

        condition1 = FlagIndexLimit(measurement_keys=(key,), parity_check_index=0, flag_sequence=array([]))
        with pytest.raises(ValueError, match='^No flag sequence was given for key "\\(cirq.MeasurementKey\\(name=\'test_key\'\\),\\)"$'):
            condition1.resolve(classical_data=store)

    def test_parity_check_index_less_than_or_equal_to_first_occurrence(self):
        store = ClassicalDataDictionaryStore()
        keys = (MeasurementKey('test_key1'), MeasurementKey('test_key2'))
        flag_sequence = np.array([[0, 1], [1, 0], [1, 1]])

        store.record_measurement(key=keys[0], measurement=(1,), qubits=(LineQubit(0),))
        store.record_measurement(key=keys[1], measurement=(0,), qubits=(LineQubit(0),))
        conditions = [FlagIndexLimit(measurement_keys=keys, parity_check_index=i, flag_sequence=flag_sequence)
                      for i in range(3)]
        assert conditions[0].resolve(classical_data=store)
        assert conditions[1].resolve(classical_data=store)
        assert not conditions[2].resolve(classical_data=store)
