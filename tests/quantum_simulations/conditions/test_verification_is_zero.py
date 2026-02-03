import pytest
from cirq import ClassicalDataDictionaryStore, MeasurementKey, LineQubit

from quantum_simulations.conditions.verification_is_zero import VerificationIsZero


class TestVerificationIsZero:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self._store = ClassicalDataDictionaryStore()
        self._key = MeasurementKey('test_key')
        self._condition = VerificationIsZero(key=self._key)
        self._qubit = LineQubit(0)

    def test_missing_key(self):
        with pytest.raises(ValueError, match='^Measurement key test_key missing when verifying all zeros\\.$'):
            self._condition.resolve(classical_data=self._store)
    
    def test_resolve_all_zeros(self):
        self._record_measurements([0, 0, 0,])
        assert self._condition.resolve(classical_data=self._store)

    def test_resolve_some_non_zeros(self):
        self._record_measurements([0, 1, 0])
        assert not self._condition.resolve(classical_data=self._store)

    def test_resolve_incremental(self):
        self._record_measurements([0, 1])
        assert not self._condition.resolve(classical_data=self._store)

        self._record_measurements([0, 0])
        assert self._condition.resolve(classical_data=self._store)

    def _record_measurements(self, measurements: list[int]):
        for measurement in measurements:
            self._store.record_measurement(key=self._key, measurement=(measurement,), qubits=(self._qubit,))
