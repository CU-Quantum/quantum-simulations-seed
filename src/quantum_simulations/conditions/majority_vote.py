from uuid import uuid4

from cirq import ClassicalDataDictionaryStore, MeasurementKey, json_cirq_type, obj_to_dict_helper
from numpy import array, bincount
from numpy._typing import NDArray

from quantum_simulations.conditions.custom_condition import CustomCondition
from quantum_simulations.globals.configuration_simulation_manager import ConfigurationSimulationManager
from quantum_simulations.utilities.measurement_key_with_stable_hash import MeasurementKeyWithStableHash


class MajorityVote(CustomCondition):
    def __init__(self, desired_measurement_key: MeasurementKey, key: MeasurementKeyWithStableHash = None, number_of_votes: int = 0,):
        self.desired_measurement_key = desired_measurement_key
        self.key = key or MeasurementKeyWithStableHash(f'FAULT_TOLERANT_MEASUREMENT_{uuid4().hex}')
        self.number_of_votes = number_of_votes or ConfigurationSimulationManager().get_configuration().majority_vote_repetitions
        self._start_index = 0

    @property
    def keys(self):
        return (self.key,)

    def replace_key(self, current: MeasurementKey, replacement: MeasurementKey):
        return MajorityVote(replacement) if self.key == current else self

    def __str__(self):
        return str(self.key)

    def __repr__(self):
        return f'{json_cirq_type(type(self))}({self.desired_measurement_key!r}, {self.key!r}, {self.number_of_votes})'

    def resolve(self, classical_data: ClassicalDataDictionaryStore) -> bool:
        if self.key not in classical_data.keys():
            raise ValueError(f'Measurement key {self.key} missing when majority voting.')
        measurements = self._get_measurements(classical_data=classical_data)
        latest_measurements = measurements[self._start_index:]
        num_measurements = len(latest_measurements)
        if num_measurements == self.number_of_votes:
            majority = int(bincount(latest_measurements).argmax())
            classical_data.record_measurement(key=self.desired_measurement_key,
                                              measurement=(majority,),
                                              qubits=classical_data.measured_qubits[self.key][0],)
            self._start_index += self.number_of_votes
            return True
        return False

    def _get_measurements(self, classical_data: ClassicalDataDictionaryStore) -> NDArray[list[int]]:
        num_measurements = len(classical_data.records[self.key])
        return array([classical_data.get_int(self.key, i) for i in range(num_measurements)])

    def _json_dict_(self):
        return obj_to_dict_helper(self, ['desired_measurement_key', 'key', 'number_of_votes'])

    @classmethod
    def _from_json_dict_(cls, desired_measurement_key: MeasurementKey, key: MeasurementKeyWithStableHash, number_of_votes: int, **kwargs):
        return cls(desired_measurement_key=desired_measurement_key, key=key, number_of_votes=number_of_votes)

    @property
    def qasm(self):
        raise ValueError('QASM is defined only for SympyConditions of type key == constant.')
