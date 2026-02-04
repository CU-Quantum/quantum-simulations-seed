from cirq import HasJSONNamespace, MEASUREMENT_KEY_SEPARATOR, MeasurementKey


class MeasurementKeyWithStableHash(MeasurementKey, HasJSONNamespace):
    """
    The base class uses caches the has value, which causes differences across multiple processes when pickling,
    leading to issues when looking for the key in a dict.

    Computing hash every time rather than caching solves this.
    """
    def __hash__(self):
        return hash(str(self))

    def __repr__(self):
        if self.path:
            return f"quantum_simulations.conditions.MeasurementKeyWithStableHash(path={self.path!r}, name='{self.name}')"
        else:
            return f"quantum_simulations.conditions.MeasurementKeyWithStableHash(name='{self.name}')"

    @classmethod
    def parse_serialized(cls, key_str: str) -> MeasurementKey:
        components = key_str.split(MEASUREMENT_KEY_SEPARATOR)
        return MeasurementKeyWithStableHash(name=components[-1], path=tuple(components[:-1]))

    @classmethod
    def _json_namespace_(cls):
        return 'quantum_simulations.utilities.measurement_key_with_stable_hash'
