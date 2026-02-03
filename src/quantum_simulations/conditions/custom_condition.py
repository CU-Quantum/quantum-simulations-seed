from abc import ABC, abstractmethod

from cirq import Condition, HasJSONNamespace, SupportsJSON


class CustomCondition(Condition, HasJSONNamespace, SupportsJSON, ABC):
    @abstractmethod
    def _json_dict_(self):
        pass

    @classmethod
    def _json_namespace_(cls):
        return 'quantum_simulations.conditions'
