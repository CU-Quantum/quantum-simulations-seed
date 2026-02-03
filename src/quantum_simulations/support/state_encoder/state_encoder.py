from abc import ABC, abstractmethod

from cirq import Circuit


class StateEncoder(ABC):
    @abstractmethod
    def encode_state(self) -> Circuit:
        pass
