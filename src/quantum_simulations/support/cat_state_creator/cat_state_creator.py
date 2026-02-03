from abc import ABC, abstractmethod

from cirq import Circuit, LineQubit


class CatStateCreator(ABC):
    def __init__(self, qubit_register: list[LineQubit]):
        self._qubit_register = qubit_register

    @abstractmethod
    def get_cat_state_circuit(self) -> Circuit:
        pass

    @abstractmethod
    def decode_state(self) -> Circuit:
        pass
