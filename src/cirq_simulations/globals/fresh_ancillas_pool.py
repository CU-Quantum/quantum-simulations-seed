from contextlib import contextmanager
from typing import Generator

from cirq import LineQubit


class FreshAncillasPool:
    _pool: list[LineQubit] = []
    _next_ancilla_num = 0
    _parallel = False

    @classmethod
    def set_first_ancilla_num(cls, first_ancilla_num: int):
        if first_ancilla_num < 0:
            raise ValueError("First ancilla number must be non-negative.")
        cls._pool = []
        cls._next_ancilla_num = first_ancilla_num

    @contextmanager
    def use_fresh_ancillas(self, num_ancillas: int) -> Generator[list[LineQubit], None, None]:
        ancillas = []
        while len(ancillas) < num_ancillas:
            if self._pool:
                ancillas.append(self._pool.pop())
            else:
                ancillas.append(LineQubit(self._next_ancilla_num))
                self.__class__._next_ancilla_num += 1

        yield ancillas

        if not self._parallel:
            for ancillas in reversed(ancillas):
                self._pool.append(ancillas)

    @contextmanager
    def parallel(self, use_parallel: bool) -> Generator[None, None, None]:
        if use_parallel:
            self.__class__._parallel = True
            old_ancilla_num = self._next_ancilla_num
            old_pool = self._pool.copy()
            yield
            self.__class__._pool = old_pool
            self.__class__._next_ancilla_num = old_ancilla_num
            self.__class__._parallel = False
        else:
            yield
