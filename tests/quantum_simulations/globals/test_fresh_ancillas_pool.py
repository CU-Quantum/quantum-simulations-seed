import pytest
from cirq import LineQubit

from quantum_simulations.globals.fresh_ancillas_pool import FreshAncillasPool


class TestFreshAncillasPool:
    @pytest.fixture(autouse=True)
    def _setup(self):
        FreshAncillasPool().set_first_ancilla_num(0)

    def test_set_first_ancilla_num_invalid(self):
        invalid_first_ancilla_num = -1
        with pytest.raises(ValueError, match="^First ancilla number must be non-negative\\.$"):
            FreshAncillasPool.set_first_ancilla_num(invalid_first_ancilla_num)

    def test_use_fresh_ancillas_from_pool(self):
        FreshAncillasPool().set_first_ancilla_num(5)

        with FreshAncillasPool().use_fresh_ancillas(1) as ancilla_qubits:
            assert ancilla_qubits == [LineQubit(5)]

            with FreshAncillasPool().use_fresh_ancillas(2) as ancilla_qubits:
                assert ancilla_qubits == [LineQubit(6), LineQubit(7)]

        with FreshAncillasPool().use_fresh_ancillas(2) as ancilla_qubits:
            assert ancilla_qubits == [LineQubit(5), LineQubit(6)]

    def test_use_fresh_ancillas_mixed_source(self):
        FreshAncillasPool().set_first_ancilla_num(5)

        with FreshAncillasPool().use_fresh_ancillas(1) as ancilla_qubits:
            assert ancilla_qubits == [LineQubit(5)]

        with FreshAncillasPool().use_fresh_ancillas(2) as ancilla_qubits:
            assert ancilla_qubits == [LineQubit(5), LineQubit(6)]

    def test_set_first_ancilla_empties_pool(self):
        first_ancilla_num = 5
        FreshAncillasPool().set_first_ancilla_num(first_ancilla_num)
        with FreshAncillasPool().use_fresh_ancillas(1) as ancilla_qubits:
            assert ancilla_qubits == [LineQubit(5)]

        FreshAncillasPool().set_first_ancilla_num(4)
        with FreshAncillasPool().use_fresh_ancillas(1) as ancilla_qubits:
            assert ancilla_qubits == [LineQubit(4)]

    def test_parallel_forces_new_qubits(self):
        ancillas  = set()
        with FreshAncillasPool().parallel(True) as ancilla_qubits:
            with FreshAncillasPool().use_fresh_ancillas(1) as ancilla_qubits:
                ancillas.update(ancilla_qubits)
            with FreshAncillasPool().use_fresh_ancillas(1) as ancilla_qubits:
                ancillas.update(ancilla_qubits)
        assert len(ancillas) == 2

    def test_parallel_puts_used_ancillas_back_into_pool(self):
        with FreshAncillasPool().parallel(True):
            with FreshAncillasPool().use_fresh_ancillas(1) as ancilla_qubits:
                pass
        with FreshAncillasPool().use_fresh_ancillas(1) as ancilla_qubits:
            assert ancilla_qubits[0].x == 0

    def test_parallel_does_not_permanently_remove_from_pool(self):
        FreshAncillasPool().set_first_ancilla_num(1)
        with FreshAncillasPool().use_fresh_ancillas(1) as ancilla_qubits:
            pass
        with FreshAncillasPool().parallel(True):
            with FreshAncillasPool().use_fresh_ancillas(1) as ancilla_qubits:
                assert ancilla_qubits[0].x == 1
        with FreshAncillasPool().use_fresh_ancillas(2) as ancilla_qubits:
            assert ancilla_qubits == [LineQubit(1), LineQubit(2)]
