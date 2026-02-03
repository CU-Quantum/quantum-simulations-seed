class GeneralizedShorCodeGenerators:
    def __init__(self, num_cats: int, num_qubits_per_cat: int):
        self._num_cats = num_cats
        self._num_qubits_per_cat = num_qubits_per_cat

    def get_z_generators(self) -> list[list[int]]:
        return [self._get_z_generator(i) for i in range(self._num_cats * (self._num_qubits_per_cat - 1))]

    def get_x_generators(self) -> list[list[int]]:
        return [self._get_x_generator(i) for i in range(self._num_cats - 1)]

    def _get_z_generator(self, i) -> list[int]:
        row = [0] * self._num_cats * self._num_qubits_per_cat * 2
        num_x_stabilizers = self._num_cats * self._num_qubits_per_cat
        cat_num = i // (self._num_qubits_per_cat - 1)
        cat_index = cat_num * self._num_qubits_per_cat
        qubit_index = cat_index + i % (self._num_qubits_per_cat - 1)
        base_index = num_x_stabilizers + qubit_index
        row[base_index] = 1
        row[base_index + 1] = 1
        return row

    def _get_x_generator(self, i) -> list[int]:
        row = [0] * self._num_cats * self._num_qubits_per_cat * 2
        index_low = i * self._num_qubits_per_cat
        index_high = index_low + 2 * self._num_qubits_per_cat
        row[index_low:index_high] = [1] * (index_high - index_low)
        return row
