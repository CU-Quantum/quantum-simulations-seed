from numpy import append, array, concatenate, flip
from numpy._typing import NDArray


class FlagSequenceGenerator:
    """
    Explained in https://quantum-journal.org/papers/q-2023-10-24-1154/
    """
    def __init__(self, num_flags: int):
        self._num_flags = num_flags

    def get_flag_sequence(self) -> NDArray[NDArray[int]]:
        if not self._num_flags:
            return array([[]])
        elif self._num_flags == 1:
            return array([[0], [1]])
        elif self._num_flags == 2:
            return array([[1, 0], [1, 1], [0, 1]])
        elif self._num_flags == 3:
            return array([[1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1], [0, 0, 1]])

        sequence_for_previous_num_flags = FlagSequenceGenerator(num_flags=self._num_flags - 1).get_flag_sequence()
        part_one = array([append(sub_array, 0) for sub_array in sequence_for_previous_num_flags[:-1]])

        part_two = flip(part_one.copy(), axis=0)
        part_two[:, -1] = 1
        part_two[:, 1], part_two[:, self._num_flags - 2] = part_two[:, self._num_flags - 2], part_two[:, 1].copy()

        part_three = [[0] * (self._num_flags - 1) + [1] for _ in range(2 * (self._num_flags - 3) + 1)]
        previous_through_step = 0
        for i, sub_array in enumerate(part_three[:-1]):
            next_through_step = 1 if i >= len(part_three) - 3 else previous_through_step + 1 + (not i)
            sub_array[previous_through_step] = 1
            if not i % 2:
                sub_array[next_through_step] = 1
                previous_through_step = next_through_step

        return concatenate([part_one, part_two, part_three])
