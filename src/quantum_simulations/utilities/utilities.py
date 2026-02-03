from functools import reduce

import numpy as np
import sympy
from cirq import KET_MINUS, KET_ONE, KET_PLUS, KET_ZERO, LineQubit, MeasurementKey, Operation, X, \
    density_matrix_from_state_vector, kron
from numpy import allclose, array, log2, trace
from numpy._typing import NDArray

TYPE_STATE_VECTOR = NDArray[complex]
TYPE_DENSITY_MATRIX = NDArray[NDArray[complex]]
TYPE_STATE_VECTOR_OR_DENSITY_MATRIX = TYPE_DENSITY_MATRIX | TYPE_STATE_VECTOR

KET_ZERO_STATE_VECTOR = KET_ZERO.state_vector()
KET_ONE_STATE_VECTOR = KET_ONE.state_vector()
KET_PLUS_STATE_VECTOR = KET_PLUS.state_vector()
KET_MINUS_STATE_VECTOR = KET_MINUS.state_vector()

KET_ZERO_DENSITY_MATRIX = density_matrix_from_state_vector(KET_ZERO.state_vector())
KET_ONE_DENSITY_MATRIX = density_matrix_from_state_vector(KET_ONE.state_vector())
KET_PLUS_DENSITY_MATRIX = density_matrix_from_state_vector(KET_PLUS_STATE_VECTOR)


def get_num_qubits_in_state(state: TYPE_STATE_VECTOR_OR_DENSITY_MATRIX) -> int:
    return int(log2(state.shape[0]))


def is_state_vector(state: TYPE_STATE_VECTOR_OR_DENSITY_MATRIX) -> bool:
    return len(state.shape) == 1


def partial_trace(rho: TYPE_DENSITY_MATRIX, keep_qubits: list[int]) -> TYPE_DENSITY_MATRIX:
    """
    Compute the partial trace of a density matrix rho, keeping only the specified qubits.

    Parameters:
    rho : np.ndarray
        The density matrix (square matrix of size 2^n x 2^n).
    keep_qubits : list of int
        The qubits to keep (0-indexed).

    Returns:
    np.ndarray
        The reduced density matrix after tracing out the unspecified qubits.
    """
    dim = get_num_qubits_in_state(state=rho)
    if any(q >= dim or q < 0 for q in keep_qubits):
        raise ValueError("Qubit index out of range.")

    trace_out = [q for q in range(dim) if q not in keep_qubits]
    reshaped_rho = rho.reshape([2] * (2 * dim))

    for qubit in reversed(trace_out):
        reshaped_rho = trace(reshaped_rho, axis1=qubit, axis2=qubit + dim)
        dim -= 1

    reduced_dim = 2 ** dim
    return reshaped_rho.reshape((reduced_dim, reduced_dim))


def trace_out_ancillas_in_zero_state(state: TYPE_STATE_VECTOR_OR_DENSITY_MATRIX, num_ancillas: int) -> TYPE_STATE_VECTOR_OR_DENSITY_MATRIX:
        num_qubits = get_num_qubits_in_state(state=state)
        if is_state_vector(state=state):
            keep_indices = [not i % (2 ** num_ancillas) for i in range(len(state))]
            return state[keep_indices]
        else:
            return partial_trace(rho=state, keep_qubits=list(range(num_qubits - num_ancillas)))


def int_to_binary_array(num: int, num_elements: int) -> list[int]:
    return list(map(int, bin(num)[2:].rjust(num_elements, '0')))


def binary_array_to_int(binary_array: list[int]) -> int:
    return int(''.join(map(str, binary_array)), 2)


def tensor(*states: TYPE_STATE_VECTOR_OR_DENSITY_MATRIX) -> TYPE_STATE_VECTOR_OR_DENSITY_MATRIX:
    return kron(*states, shape_len=len(states[0].shape)) if states else array([])


def cx_sequentially_closer_qubits_from_first(qubits: list[LineQubit]) -> list[Operation]:
    return list(reversed(cx_sequentially_further_qubits_from_first(qubits=qubits)))


def cx_sequentially_further_qubits_from_first(qubits: list[LineQubit]) -> list[Operation]:
    return [X(qubits[i]).controlled_by(qubits[0]) for i in range(1, len(qubits))]


def states_are_equal(state1: TYPE_STATE_VECTOR_OR_DENSITY_MATRIX, state2: TYPE_STATE_VECTOR_OR_DENSITY_MATRIX) -> bool:
    element_wise_division = state1 / state2
    no_nans = element_wise_division[~np.isnan(element_wise_division)]
    has_global_phase = len(no_nans) and np.all(np.isclose(no_nans, no_nans[0], 1e-5))
    global_phase = no_nans[0] if has_global_phase else 1
    return np.isclose(abs(global_phase), 1, atol=1e-4) and allclose(state1 / global_phase, state2, atol=1e-7)


def get_sympy_conditions_all_equal(measurement_keys: list[MeasurementKey], values: list[int]) -> list[sympy.Expr]:
    measurement_symbols = sympy.symbols([measurement_key.name for measurement_key in measurement_keys])
    return [sympy.Eq(measurement_symbol, bit) for measurement_symbol, bit in zip(measurement_symbols, values)]
