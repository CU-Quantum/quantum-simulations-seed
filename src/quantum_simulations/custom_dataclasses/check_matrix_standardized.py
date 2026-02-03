from dataclasses import dataclass

import numpy
from numpy.ma.core import allequal

from quantum_simulations.custom_dataclasses.check_matrix import CheckMatrix, TYPE_CHECK_MATRIX


@dataclass
class CheckMatrixSubmatrices:
    a1: TYPE_CHECK_MATRIX
    a2: TYPE_CHECK_MATRIX
    b: TYPE_CHECK_MATRIX
    c: TYPE_CHECK_MATRIX
    d: TYPE_CHECK_MATRIX
    e: TYPE_CHECK_MATRIX

    def __eq__(self, other):
        return (self.a1.tolist() == other.a1.tolist()
                and self.a2.tolist() == other.a2.tolist()
                and self.b.tolist() == other.b.tolist()
                and self.c.tolist() == other.c.tolist()
                and self.d.tolist() == other.d.tolist()
                and self.e.tolist() == other.e.tolist())


@dataclass
class CheckMatrixStandardized(CheckMatrix):
    def __post_init__(self):
        super().__post_init__()

        pauli_x_portion_identity = self.matrix[:self.rank_of_pauli_x_portion, :self.rank_of_pauli_x_portion]
        if not allequal(pauli_x_portion_identity, numpy.identity(self.rank_of_pauli_x_portion)):
            raise ValueError("The first (r)x(r) submatrix must be the identity.")

        pauli_x_portion_zeros = self.matrix[self.rank_of_pauli_x_portion:, :self.num_physical_qubits]
        if any(pauli_x_portion_zeros.flatten()):
            raise ValueError("All rows in the pauli_x portion below the rank must be 0.")

        pauli_z_portion_zeros = self.matrix[:self.rank_of_pauli_x_portion, self.num_physical_qubits + self.rank_of_pauli_x_portion:-self.num_logical_qubits]
        if any(pauli_z_portion_zeros.flatten()):
            raise ValueError("The (r)x(n-k-r) submatrix beginning at index [0, n+r] must be 0.")

        pauli_z_portion_identity = self.matrix[self.rank_of_pauli_x_portion:, self.num_physical_qubits + self.rank_of_pauli_x_portion:-self.num_logical_qubits]
        if not allequal(pauli_z_portion_identity, numpy.identity(self.num_physical_qubits - self.num_logical_qubits - self.rank_of_pauli_x_portion)):
            raise ValueError("The (n-k-r)x(n-k-r) submatrix beginning at index [r, n+r] must be the identity.")

        self.logical_xs = self._get_logical_xs()
        self.logical_zs = self._get_logical_zs()

    def _get_logical_xs(self) -> TYPE_CHECK_MATRIX:
        return numpy.concatenate(
            [
                numpy.zeros((self.num_logical_qubits, self.rank_of_pauli_x_portion)),
                self.submatrices.e.transpose(),
                numpy.identity(self.num_logical_qubits),
                self.submatrices.c.transpose(),
                numpy.zeros(self.submatrices.e.transpose().shape),
                numpy.zeros((self.num_logical_qubits, self.num_logical_qubits))
            ],
            axis=1,
        )

    def _get_logical_zs(self) -> TYPE_CHECK_MATRIX:
        return numpy.concatenate(
            [
                numpy.zeros((self.num_logical_qubits, self.rank_of_pauli_x_portion)),
                numpy.zeros(self.submatrices.e.transpose().shape),
                numpy.zeros((self.num_logical_qubits, self.num_logical_qubits)),
                self.submatrices.a2.transpose(),
                numpy.zeros(self.submatrices.e.transpose().shape),
                numpy.identity(self.num_logical_qubits)
            ],
            axis=1,
        )

    @property
    def submatrices(self) -> CheckMatrixSubmatrices:
        return CheckMatrixSubmatrices(
            a1=self.matrix[:self.rank_of_pauli_x_portion, self.rank_of_pauli_x_portion:self.num_physical_qubits - self.num_logical_qubits],
            a2=self.matrix[:self.rank_of_pauli_x_portion, self.num_physical_qubits - self.num_logical_qubits:self.num_physical_qubits],
            b=self.matrix[:self.rank_of_pauli_x_portion, self.num_physical_qubits:self.num_physical_qubits + self.rank_of_pauli_x_portion],
            c=self.matrix[:self.rank_of_pauli_x_portion, -self.num_logical_qubits:],
            d=self.matrix[self.rank_of_pauli_x_portion:, self.num_physical_qubits:self.num_physical_qubits + self.rank_of_pauli_x_portion],
            e=self.matrix[self.rank_of_pauli_x_portion:, -self.num_logical_qubits:],
        )

    def __eq__(self, other):
        return self.matrix.tolist() == other.matrix.tolist() and self.qubit_order == other.qubit_order
