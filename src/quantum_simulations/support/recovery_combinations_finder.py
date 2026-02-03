from collections import defaultdict
from dataclasses import replace
from itertools import combinations
from typing import Optional

import numpy as np
from cirq import LineQubit, X, Y, Z

from quantum_simulations.custom_dataclasses.recovery import RecoveryGate, RecoveryOperation


class RecoveryCombinationsFinder:
    def __init__(self, max_num_errors: Optional[int] = None, max_num_x_errors: Optional[int] = None, max_num_z_errors: Optional[int] = None):
        self._max_num_errors = max_num_errors
        self._max_num_x_errors = max_num_x_errors
        self._max_num_z_errors = max_num_z_errors

    def find_recovery_operations(self, single_error_recoveries: list[RecoveryOperation]) -> list[RecoveryOperation]:
        recovery_gates = [
            RecoveryGate(
                gate=recovery_operation.operation.gate,
                qubit_index=recovery_operation.operation.qubits[0].x,
                symptom=recovery_operation.symptom,
            )
            for recovery_operation in single_error_recoveries
        ]
        combos_gates = self.find_recovery_gates(single_error_recoveries=recovery_gates)
        recovery_operations = [
            RecoveryOperation(
                operation=recovery_operation.gate(LineQubit(recovery_operation.qubit_index)),
                symptom=recovery_operation.symptom,
            )
            for recovery_operation in combos_gates
        ]
        return recovery_operations

    def find_recovery_gates(self, single_error_recoveries: list[RecoveryGate]) -> list[RecoveryGate]:
        self._validate()
        recovery_gates_one_group_per_symptom = defaultdict(list)
        for num_errors in range(1, self._max_num_errors + 1):
            combos = combinations(single_error_recoveries, num_errors)
            for combo in combos:
                symptoms = [recovery_gate.symptom for recovery_gate in combo]
                symptoms_combined = np.mod(np.sum(symptoms, axis=0), 2).tolist()

                error_types = [recovery_gate.gate for recovery_gate in combo]
                y_errors = error_types.count(Y)
                x_is_below_limit = error_types.count(X) + y_errors <= self._max_num_x_errors
                z_is_below_limit = error_types.count(Z) + y_errors <= self._max_num_z_errors
                symptom_already_corrected = tuple(symptoms_combined) in recovery_gates_one_group_per_symptom
                if x_is_below_limit and z_is_below_limit and not symptom_already_corrected:
                    for recovery_gate in combo:
                        recovery_gates_one_group_per_symptom[tuple(symptoms_combined)].append(replace(recovery_gate, symptom=symptoms_combined))
        recovery_gates = [gate for gates in recovery_gates_one_group_per_symptom.values() for gate in gates]
        return recovery_gates

    def _validate(self) -> None:
        if self._max_num_errors is None and (self._max_num_x_errors is None or self._max_num_z_errors is None):
            raise ValueError(
                'Either "max_num_errors" must be provided or both "max_num_x_errors" and "max_num_z_errors" must be provided.'
            )

        if self._max_num_errors is None:
            self._max_num_errors = self._max_num_x_errors + self._max_num_z_errors
        if self._max_num_x_errors is None:
            self._max_num_x_errors = self._max_num_errors
        if self._max_num_z_errors is None:
            self._max_num_z_errors = self._max_num_errors
