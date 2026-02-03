from dataclasses import dataclass

import pytest
from cirq import Circuit, I
from numpy import sqrt

from quantum_simulations.custom_dataclasses.logical_operation import LogicalGateLabel, LogicalOperation
from quantum_simulations.error_correcting_codes.error_correcting_code.error_correcting_code import ErrorCorrectingCode
from quantum_simulations.error_correcting_codes.five_qubit_code.five_qubit_code import FiveQubitCode
from quantum_simulations.error_correcting_codes.generalized_shor_code.generalized_shor_code import GeneralizedShorCode
from quantum_simulations.error_correcting_codes.generalized_shor_code_hadamard.generalized_shor_code_hadamard import \
    GeneralizedShorCodeHadamard
from quantum_simulations.error_correcting_codes.repetition_code.repetition_code import RepetitionCodeOneLogical
from quantum_simulations.error_correcting_codes.shors_code.shors_repetition_code import ShorsRepetitionCode
from quantum_simulations.error_correcting_codes.stabilizer_standardized_code.stabilizer_standardized_code import \
    StabilizerStandardizedCode
from quantum_simulations.error_correcting_codes.steane_code.staene_code import SteaneCode
from quantum_simulations.globals.fresh_ancillas_pool import FreshAncillasPool
from quantum_simulations.predefined_check_matrix_values import get_check_matrix_values_5_qubit
from quantum_simulations.simulators.circuit_simulator import CircuitSimulatorStateVector
from quantum_simulations.utilities.utilities import states_are_equal
from tests.quantum_simulations.error_correcting_codes.expected_states.expected_states import ExpectedStates
from tests.quantum_simulations.error_correcting_codes.five_qubit_code.expected_states_five_qubit import \
    ExpectedStatesFiveQubit
from tests.quantum_simulations.error_correcting_codes.generalized_shor_code.expected_states_generalized_shor import \
    ExpectedStatesGeneralizedShor
from tests.quantum_simulations.error_correcting_codes.generalized_shor_code_hadamard.expected_states_generalized_shor_hadamard import \
    ExpectedStatesGeneralizedShorHadamard
from tests.quantum_simulations.error_correcting_codes.repetition_code.expected_states_repetition import \
    ExpectedStatesRepetition
from tests.quantum_simulations.error_correcting_codes.shors_code.expected_states_shor import ExpectedStatesShor
from tests.quantum_simulations.error_correcting_codes.stabilizer_standardized_code.expected_states_standardized_5_qubit import \
    ExpectedStatesGenericFiveQubit
from tests.quantum_simulations.error_correcting_codes.steane_code.expected_states_steane import ExpectedStatesSteane
from tests.quantum_simulations.utilities_for_tests import set_configuration_to_reduce_ancilla_qubits


@dataclass
class ParametersForLogicalGatesTest:
    code: ErrorCorrectingCode
    expected_states: ExpectedStates


PARAMETERS = {
    "MultipleCatCode":ParametersForLogicalGatesTest(
        code=GeneralizedShorCode(num_cats=ExpectedStatesGeneralizedShor().arbitrary_num_cats,
                                 num_qubits_per_cat=ExpectedStatesGeneralizedShor().arbitrary_num_qubits_per_cat),
        expected_states=ExpectedStatesGeneralizedShor(),
    ),
    "RepetitionCode": ParametersForLogicalGatesTest(
        code=RepetitionCodeOneLogical(num_qubits=ExpectedStatesRepetition().arbitrary_num_qubits),
        expected_states=ExpectedStatesRepetition(),
    ),
    "CatParityCode": ParametersForLogicalGatesTest(
        code=GeneralizedShorCodeHadamard(num_cats=ExpectedStatesGeneralizedShorHadamard().num_cats,
                                         num_qubits_per_cat=ExpectedStatesGeneralizedShorHadamard().num_qubits_per_cat),
        expected_states=ExpectedStatesGeneralizedShorHadamard(),
    ),
    "GenericStabilizerCodeFiveQubit": ParametersForLogicalGatesTest(
        code=StabilizerStandardizedCode(generators=get_check_matrix_values_5_qubit()),
        expected_states=ExpectedStatesGenericFiveQubit()
    ),
    "FiveQubitCode": ParametersForLogicalGatesTest(
        code=FiveQubitCode(),
        expected_states=ExpectedStatesFiveQubit(),
    ),
    "SteaneCode": ParametersForLogicalGatesTest(
        code=SteaneCode(),
        expected_states=ExpectedStatesSteane(),
    ),
    "ShorsRepetitionCode": ParametersForLogicalGatesTest(
        code=ShorsRepetitionCode(),
        expected_states=ExpectedStatesShor(),
    ),
}


PARAMETERS_FLATTENED = [pytest.param(parameters, id=name) for name, parameters in PARAMETERS.items()]


class TestLogicalGates:
    @pytest.fixture(autouse=True, params=PARAMETERS_FLATTENED)
    def _setup(self, request):
        self._parameters: ParametersForLogicalGatesTest = request.param
        self._num_data_qubits = len(self._parameters.code.data_qubits)
        FreshAncillasPool().set_first_ancilla_num(first_ancilla_num=len(self._parameters.code.data_qubits))
        set_configuration_to_reduce_ancilla_qubits()

    def test_logical_x(self):
        target_index = self._parameters.code.num_logical_qubits - 1
        operation = LogicalOperation(gate=LogicalGateLabel.X, qubit_index=target_index)

        initial_data_state = self._parameters.expected_states.get_logical_zero_state_vector()
        expected_state = self._parameters.expected_states.get_logical_one_state_vector()

        circuit_simulator = CircuitSimulatorStateVector()
        current_state = circuit_simulator.run_simulation(
            circuit=Circuit(
                [I(qubit) for qubit in self._parameters.code.data_qubits],
                self._parameters.code.get_operation_circuit(operation=operation)
            ),
            num_data_qubits=self._num_data_qubits,
            initial_data_state=initial_data_state,
        ).state
        assert states_are_equal(current_state, expected_state)

    def test_logical_z(self):
        target_index = self._parameters.code.num_logical_qubits - 1
        operation = LogicalOperation(gate=LogicalGateLabel.Z, qubit_index=target_index)

        initial_data_state = (1 / sqrt(2)) * (
            self._parameters.expected_states.get_logical_zero_state_vector()
            + self._parameters.expected_states.get_logical_one_state_vector()
        )
        expected_state = (1 / sqrt(2)) * (
            self._parameters.expected_states.get_logical_zero_state_vector()
            - self._parameters.expected_states.get_logical_one_state_vector()
        )

        circuit_simulator = CircuitSimulatorStateVector()
        simulated_state = circuit_simulator.run_simulation(
            circuit=Circuit(
                [I(qubit) for qubit in self._parameters.code.data_qubits],
                self._parameters.code.get_operation_circuit(operation=operation)
            ),
            num_data_qubits=self._num_data_qubits,
            initial_data_state=initial_data_state,
        ).state
        assert states_are_equal(simulated_state, expected_state)
