from dataclasses import dataclass

import numpy
import pytest
from cirq import Circuit, Gate, LineQubit, Operation, X, Y, Z

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
from quantum_simulations.utilities.utilities import TYPE_STATE_VECTOR_OR_DENSITY_MATRIX, states_are_equal, tensor
from tests.quantum_simulations.error_correcting_codes.five_qubit_code.expected_states_five_qubit import \
    ExpectedStatesFiveQubit
from tests.quantum_simulations.error_correcting_codes.generalized_shor_code_hadamard.expected_states_generalized_shor_hadamard import \
    ExpectedStatesGeneralizedShorHadamard
from tests.quantum_simulations.error_correcting_codes.repetition_code.expected_states_repetition import \
    ExpectedStatesRepetition
from tests.quantum_simulations.error_correcting_codes.shors_code.expected_states_shor import ExpectedStatesShor
from tests.quantum_simulations.error_correcting_codes.stabilizer_standardized_code.expected_states_standardized_5_qubit import \
    ExpectedStatesGenericFiveQubit
from tests.quantum_simulations.error_correcting_codes.steane_code.expected_states_steane import ExpectedStatesSteane
from tests.quantum_simulations.utilities_for_tests import get_cat_state_vector, \
    set_configuration_to_reduce_ancilla_qubits

QUBIT_INDICES_IN_DIFFERENT_POSITIONS_IN_DIFFERENT_SHOR_BLOCKS = [0, 4, 8]
ARBITRARY_QUBIT_INDICES = [0, 2, 6]
QUBIT_INDICES_IN_DIFFERENT_POSITIONS_IN_DIFFERENT_CAT_PARITY_CODE_SUBREGISTERS = [
    0,
    ExpectedStatesGeneralizedShorHadamard().num_qubits_per_cat + 1,
    ExpectedStatesGeneralizedShorHadamard().num_qubits_per_cat * ExpectedStatesGeneralizedShorHadamard().num_cats - 1
]


@dataclass
class ParametersForCorrectionsTest:
    code: ErrorCorrectingCode
    initial_state: TYPE_STATE_VECTOR_OR_DENSITY_MATRIX
    qubit_indices_to_test: list[int]


SINGLE_ERROR_PARAMETERS = {
    "GeneralizedShorCode": ParametersForCorrectionsTest(
        code=GeneralizedShorCode(num_cats=3,
                                 num_qubits_per_cat=3),
        initial_state=tensor(*[get_cat_state_vector(num_qubits=3)] * 3),
        qubit_indices_to_test=[6]
    ),
    "RepetitionCode": ParametersForCorrectionsTest(
        code=RepetitionCodeOneLogical(num_qubits=ExpectedStatesRepetition().arbitrary_num_qubits),
        initial_state=ExpectedStatesRepetition().get_logical_zero_state_vector(),
        qubit_indices_to_test=list(range(3)),
    ),
    "GeneralizedShorCodeHadamardZeroState": ParametersForCorrectionsTest(
        code=GeneralizedShorCodeHadamard(num_cats=ExpectedStatesGeneralizedShorHadamard().num_cats,
                                         num_qubits_per_cat=ExpectedStatesGeneralizedShorHadamard().num_qubits_per_cat),
        initial_state=ExpectedStatesGeneralizedShorHadamard().get_logical_zero_state_vector(),
        qubit_indices_to_test=QUBIT_INDICES_IN_DIFFERENT_POSITIONS_IN_DIFFERENT_CAT_PARITY_CODE_SUBREGISTERS
    ),
    "GeneralizedShorCodeHadamardOneState": ParametersForCorrectionsTest(
        code=GeneralizedShorCodeHadamard(num_cats=ExpectedStatesGeneralizedShorHadamard().num_cats,
                                         num_qubits_per_cat=ExpectedStatesGeneralizedShorHadamard().num_qubits_per_cat),
        initial_state=ExpectedStatesGeneralizedShorHadamard().get_logical_one_state_vector(),
        qubit_indices_to_test=QUBIT_INDICES_IN_DIFFERENT_POSITIONS_IN_DIFFERENT_CAT_PARITY_CODE_SUBREGISTERS
    ),
    "GenericStabilizerCodeFiveQubit": ParametersForCorrectionsTest(
        code=StabilizerStandardizedCode(generators=get_check_matrix_values_5_qubit()),
        initial_state=ExpectedStatesGenericFiveQubit().get_logical_zero_state_vector(),
        qubit_indices_to_test=list(range(5)),
    ),
    "FiveQubitCode": ParametersForCorrectionsTest(
        code=FiveQubitCode(),
        initial_state=ExpectedStatesFiveQubit().get_logical_zero_state_vector(),
        qubit_indices_to_test=list(range(5)),
    ),
    "SteaneCode": ParametersForCorrectionsTest(
        code=SteaneCode(),
        initial_state=ExpectedStatesSteane().get_logical_zero_state_vector(),
        qubit_indices_to_test=ARBITRARY_QUBIT_INDICES
    ),
    "ShorsRepetitionCode": ParametersForCorrectionsTest(
        code=ShorsRepetitionCode(),
        initial_state=ExpectedStatesShor().get_logical_zero_state_vector(),
        qubit_indices_to_test=QUBIT_INDICES_IN_DIFFERENT_POSITIONS_IN_DIFFERENT_SHOR_BLOCKS
    ),
}


SINGLE_ERROR_PARAMETERS_FLATTENED = [pytest.param((parameters, qubit_index), id=f'{name}_qubit-{qubit_index}')
                                     for name, parameters in SINGLE_ERROR_PARAMETERS.items() for qubit_index in parameters.qubit_indices_to_test]


class TestCorrections:
    @pytest.fixture(autouse=True)
    def _setup(self):
        numpy.random.seed(0)
        set_configuration_to_reduce_ancilla_qubits()

    @pytest.mark.parametrize("req", SINGLE_ERROR_PARAMETERS_FLATTENED)
    @pytest.mark.parametrize("error_gate", [
        pytest.param(X, id='X'),
        pytest.param(Y, id='Y'),
        pytest.param(Z, id='Z')
    ])
    def test_one_error_is_corrected(self, error_gate: Gate, req: (ParametersForCorrectionsTest, int)):
        params = req[0]
        qubit_index = req[1]
        FreshAncillasPool().set_first_ancilla_num(first_ancilla_num=len(params.code.data_qubits))

        error_operations = [error_gate(LineQubit(qubit_index))]
        self._error_is_corrected(error_operations=error_operations, code=params.code, initial_state=params.initial_state)

    @pytest.mark.slow
    @pytest.mark.parametrize("params", [
        pytest.param((
                GeneralizedShorCodeHadamard(num_cats=5, num_qubits_per_cat=5),
                ExpectedStatesGeneralizedShorHadamard(num_cats=5, num_qubits_per_cat=5).get_logical_zero_state_vector(),
                [X(LineQubit(1)), X(LineQubit(2))]
        ), id='CatParityCode_2-Xs'),
    ])
    def test_multiple_errors_are_corrected(self, params: (ErrorCorrectingCode, TYPE_STATE_VECTOR_OR_DENSITY_MATRIX, list[Operation])):
        code = params[0]
        initial_state = params[1]
        error_operations = params[2]
        FreshAncillasPool().set_first_ancilla_num(first_ancilla_num=len(code.data_qubits))

        self._error_is_corrected(error_operations=error_operations, code=code, initial_state=initial_state)

    def _error_is_corrected(self,
                            error_operations: list[Operation],
                            code: ErrorCorrectingCode,
                            initial_state: TYPE_STATE_VECTOR_OR_DENSITY_MATRIX) -> None:
        circuit_simulator = CircuitSimulatorStateVector()
        simulation_state = circuit_simulator.run_simulation(
            circuit=Circuit(
                error_operations,
                code.get_error_correction_circuit().full_circuit,
            ),
            num_data_qubits=len(code.data_qubits),
            initial_data_state=initial_state,
        ).state
        assert states_are_equal(simulation_state, initial_state)
