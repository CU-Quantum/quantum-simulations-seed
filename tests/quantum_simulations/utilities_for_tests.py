import random
from math import sqrt

import numpy as np

from quantum_simulations.globals.configuration_simulation_manager import ConfigurationSimulationManager
from quantum_simulations.support.cat_state_creator.cat_state_creator_cx_from_first_qubit import \
    CatStateCreatorCxFromFirstQubit
from quantum_simulations.support.measurer.measurer_with_single_qubit_sequential import MeasurerWithSingleQubitSequential
from quantum_simulations.utilities.utilities import KET_ONE_STATE_VECTOR, KET_ZERO_STATE_VECTOR, TYPE_STATE_VECTOR, \
    tensor


def set_seed(seed: int):
    configuration = ConfigurationSimulationManager().get_configuration()
    configuration.seed = seed
    random.seed(seed)
    np.random.seed(seed)


def get_cat_state_vector(num_qubits: int) -> TYPE_STATE_VECTOR:
    return (1 / sqrt(2)) * (tensor(*[KET_ZERO_STATE_VECTOR] * num_qubits) + tensor(*[KET_ONE_STATE_VECTOR] * num_qubits))


def set_configuration_to_reduce_ancilla_qubits() -> None:
    configuration = ConfigurationSimulationManager().get_configuration()
    configuration.cat_state_creator_type = CatStateCreatorCxFromFirstQubit
    configuration.measurer_type = MeasurerWithSingleQubitSequential
