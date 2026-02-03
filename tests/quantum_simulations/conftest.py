import pytest

from quantum_simulations.globals.configuration_simulation_manager import ConfigurationSimulationManager


@pytest.fixture(autouse=True)
def reset_configuration():
    ConfigurationSimulationManager.reset_configuration()
