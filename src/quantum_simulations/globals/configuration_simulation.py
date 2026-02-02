from quantum_simulations.custom_dataclasses.configuration_simulation import ConfigurationSimulation


class ConfigurationSimulationManager:
    _configuration = None

    @classmethod
    def get_configuration(cls) -> 'ConfigurationSimulationManager':
        if cls._configuration is None:
            cls.reset_configuration()
        return cls._configuration

    @classmethod
    def reset_configuration(cls) -> None:
        cls._configuration = ConfigurationSimulation(
            seed=None,
        )
