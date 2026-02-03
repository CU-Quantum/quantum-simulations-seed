class ConfigurationSimulationManager:
    _configuration = None

    @classmethod
    def get_configuration(cls) -> 'ConfigurationSimulation':
        if cls._configuration is None:
            cls.reset_configuration()
        return cls._configuration

    @classmethod
    def reset_configuration(cls) -> None:
        from quantum_simulations.support.cat_state_creator.cat_state_creator_flag_pattern.cat_state_creator_flag_pattern import \
            CatStateCreatorFlagPattern
        from quantum_simulations.custom_dataclasses.configuration_simulation import ConfigurationSimulation
        from quantum_simulations.support.cat_state_creator.cat_state_creator_basic_nondeterministic.support.parity_verifier_sequential import \
            ParityVerifierSequential
        from quantum_simulations.support.measurer.fault_tolerant_measurer_sequential import \
            FaultTolerantMeasurerSequential
        cls._configuration = ConfigurationSimulation(
            cat_state_creator_type=CatStateCreatorFlagPattern,
            majority_vote_repetitions=3,
            measurer_type=FaultTolerantMeasurerSequential,
            parity_verifier_type=ParityVerifierSequential,
            seed=None,
        )
