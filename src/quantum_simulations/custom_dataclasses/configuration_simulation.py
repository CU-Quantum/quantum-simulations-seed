from dataclasses import dataclass
from typing import Optional

from quantum_simulations.support.cat_state_creator.cat_state_creator import CatStateCreator
from quantum_simulations.support.cat_state_creator.cat_state_creator_basic_nondeterministic.support.parity_verifier import \
    ParityVerifier
from quantum_simulations.support.measurer.measurer import Measurer


@dataclass
class ConfigurationSimulation:
    majority_vote_repetitions: int
    seed: Optional[int]

    cat_state_creator_type: type[CatStateCreator]
    measurer_type: type[Measurer]
    parity_verifier_type: type[ParityVerifier]
