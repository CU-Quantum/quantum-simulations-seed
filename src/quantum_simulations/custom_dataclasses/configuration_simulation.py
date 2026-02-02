from dataclasses import dataclass
from typing import Optional


@dataclass
class ConfigurationSimulation:
    seed: Optional[int]
