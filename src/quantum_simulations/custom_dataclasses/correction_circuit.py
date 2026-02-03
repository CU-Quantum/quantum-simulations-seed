from dataclasses import dataclass, field

from cirq import Circuit


@dataclass
class CorrectionCircuit:
    syndrome_circuit: Circuit = field(default_factory=Circuit)
    recovery_circuit: Circuit = field(default_factory=Circuit)

    @property
    def full_circuit(self) -> Circuit:
        return Circuit(
            self.syndrome_circuit,
            self.recovery_circuit,
        )
