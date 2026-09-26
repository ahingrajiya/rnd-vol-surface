from dataclasses import dataclass


@dataclass(frozen=True)
class GBMModel:
    volatility: float

    def __post_init__(self):
        if self.volatility <= 0:
            raise ValueError(
                f"Volatility sigma must be positive, got {self.volatility}"
            )
