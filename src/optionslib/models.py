from dataclasses import dataclass, field
import numpy as np

@dataclass(frozen=True)
class GBMModel:
    volatility : float
    
    def __post_init__(self):
        if self.volatility<=0:
            raise ValueError(f'Volatility sigma must be positive, got {self.volatility}')