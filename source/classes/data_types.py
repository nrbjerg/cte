
from numpy.typing import ArrayLike
from dataclasses import dataclass
from typing import Tuple

Position = ArrayLike
Angle = float

@dataclass
class State:
    pos: Position
    angle: Angle

    def to_tuple(self) -> Tuple[float, float, float]:
        """Converts the state to a tuple."""
        return (self.pos[0], self.pos[1], self.angle)
        