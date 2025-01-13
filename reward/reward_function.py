from typing import Dict

from game.dating_game import DatingGame
from abc import ABC, abstractmethod
from numpy import float32

"""
A generic interface for calculating the rewards for players in the dating game simulation.
It should produce a mapping of players in the game to each player in the game. 
"""


class RewardFunction(ABC):
    @abstractmethod
    def calculate_reward(self, game: DatingGame) -> Dict[str, float32]:
        pass
