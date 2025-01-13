from typing import Dict
from .reward_function import RewardFunction
from game.dating_game import DatingGame
from numpy import float32


class CoefficientReward(RewardFunction):
    """
    Reward function that gives:
    1. 0.2 points for each match
    2. Additional reward equal to match's attractiveness (0-1)
    """
    def __init__(self):
        """
        Initialize reward function with fixed rewards:
        - 0.2 points per match
        - 0-1 points based on match attractiveness
        """
        self.match_reward = 0.2
    
    def calculate_reward(self, game: DatingGame) -> Dict[str, float32]:
        result = {}
        matches = game.get_all_matches()
        traits = game.get_all_traits()
        
        for player in game.players:
            total_reward = 0
            player_matches = matches[player]
            
            for match in player_matches:
                total_reward += self.match_reward
                
                match_attractiveness = traits[match][0] 
                total_reward += match_attractiveness
            
            result[player] = float32(total_reward)
        
        return result

    def __repr__(self):
        return (f"{self.__class__.__name__} calculates reward as: "
                f"{self.match_reward} points per match + "
                f"match's attractiveness value (0-1)")