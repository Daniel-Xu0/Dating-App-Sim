from functools import lru_cache
from helper.timing import timer

from reward.reward_function import RewardFunction
from game.prioritized_game import PrioritizedDatingGame, NONE

import gymnasium
import numpy as np
from gymnasium.spaces import Discrete, Box

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers


def env(config, reward_function: RewardFunction, render_mode=None):
    """
    The env function wraps the environment in wrappers by default.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    environment = PrioritizedDatingEnv(config, reward_function, render_mode=internal_render_mode)
    if render_mode == "ansi":
        environment = wrappers.CaptureStdoutWrapper(environment)
    environment = wrappers.AssertOutOfBoundsWrapper(environment)
    environment = wrappers.OrderEnforcingWrapper(environment)
    return environment


class PrioritizedDatingEnv(AECEnv):
    metadata = {"render_modes": ["human"], "name": "prioritized_dating_v1"}

    def __init__(self, config, reward_function: RewardFunction, render_mode=None):
        super().__init__()
        self.num_men = config["num_men"]
        self.num_women = config["num_women"]
        self.attributes = config["attributes"]
        self.men_max_swipes = config["men_max_swipes"]
        self.women_max_swipes = config["women_max_swipes"]
        
        self.min_window = config.get("min_window", 5)
        self.max_window = config.get("max_window", 10)
        self.min_buffer = config.get("min_buffer", 3)
        self.max_buffer = config.get("max_buffer", 7)
        
        self._game = self.create_game()
        self.rewards = {agent: 0 for agent in self._game.players}
        self.reward_function = reward_function
        self.render_mode = render_mode
        self.observations = {agent: [0] for agent in self._game.players}
        self._agent_selector = None

    def create_game(self):
        return PrioritizedDatingGame(
            self.num_men,
            self.num_women,
            self.attributes,
            self.men_max_swipes,
            self.women_max_swipes,
            self.min_window,
            self.max_window,
            self.min_buffer,
            self.max_buffer
        )

    @property
    def game(self):
        return self._game

    @lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

    @lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(3)

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return
        print(self)

    def observe(self, agent):
        return np.array(self.observations[agent])

    def close(self):
        pass

    def reset(self, seed=None, options=None):
        game = self.create_game()
        self._game = game
        self.agents = game.get_players()
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.observations = {agent: game.trait_map[game.queues[agent][0]] for agent in self.agents}
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    @timer
    def step(self, action):
        if action == NONE:
            self.agent_selection = self._agent_selector.next()
            return

        if (
                self.terminations[self.agent_selection]
                or self.truncations[self.agent_selection]
        ):
            action = None
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        game = self.game
        game.swipe(agent, action)

        if game.candidate_exists(agent):
            candidate = game.get_candidate_for(agent)
            self.observations[agent] = game.get_traits_for(candidate)

        if game.all_swipes_done():
            self.terminations = {agent: True for agent in self.agents}

        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

    def __repr__(self):
        return (
            f"<PrioritizedDatingEnv(num_men={self.num_men}, num_women={self.num_women}, "
            f"window_range={self.min_window}-{self.max_window}, buffer_range={self.min_buffer}-{self.max_buffer}, "
            f"men_swipes={self.men_max_swipes}, women_swipes={self.women_max_swipes}, "
            f"render_mode={self.render_mode})>"
        )