from random import shuffle
import numpy as np

LEFT = 0
RIGHT = 1
NONE = 2


def generate_profiles(attributes, num_profiles):
    """ Randomly generate profiles based on attributes and their respective distributions inputted """
    profiles = np.zeros((num_profiles, len(attributes)))

    for i, attribute in enumerate(attributes):
        attr_config = attributes[attribute]
        distribution = attr_config['distribution']

        if distribution == 'normal':
            profiles[:, i] = np.random.normal(attr_config['mean'], attr_config['std'], num_profiles)
        elif distribution == 'uniform':
            profiles[:, i] = np.random.uniform(attr_config['low'], attr_config['high'], num_profiles)
        elif distribution == 'exponential':
            profiles[:, i] = np.random.exponential(attr_config['scale'], num_profiles)
        elif distribution == 'binary':
            profiles[:, i] = np.random.choice([0, 1], num_profiles, p=[1 - attr_config['prob'], attr_config['prob']])
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")

    return np.clip(profiles, 0, 1)


class DatingGame:

    def __init__(self, num_men, num_women, traits, men_max_swipes, women_max_swipes):
        self.men = ["man_" + str(r) for r in range(num_men)]
        self.women = ["woman_" + str(r) for r in range(num_women)]
        self.men_max_swipes = men_max_swipes
        self.women_max_swipes = women_max_swipes
        self.players = self.men + self.women
        self.trait_map = dict(zip(self.players, generate_profiles(traits, len(self.players))))
        self.queues = self.generate_queues()
        self.swipes = {player: set() for player in self.players}
        self.swiped_on = {player: set() for player in self.players}
        self.dones = {agent: False for agent in self.players}

    def generate_queues(self):
        male_queues = {man: [woman for woman in self.women] for man in self.men}
        female_queues = {woman: [man for man in self.men] for woman in self.women}
        queues = male_queues | female_queues

        for q in queues:
            shuffle(queues[q])

        return queues

    def get_players(self):
        return self.players.copy()

    def swipe(self, player, action):
        if self.dones[player]:
            if action != NONE:
                print("INCORRECT ACTION")
                exit(1)

        candidates = self.queues[player]

        if action == RIGHT:
            candidate = candidates[0]
            self.swipes[player].add(candidate)
            self.swiped_on[candidate].add(player)

        candidates.pop(0)

        num_remaining_candidates = len(candidates)
        max_swipes = self.men_max_swipes if player in self.men else self.women_max_swipes
        num_remaining_swipes = max_swipes - len(self.swipes[player])

        if num_remaining_candidates == 0 or num_remaining_swipes == 0:
            self.dones[player] = True

    def candidate_exists(self, player):
        return len(self.queues[player]) > 0

    def get_candidate_for(self, player):
        return self.queues[player][0]

    def get_traits_for(self, player):
        return self.trait_map[player]

    def get_all_traits(self):
        return self.trait_map.copy()
    
    def all_swipes_done(self):
        return all(self.dones.values())

    def get_swipes_for(self, player):
        return self.swipes[player].copy()

    def get_all_matches(self):
        matches = {}
        for player in self.players:
            swipes = self.swipes[player]
            swiped_on = self.swiped_on[player]
            matches[player] = swipes.intersection(swiped_on)

        return matches

