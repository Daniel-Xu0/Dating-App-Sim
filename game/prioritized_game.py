from random import shuffle, randint
import numpy as np
from .dating_game import DatingGame, generate_profiles, LEFT, RIGHT, NONE

class PrioritizedDatingGame(DatingGame):
    def __init__(self, num_men, num_women, traits, men_max_swipes, women_max_swipes, 
                 min_window=5, max_window=10, min_buffer=3, max_buffer=7):
        """
        Extended DatingGame with randomized prioritized queues.
        
        Args:
            num_men (int): Number of male profiles
            num_women (int): Number of female profiles
            traits (dict): Dictionary of traits and their distributions
            men_max_swipes (int): Maximum number of swipes for men
            women_max_swipes (int): Maximum number of swipes for women
            min_window (int): Minimum size of priority window
            max_window (int): Maximum size of priority window
            min_buffer (int): Minimum profiles between priority profiles
            max_buffer (int): Maximum profiles between priority profiles
        """
        self.min_window = min_window
        self.max_window = max_window
        self.min_buffer = min_buffer
        self.max_buffer = max_buffer
        self.last_priority_position = {}
        
        super().__init__(num_men, num_women, traits, men_max_swipes, women_max_swipes)
        self.pending_views = {player: set() for player in self.players}
        self.last_priority_position = {player: 0 for player in self.players}

    def generate_queues(self):
        """Generate initial randomized queues for all players"""
        male_queues = {man: [woman for woman in self.women] for man in self.men}
        female_queues = {woman: [man for man in self.men] for woman in self.women}
        queues = male_queues | female_queues

        for q in queues:
            shuffle(queues[q])

        return queues

    def _get_next_valid_position(self, queue, current_pos):
        """
        Calculate the next valid position for a priority profile insertion
        considering the random buffer size.
        
        Args:
            queue (list): The queue to insert into
            current_pos (int): Current position in the queue
        
        Returns:
            int: Next valid position for insertion
        """
        buffer_size = randint(self.min_buffer, self.max_buffer)
        next_pos = current_pos + buffer_size
        return min(next_pos, len(queue))

    def _insert_with_random_window(self, queue, profile, last_position):
        """
        Insert a profile using a random window size and maintaining buffer zones
        
        Args:
            queue (list): The queue to insert into
            profile (str): The profile to insert
            last_position (int): Position of the last priority profile insertion
        
        Returns:
            int: Position where the profile was inserted
        """
        if len(queue) == 0:
            queue.append(profile)
            return 0

        # Get next valid position after buffer
        start_pos = self._get_next_valid_position(queue, last_position)
        window_size = randint(self.min_window, self.max_window)
        end_pos = min(start_pos + window_size, len(queue))
        
        if start_pos >= len(queue):
            queue.append(profile)
            return len(queue) - 1
            
        # Random position within the window
        insert_position = randint(start_pos, end_pos)
        queue.insert(insert_position, profile)
        return insert_position

    def swipe(self, player, action):
        """
        Enhanced swipe method that handles priority queue updates with random windows
        and buffer zones
        
        Args:
            player (str): The player making the swipe
            action (int): The swipe action (LEFT, RIGHT, or NONE)
        """
        if self.dones[player]:
            if action != NONE:
                print("INCORRECT ACTION")
                exit(1)
            return

        candidates = self.queues[player]
        
        if len(candidates) == 0:
            self.dones[player] = True
            return

        candidate = candidates[0]

        if action == RIGHT:
            self.swipes[player].add(candidate)
            self.swiped_on[candidate].add(player)
            if player not in self.swipes[candidate] and player not in self.queues[candidate]:
                self.pending_views[candidate].add(player)

        candidates.pop(0)

        # Process any pending views for this player with random windows and buffers
        if self.pending_views[player]:
            pending_profiles = list(self.pending_views[player])
            for profile in pending_profiles:
                if profile not in candidates:
                    new_pos = self._insert_with_random_window(
                        candidates,
                        profile,
                        self.last_priority_position[player]
                    )
                    self.last_priority_position[player] = new_pos
            self.pending_views[player].clear()

        num_remaining_candidates = len(candidates)
        max_swipes = self.men_max_swipes if player in self.men else self.women_max_swipes
        num_remaining_swipes = max_swipes - len(self.swipes[player])

        if num_remaining_candidates == 0 or num_remaining_swipes == 0:
            self.dones[player] = True

    def __repr__(self):
        return (
            f"<PrioritizedDatingGame(num_men={len(self.men)}, num_women={len(self.women)}, "
            f"window_range={self.min_window}-{self.max_window}, "
            f"buffer_range={self.min_buffer}-{self.max_buffer}, "
            f"men_swipes={self.men_max_swipes}, women_swipes={self.women_max_swipes})>"
        )