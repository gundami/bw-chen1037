import gymnasium as gym
from gymnasium import spaces
import pygame
from screen import Display
from swiplserver import PrologMQI, PrologThread
import numpy as np


class BlocksWorldEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        # size parameter is kept for API compatibility but not used in blocks world

        # Start Prolog interpreter and load the blocks world knowledge base.
        # Python scripts must be run from the folder containing blocks_world.pl.
        self.mqi = PrologMQI()
        self.prolog_thread = self.mqi.create_thread()
        result = self.prolog_thread.query('[blocks_world]')

        # Build state dictionary mapping 3-digit Prolog state strings to integers.
        # Query state_helper/1 (the renamed 3-digit state predicate) to get all valid states.
        # result is like [{'State': '13a'}, {'State': '13b'}, ...]
        result = self.prolog_thread.query("state_helper(State)")
        self.states_dict = {d['State']: i for i, d in enumerate(result)}

        # Build action dictionary mapping action integers to Prolog action strings.
        # This allows us to convert gym integer actions back to Prolog predicates.
        self.actions_dict = {}
        result = self.prolog_thread.query("action(A)")
        # result is like this, where the first action is move(a,b,c):
        # [{'A': {'args': ['a', 'b', 'c'], 'functor': 'move'}}, ...]
        for i, A in enumerate(result):
            # Start building the action string with the functor name (e.g. "move")
            action_string = A['A']['functor']
            first = True
            for arg in A['A']['args']:
                if first:
                    # First argument: open the parenthesis
                    first = False
                    action_string += '('
                else:
                    # Subsequent arguments: separate with a comma
                    action_string += ','
                # Append each argument (block or place name)
                action_string += str(arg)
            # Close the parenthesis to complete the action string, e.g. "move(a,1,2)"
            action_string += ')'
            self.actions_dict[i] = action_string

        # Observation space: a single integer index into the states dictionary
        self.observation_space = spaces.Discrete(len(self.states_dict))

        # Action space: a single integer index into the actions dictionary
        self.action_space = spaces.Discrete(len(self.actions_dict))

        # Initial state and target (integers)
        self.state = 0
        self.target = 0

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.display = None
        if self.render_mode == "human":
            self.display = Display()

    def _state_int_to_str(self, state_int):
        """Convert a state integer back to its 3-digit Prolog state string."""
        return list(self.states_dict.keys())[list(self.states_dict.values()).index(state_int)]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Randomly choose a new target state (integer index)
        self.target = int(self.np_random.integers(0, len(self.states_dict)))

        # Update the display target string if rendering
        if self.display is not None:
            self.display.target = self._state_int_to_str(self.target)

        # Reset Prolog to the initial block configuration
        self.prolog_thread.query("reset")

        # Retrieve the current state from Prolog and convert to integer
        result = self.prolog_thread.query("current_state(State)")
        state_str = result[0]['State']
        self.state = self.states_dict[state_str]

        return self.state, {}

    def step(self, action):
        # Convert the integer action to a Prolog action string (e.g. "move(a,1,2)")
        action_string = self.actions_dict[action]

        # Attempt to perform the action in Prolog via step/1
        result = self.prolog_thread.query(f"step({action_string})")

        if result is not False:
            # Action succeeded: retrieve the updated current state
            result = self.prolog_thread.query("current_state(State)")
            state_str = result[0]['State']
            self.state = self.states_dict[state_str]
            reward = -1  # Valid move costs 1
        else:
            # Action was not possible in the current state
            reward = -10  # Penalty for impossible move

        # Episode ends when the agent reaches the target state
        terminated = (self.state == self.target)
        if terminated:
            reward = 100  # Bonus for reaching the goal

        if self.render_mode == "human" and self.display is not None:
            self.render()

        return self.state, reward, terminated, False, {}

    def render(self):
        if self.display is not None:
            state_str = self._state_int_to_str(self.state)
            self.display.step(state_str)

    def close(self):
        self.mqi.stop()
