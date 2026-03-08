import gymnasium as gym
from gymnasium import spaces
import pygame
from screen import Display
from swiplserver import PrologMQI, PrologThread
import numpy as np


class BlocksWorldTargetEnv(gym.Env):
    """BlocksWorld-v1: 6-digit state encoding both the agent's current block
    positions (first 3 digits) and the target configuration (last 3 digits).
    This gives the agent full observability of what it needs to achieve."""

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        # size parameter is kept for API compatibility but not used

        # Start Prolog interpreter and load the blocks world knowledge base
        self.mqi = PrologMQI()
        self.prolog_thread = self.mqi.create_thread()
        result = self.prolog_thread.query('[blocks_world]')

        # Build 6-digit state dictionary using the state/1 predicate which
        # generates all combinations of (agent_3digit + target_3digit).
        # result is like [{'State': '13a13a'}, {'State': '13a13b'}, ...]
        result = self.prolog_thread.query("state(State)")
        self.states_dict = {d['State']: i for i, d in enumerate(result)}

        # Build action dictionary mapping action integers to Prolog action strings.
        # Same structure as BlocksWorld-v0.
        self.actions_dict = {}
        result = self.prolog_thread.query("action(A)")
        # result is like [{'A': {'args': ['a', 'b', 'c'], 'functor': 'move'}}, ...]
        for i, A in enumerate(result):
            # Build the action string e.g. "move(a,1,2)"
            action_string = A['A']['functor']
            first = True
            for arg in A['A']['args']:
                if first:
                    first = False
                    action_string += '('
                else:
                    action_string += ','
                action_string += str(arg)
            action_string += ')'
            self.actions_dict[i] = action_string

        # Observation space: integer index into the 6-digit states dictionary
        self.observation_space = spaces.Discrete(len(self.states_dict))

        # Action space: integer index into the actions dictionary
        self.action_space = spaces.Discrete(len(self.actions_dict))

        self.state = 0
        self.target_str = ""  # 3-digit target state string

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.display = None
        if self.render_mode == "human":
            self.display = Display()

    def _state_int_to_str(self, state_int):
        """Convert a state integer back to its 6-digit state string."""
        return list(self.states_dict.keys())[list(self.states_dict.values()).index(state_int)]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Pick a random 6-digit state and take the last 3 chars as the target.
        # This gives us a random valid 3-digit target configuration.
        rand_idx = int(self.np_random.integers(0, len(self.states_dict)))
        rand_state_str = self._state_int_to_str(rand_idx)
        self.target_str = rand_state_str[-3:]  # last 3 chars = target

        # Update display target if rendering
        if self.display is not None:
            self.display.target = self.target_str

        # Reset Prolog to the initial block configuration
        self.prolog_thread.query("reset")

        # Get 3-digit current state from Prolog, combine with target to form 6-digit obs
        result = self.prolog_thread.query("current_state(State)")
        agent_str = result[0]['State']  # 3-digit current agent state
        state_6digit = agent_str + self.target_str
        self.state = self.states_dict[state_6digit]

        return self.state, {}

    def step(self, action):
        action_string = self.actions_dict[action]

        # Attempt the action in Prolog
        result = self.prolog_thread.query(f"step({action_string})")

        if result is not False:
            # Action succeeded: get new 3-digit state from Prolog
            result = self.prolog_thread.query("current_state(State)")
            agent_str = result[0]['State']
            # Append saved target to form the 6-digit observation
            state_6digit = agent_str + self.target_str
            self.state = self.states_dict[state_6digit]
            reward = -1
        else:
            # Impossible move
            reward = -10

        # Goal: the first 3 digits of current state equal the target
        state_str = self._state_int_to_str(self.state)
        agent_str = state_str[:3]
        terminated = (agent_str == self.target_str)
        if terminated:
            reward = 100

        if self.render_mode == "human" and self.display is not None:
            self.render()

        return self.state, reward, terminated, False, {}

    def render(self):
        if self.display is not None:
            state_str = self._state_int_to_str(self.state)
            agent_str = state_str[:3]
            self.display.step(agent_str)

    def close(self):
        self.mqi.stop()
