"""
CST8509 Assignment 1 - Stable-Baselines3 DQN Agent for BlocksWorld-v1
Chen1037 (Algonquin ID)

Trains a DQN agent on the BlocksWorld-v1 environment (6-digit state that
includes both the current block configuration and the target configuration).
"""

import os
import gymnasium as gym
import chen1037_blocksworld_env  # registers BlocksWorld-v0 and BlocksWorld-v1
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback


class EpisodeLogger(BaseCallback):
    """Callback that records episode returns and lengths during training."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self._current_reward = 0.0
        self._current_length = 0

    def _on_step(self):
        self._current_reward += self.locals["rewards"][0]
        self._current_length += 1

        if self.locals["dones"][0]:
            self.episode_rewards.append(self._current_reward)
            self.episode_lengths.append(self._current_length)
            self._current_reward = 0.0
            self._current_length = 0
        return True


def main():
    env = gym.make("chen1037_blocksworld_env/BlocksWorld-v1")

    callback = EpisodeLogger()

    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-3,
        buffer_size=10000,
        learning_starts=500,
        batch_size=64,
        gamma=0.99,
        exploration_fraction=0.5,
        exploration_final_eps=0.05,
    )

    print("Training DQN on BlocksWorld-v1 ...")
    model.learn(total_timesteps=100_000, callback=callback)
    print("Training complete.")

    env.close()

    # ── Plot training curves ─────────────────────────────────────────────────
    rewards = callback.episode_rewards
    lengths = callback.episode_lengths
    n_episodes = len(rewards)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(rewards, alpha=0.5, label="Return")
    if n_episodes >= 20:
        window = 20
        smoothed = np.convolve(rewards, np.ones(window) / window, mode='valid')
        ax1.plot(range(window - 1, n_episodes), smoothed, color='red',
                 label=f"Moving avg ({window})")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Return")
    ax1.set_title("DQN Returns per Episode - BlocksWorld-v1")
    ax1.legend()

    ax2.plot(lengths, alpha=0.5, label="Steps")
    if n_episodes >= 20:
        smoothed_l = np.convolve(lengths, np.ones(window) / window, mode='valid')
        ax2.plot(range(window - 1, n_episodes), smoothed_l, color='red',
                 label=f"Moving avg ({window})")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Steps")
    ax2.set_title("DQN Steps per Episode - BlocksWorld-v1")
    ax2.legend()

    plt.tight_layout()
    os.makedirs("screenshots", exist_ok=True)
    filename = os.path.join("screenshots", "DQN_BlocksWorld_v1.png")
    plt.savefig(filename)
    print(f"Saved: {filename}")
    plt.show()

    # ── Quick evaluation ─────────────────────────────────────────────────────
    eval_env = gym.make("chen1037_blocksworld_env/BlocksWorld-v1")
    eval_rewards = []
    for _ in range(20):
        obs, _ = eval_env.reset()
        ep_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(int(action))
            done = terminated or truncated
            ep_reward += reward
        eval_rewards.append(ep_reward)
    eval_env.close()

    print(f"\nEvaluation over 20 episodes:")
    print(f"  Mean return : {np.mean(eval_rewards):.2f}")
    print(f"  Std  return : {np.std(eval_rewards):.2f}")


if __name__ == "__main__":
    main()
