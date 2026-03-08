"""
CST8509 Assignment 1 - Q-Learning Agent for BlocksWorld-v0
Chen1037 (Algonquin ID)

Runs Q-learning with 4 different hyperparameter configurations and plots
returns per episode and steps per episode for each run.
"""

import os
import gymnasium as gym
import chen1037_blocksworld_env  # registers BlocksWorld-v0 and BlocksWorld-v1
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def run_qlearning(alpha, gamma, epsilon_start, epsilon_min, epsilon_decay,
                  n_episodes, title):
    """
    Run Q-learning on BlocksWorld-v0.

    Parameters
    ----------
    alpha        : learning rate
    gamma        : discount factor
    epsilon_start: initial exploration rate
    epsilon_min  : minimum exploration rate
    epsilon_decay: multiplicative decay applied to epsilon each episode
    n_episodes   : number of training episodes
    title        : label used in plot titles and screenshot filename
    """
    env = gym.make("chen1037_blocksworld_env/BlocksWorld-v0")

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # Q-table initialised to zeros: shape (n_states, n_actions)
    Q = np.zeros((n_states, n_actions))

    epsilon = epsilon_start
    returns = []           # total reward per episode
    steps_per_episode = [] # number of steps per episode

    pbar = tqdm(range(n_episodes), desc=title, unit="ep")
    for ep in pbar:
        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done:
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = env.action_space.sample()   # explore
            else:
                action = int(np.argmax(Q[obs]))      # exploit

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Bellman update
            Q[obs, action] += alpha * (
                reward + gamma * np.max(Q[next_obs]) - Q[obs, action]
            )

            obs = next_obs
            total_reward += reward
            steps += 1

        # Decay exploration rate
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        returns.append(total_reward)
        steps_per_episode.append(steps)

        # Update progress bar postfix every episode
        pbar.set_postfix({
            "avg_return": f"{np.mean(returns[-50:]):.1f}",
            "steps": steps,
            "eps": f"{epsilon:.3f}",
        })

    env.close()

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(returns, alpha=0.6, label="Return")
    # Smoothed line for readability
    window = 20
    smoothed = np.convolve(returns, np.ones(window) / window, mode='valid')
    ax1.plot(range(window - 1, n_episodes), smoothed, color='red',
             label=f"Moving avg ({window})")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Return")
    ax1.set_title(f"Returns per Episode\n{title}")
    ax1.legend()

    ax2.plot(steps_per_episode, alpha=0.6, label="Steps")
    smoothed_steps = np.convolve(steps_per_episode, np.ones(window) / window,
                                 mode='valid')
    ax2.plot(range(window - 1, n_episodes), smoothed_steps, color='red',
             label=f"Moving avg ({window})")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Steps")
    ax2.set_title(f"Steps per Episode\n{title}")
    ax2.legend()

    plt.tight_layout()
    os.makedirs("screenshots", exist_ok=True)
    filename = os.path.join("screenshots",
                            title.replace(" ", "_").replace("=", "").replace(",", "") + ".png")
    plt.savefig(filename)
    print(f"Saved: {filename}")
    plt.show()

    return returns, steps_per_episode


# ── Experiment 1: Original Hyperparameters ──────────────────────────────────
run_qlearning(
    alpha=0.1,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.995,
    n_episodes=500,
    title="Original Hyperparameters",
)

# ── Experiment 2: Higher Learning Rate ──────────────────────────────────────
run_qlearning(
    alpha=0.5,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.995,
    n_episodes=500,
    title="alpha=0.5 gamma=0.99 decay=0.995",
)

# ── Experiment 3: Lower Discount Factor ─────────────────────────────────────
run_qlearning(
    alpha=0.1,
    gamma=0.5,
    epsilon_start=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.995,
    n_episodes=500,
    title="alpha=0.1 gamma=0.5 decay=0.995",
)

# ── Experiment 4: Slower Epsilon Decay ──────────────────────────────────────
run_qlearning(
    alpha=0.1,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.999,
    n_episodes=500,
    title="alpha=0.1 gamma=0.99 decay=0.999",
)
