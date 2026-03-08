from gymnasium.envs.registration import register

register(
    id="chen1037_blocksworld_env/GridWorld-v0",
    entry_point="chen1037_blocksworld_env.envs:GridWorldEnv",
)

register(
    id="chen1037_blocksworld_env/BlocksWorld-v0",
    entry_point="chen1037_blocksworld_env.envs:BlocksWorldEnv",
    max_episode_steps=200,
)

register(
    id="chen1037_blocksworld_env/BlocksWorld-v1",
    entry_point="chen1037_blocksworld_env.envs:BlocksWorldTargetEnv",
    max_episode_steps=200,
)
