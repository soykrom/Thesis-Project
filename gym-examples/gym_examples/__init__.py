from gym.envs.registration import register

register(
    id="gym_examples/GridWorld-v0",
    entry_point="gym_examples.envs:GridWorldEnv",
)

register(
    id='gym_examples/RFactor2-v0',
    entry_point='gym_examples.envs:RFactor2Environment',
    max_episode_steps=3000,
)