from gym.envs.registration import register

register(
    id='foo-v0',
    # tags={'wrapper_config.TimeLimit.max_episode_steps': 200},

    entry_point='gym_foo.envs:FooEnv',
    # reward_threshold=4750.0
)
