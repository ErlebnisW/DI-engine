import gym
env = gym.make("MountainCar-v0")
print(env.observation_space)
print(env.action_space)
print(env.action_space.n)
print(env.observation_space.low)
print(env.observation_space.high)