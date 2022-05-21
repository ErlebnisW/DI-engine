import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(100000):
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    print(f"observation:{observation}, reward:{reward}")
env.close()