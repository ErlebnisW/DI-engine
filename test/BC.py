import PPO
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm



class BehaviorClone:
    def __init__(self, state_dim, hidden_dim, action_dim, lr):
        self.policy = PPO.PolicyNet(state_dim, hidden_dim, action_dim).to(PPO.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    # train PolicyNet
    def learn(self, states, actions):
        states = torch.tensor(states, dtype=torch.float).to(PPO.device)
        actions = torch.tensor(actions).view(-1, 1).to(PPO.device)
        log_probs = torch.log(self.policy(states).gather(1, actions))
        bc_loss = torch.mean(-log_probs)

        self.optimizer.zero_grad()
        bc_loss.backward()
        self.optimizer.step()

    # take an action
    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(PPO.device)
        probs = self.policy(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()
    
def test_agent(agent, env, n_episode):
    return_list = []
    for episode in range(n_episode):
        episode_return = 0
        state = env.reset()
        done = False
        while not done:
            action = agent.take_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_return += reward
        return_list.append(episode_return)
    return np.mean(return_list)

PPO.env.seed(0)
torch.manual_seed(0)
np.random.seed(0)

lr = 1e-3
bc_agent = BehaviorClone(PPO.state_dim, PPO.hidden_dim, PPO.action_dim, lr)
n_iterations = 1000
batch_size = 64
test_returns = []

with tqdm(total=n_iterations, desc="进度条") as pbar:
    for i in range(n_iterations):
        sample_indices = np.random.randint(low=0, high=PPO.expert_s.shape[0], 
                    size=batch_size)
        bc_agent.learn(PPO.expert_s[sample_indices], PPO.expert_a[sample_indices])
        current_return = test_agent(bc_agent, PPO.env, 5)
        test_returns.append(current_return)
        if (i+1) % 10 == 0:
            pbar.set_postfix({'return': '%.3f' % np.mean(test_returns[-10:])})
        pbar.update(1)

iteration_list = list(range(len(test_returns)))
plt.plot(iteration_list, test_returns)
plt.xlabel('Iterations')
plt.ylabel('Returns')
plt.title('BC on {}'.format(PPO.env_name))
plt.show()
        
