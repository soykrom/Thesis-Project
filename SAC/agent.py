import torch as T
import torch.nn as nn
from networks import ActorNetwork, CriticNetwork


class Agent(nn.Module):

    def __init__(self, alpha=0.0003, beta=0.0003, input_dims=4,
                 env=None, gamma=0.99, n_actions=2, tau=0.005):
        super().__init__()

        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions

        self.actor = ActorNetwork(alpha, beta, input_dims, n_actions=n_actions,
                                  name='actor', max_action=env.action_space.high)
        self.critic_1 = CriticNetwork(beta, input_dims, n_actions=n_actions,
                                      name='critic_1')
        self.critic_2 = CriticNetwork(beta, input_dims, n_actions=n_actions,
                                      name='critic_2')

        print(f"Input dims: {input_dims}")

    def choose_action(self, observation):
        actions, _ = self.actor.sample_normal(T.as_tensor(observation, dtype=T.float32), reparameterize=True)

        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    # def update_network_parameters(self, tau=None):
    #     if tau is None:
    #         tau = self.tau
    #
    #     target_value_params = self.target_value.named_parameters()
    #     value_params = self.value.named_parameters()
    #
    #     target_value_state_dict = dict(target_value_params)
    #     value_state_dict = dict(value_params)
    #
    #     for name in value_state_dict:
    #         value_state_dict[name] = tau * value_state_dict[name].clone() + \
    #                                  (1 - tau) * target_value_state_dict[name].clone()
    #
    #     self.target_value.load_state_dict(value_state_dict)
    #
    # def learn(self):
    #     if self.memory.mem_cntr < self.batch_size:
    #         return
    #
    #     state, action, reward, new_state, done = \
    #         self.memory.sample_buffer(self.batch_size)
    #
    #     reward = T.tensor(reward, dtype=T.float32).to(self.actor.device)
    #     done = T.tensor(done).to(self.actor.device)
    #     state_ = T.tensor(new_state, dtype=T.float32).to(self.actor.device)
    #     state = T.tensor(state, dtype=T.float32).to(self.actor.device)
    #     action = T.tensor(action, dtype=T.float32).to(self.actor.device)
    #
    #     value = self.value(state).view(-1)
    #     value_ = self.target_value(state_).view(-1)
    #     # print(f"Value_: {value_}\n")
    #     value_[done] = 0.0
    #
    #     self.value_mean.append(T.mean(value_).item())
    #
    #     # Value network loss
    #     actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
    #     log_probs = log_probs.view(-1)
    #
    #     critic_value = self.obtain_critic_value(state, actions)
    #
    #     self.value.optimizer.zero_grad()
    #     value_target = critic_value - log_probs
    #     value_loss = 0.5 * F.mse_loss(value, value_target)
    #     value_loss.backward(retain_graph=True)
    #     self.value.optimizer.step()
    #
    #     # Critic network loss
    #     self.critic_1.optimizer.zero_grad()
    #     self.critic_2.optimizer.zero_grad()
    #
    #     q1_old_policy = self.critic_1.forward(state, action).view(-1)
    #     q2_old_policy = self.critic_2.forward(state, action).view(-1)
    #
    #     q_hat = reward + self.gamma * value_
    #     # From spinningup_sac: q_hat = reward + self.gamma * (1 - done) *
    #     #       (target_critic_value - self.actor.alpha * actions_log_prob_next_state)
    #
    #     # Spinningup doesn't do this 0.5, but here they are averaged out. Good??
    #     critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
    #     critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)
    #
    #     critic_loss = critic_1_loss + critic_2_loss
    #     critic_loss.backward()
    #     self.critic_1.optimizer.step()
    #     self.critic_2.optimizer.step()
    #
    #     # Actor network loss
    #     self.actor.optimizer.zero_grad()
    #
    #     actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
    #     log_probs = log_probs.view(-1)
    #
    #     critic_value = self.obtain_critic_value(state, actions)
    #
    #     actor_loss = log_probs * self.actor.alpha - critic_value
    #     actor_loss = T.mean(actor_loss)
    #     actor_loss.backward()
    #     self.actor.optimizer.step()
    #
    #     self.update_network_parameters()
    #
    # def obtain_critic_value(self, state, actions):
    #     q1_new_policy = self.critic_1.forward(state, actions)
    #     q2_new_policy = self.critic_2.forward(state, actions)
    #     critic_value = T.min(q1_new_policy, q2_new_policy)
    #     critic_value = critic_value.view(-1)
    #
    #     return critic_value

    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
