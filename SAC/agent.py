import torch as T
import torch.nn as nn
from networks import ActorNetwork, CriticNetwork


class Agent(nn.Module):

    def __init__(self, alpha=0.0003, lr=0.0003, input_dims=4,
                 env=None, gamma=0.99, n_actions=2, tau=0.005):
        super().__init__()

        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions

        self.actor = ActorNetwork(alpha, lr, input_dims, n_actions=n_actions,
                                  name='actor')
        self.critic_1 = CriticNetwork(lr, input_dims, n_actions=n_actions,
                                      name='critic_1')
        self.critic_2 = CriticNetwork(lr, input_dims, n_actions=n_actions,
                                      name='critic_2')

        print(f"Input dims: {input_dims}")

    def choose_action(self, observation):
        actions, _ = self.actor.sample_normal(T.as_tensor(observation, dtype=T.float32), reparameterize=True)

        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

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
