import os
from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam

from SAC.utils import plot_learning_curve
from agent import Agent
from buffer import ReplayBuffer
from environment.rFactor2Environment import RFactor2Environment
import environment.utils.fidgrovePluginUtils as utils


def sac(seed=0, skip_initial=False,
        alpha=0.2, gamma=0.99, tau=0.995, lr=1e-3,
        replay_size=1e6, batch_size=1024, update_after=2e3, update_every=25,
        steps_per_epoch=2000, epochs=100, max_ep_len=10000, start_steps=4e3):

    # ============================ INITIAL SETUP ============================
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = RFactor2Environment()

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    agent = Agent(input_dims=obs_dim, n_actions=act_dim,
                  gamma=gamma, tau=tau, lr=lr)
    agent_targ = deepcopy(agent)

    for p in agent_targ.parameters():
        p.requires_grad = False

    replay_size = replay_size
    replay_buffer = ReplayBuffer(input_shape=obs_dim[0], n_actions=act_dim, max_size=replay_size)

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(agent.actor.parameters(), lr=lr)
    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(agent.critic_1.parameters(), agent.critic_2.parameters())
    q_optimizer = Adam(q_params, lr=lr)

    total_steps = steps_per_epoch * epochs

    # ============================ LOSS FUNCTIONS ============================
    def compute_loss_q(data):
        q1 = agent.critic_1(data['state'], data['action'])
        q2 = agent.critic_2(data['state'], data['action'])

        with torch.no_grad():
            actions, log_probs = agent.actor.sample_normal(data['state_'])

            q1_target = agent_targ.critic_1(data['state'], actions)
            q2_target = agent_targ.critic_2(data['state'], actions)
            q_target = torch.min(q1_target, q2_target)
            backup = data['reward'] + gamma * (1 - data['done']) * (q_target - alpha * log_probs)

        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        return loss_q

    def compute_loss_pi(data):
        actions, log_probs = agent.actor.sample_normal(data['state'])
        q1_pi = agent.critic_1(data['state'], actions)
        q2_pi = agent.critic_2(data['state'], actions)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss (making it a soft policy)
        loss_pi = (alpha * log_probs - q_pi).mean()

        return loss_pi

    # ============================ UPDATE ============================
    def learn():
        # Obtain batch data
        data = replay_buffer.sample_buffer(batch_size)

        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Freeze Q-Networks
        for p_update in q_params:
            p_update.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-networks as to optimize them
        for p_update in q_params:
            p_update.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p_update, p_targ in zip(agent.parameters(), agent_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(tau)
                p_targ.data.add_((1 - tau) * p_update.data)

    # ============================ MAIN LOOP ============================

    # Reset as preparation for environment interation
    obs, ep_reward, ep_len = env.reset(), 0, 0
    best_reward = 0

    score_history = []

    # if skip_initial:
    #     agent.load_models()
    #     agent_targ.load_models()

    # Main loop
    step_count = 0
    while step_count < total_steps:
        if step_count < start_steps and not skip_initial:
            action = env.action_space.sample()
        else:
            action = agent.choose_action(obs)

        # Environment step
        obs_, reward, done, _, _ = env.step(action)
        if utils.calculate_throttle_action(utils.convert_mps_to_kph(obs_[0])) > 0.8 and obs[0] == obs_[0]:
            # Ignore step while at red light
            continue

        step_count += 1
        ep_reward += reward
        ep_len += 1

        # Store experience to replay buffer
        replay_buffer.store_transition(obs, action, reward, obs_, done)

        # Update most recent observation
        obs = obs_

        if done:
            obs, ep_reward, ep_len = env.reset(), 0, 0

        # Handling Agent Learning
        if step_count >= update_after and step_count % update_every == 0:
            for i in range(update_every):
                learn()

        score_history.append(ep_reward)
        avg_score = np.mean(score_history[-100:])
        if ep_reward > best_reward:
            best_reward = ep_reward

        if avg_score > best_reward:
            agent.save_models()
            agent_targ.save_models()

    # Plot learning curve
    filename = 'inverted_pendulum.png'
    figure_file = os.path.join(os.path.abspath('SAC\\plots'), filename)
    print(figure_file)

    score = [i + 1 for i in range(len(score_history))]
    plot_learning_curve(score, score_history, figure_file, len(score_history))

    # def process_transitions(actions_df, states_df):
    #     print("Processing initial transitions")
    #     timer = time.process_time()
    #     actions = []
    #     updates = 0
    #
    #     previous_states_df = states_df['Previous State'].apply(lambda x: x.strip('[]').split(','))
    #     new_states_df = states_df['New State'].apply(lambda x: x.strip('[]').split(','))
    #
    #     for index, act in actions_df.iterrows():
    #         act = np.array(action[0])
    #
    #         prev_state = np.array(previous_states_df[index], dtype=float)
    #         new_state = np.array(new_states_df[index], dtype=float)
    #
    #         actions.append(action)
    #
    #         d = utils.episode_finish(prev_state, new_state)
    #         r = utils.calculate_reward(prev_state, new_state)
    #
    #         replay_buffer.store_transition(prev_state, act, r, new_state, d)
    #
    #         # Update parameters of all the networks
    #         learn()
    #         updates += 1
    #
    #     elapsed_time = time.process_time() - timer
    #     print(f"Initial inputs and parameter updates finished after {elapsed_time} seconds.")
    #
    #     return updates
