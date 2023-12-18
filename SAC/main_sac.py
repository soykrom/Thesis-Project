import os
import argparse

import gym
import numpy as np
import pandas
from sac_torch import Agent
from utils import plot_learning_curve
from gym import wrappers

import fidgrovePluginUtils as utils
import rFactor2Environment


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--env-name', default='RFactor2-v0',
                        help='Mujoco Gym environment (default: RFactor2-v0)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--beta', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--n_episodes', type=int, default=250, metavar='N',
                        help='number of 250 (default: 250)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--epsilon', type=float, default=0.10, help='epsilon for epsilon greedy (default: 0.10')
    parser.add_argument('--input_file', default=os.path.abspath('common/inputs.csv'), help='file name with\
                                                                                         initial inputs')
    parser.add_argument('--states_file', default=os.path.abspath('common/transitions.csv'), help='file name with\
                                                                            state transitions of initial inputs')
    parser.add_argument('--skip_initial', type=bool, default=False, help='skip initial transitions training')
    parser.add_argument('--training_file', default=os.path.abspath('common/initial_training.pkl'),
                        help='file name with training inputs')
    parser.add_argument('--coefficients', nargs=3, type=float, default=None, help='to be used coefficients')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    print(f"Creating Environment {args.env_name}")
    env = gym.make(args.env_name)

    agent = Agent(alpha=args.alpha, gamma=args.gamma, tau=args.tau, beta=args.beta,
                  input_dims=env.observation_space.shape,
                  env=env, n_actions=env.action_space.shape[0],
                  layer1_size=args.hidden_size, layer2_size=args.hidden_size,
                  batch_size=args.batch_size)
    n_episodes = args.n_episodes

    # uncomment this line and do a mkdir models && mkdir video if you want to
    # record video of the agent playing the game.
    # env = wrappers.Monitor(env, 'models/video', video_callable=lambda episode_id: True, force=True)
    filename = 'inverted_pendulum.png'
    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    if args.skip_initial:
        updates = 0
        agent.load_models()
        states_df = pandas.read_csv(args.states_file)
        utils.plot(states_df['Previous State'].apply(lambda y: y.strip('[]').split(',')), agent)
    else:
        states_df = pandas.read_csv(args.states_file)
        updates = utils.process_transitions(pandas.read_csv(args.input_file, header=1),
                                            states_df,
                                            agent)
        utils.save_initial(args.training_file, agent)
        utils.plot(states_df['Previous State'].apply(lambda y: y.strip('[]').split(',')), agent)

    for i in range(n_episodes):
        observation = env.reset()[0]
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, _, _ = env.step(action)
            score += reward

            agent.remember(observation, action, reward, observation_, done)

            if not load_checkpoint:
                agent.learn()

            observation = observation_

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

    if not load_checkpoint:
        x = [i + 1 for i in range(n_episodes)]
        plot_learning_curve(x, score_history, figure_file)
