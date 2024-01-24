import os
import argparse

import gym
import pandas
from spinup import sac_pytorch as sac
from gym import wrappers
import torch.nn as nn

import environment.utils.fidgrovePluginUtils as utils
import environment.rFactor2Environment


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--env-name', default='RFactor2-v0',
                        help='Mujoco Gym environment (default: RFactor2-v0)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--beta', type=float, default=0.001, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--n_episodes', type=int, default=1000, metavar='N',
                        help='number of episodes (default: 1000)')
    parser.add_argument('--n_start_steps', type=int, default=1000, metavar='N',
                        help='number of start steps for uniform-random action selection (default: 40000)')
    parser.add_argument('--n_steps_epoch', type=int, default=20000, metavar='N',
                        help='number of total steps per epoch (default: 20000)')
    parser.add_argument('--batch_size', type=int, default=4096, metavar='N',
                        help='batch size (default: 4096)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--input_file', default=os.path.abspath('common/inputs.csv'), help='file name with\
                                                                                         initial inputs')
    parser.add_argument('--states_file', default=os.path.abspath('common/transitions.csv'), help='file name with\
                                                                            state transitions of initial inputs')
    parser.add_argument('--skip_initial', type=bool, default=False, help='skip initial transitions training')
    parser.add_argument('--training_file', default=os.path.abspath('common/initial_training.pkl'),
                        help='file name with training inputs')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    print(f"Creating Environment {args.env_name}")
    env = gym.make(args.env_name)
    ac_kwargs = dict(hidden_sizes=[256, 256], activation=nn.functional.relu)
    logger_kwargs = dict(output_dir=os.path.abspath('epochLogs'), exp_name=args.env_name)

    sac(env, ac_kwargs=ac_kwargs, replay_size=args.replay_size, batch_size=args.batch_size,
        start_steps=args.start_steps, steps_per_epoch=args.n_steps_epoch, max_ep_len=args.n_episodes,
        gamma=args.gamma, lr=args.beta, alpha=args.alpha,
        logger_kwargs=logger_kwargs)

    load_checkpoint = False

    if args.skip_initial:
        updates = 0
        # agent.load_models()
        states_df = pandas.read_csv(args.states_file)
        # utils.plot(states_df['Previous State'].apply(lambda y: y.strip('[]').split(',')), agent)
    else:
        states_df = pandas.read_csv(args.states_file)
        # updates = utils.process_transitions(pandas.read_csv(args.input_file, header=1), states_df, agent)
        # agent.save_models()
        # utils.save_initial(args.training_file, agent)
        # utils.plot(states_df['Previous State'].apply(lambda y: y.strip('[]').split(',')), agent)

    for i in range(args.n_episodes):
        observation = env.reset()[0]
        done = False
        score = 0
        while not done:
            break
            # action = agent.choose_action(observation)
            # observation_, reward, done, _, _ = env.step(action)
            # score += reward

            # agent.remember(observation, action, reward, observation_, done)

            # if not load_checkpoint:
            #     agent.learn()

            # observation = observation_
