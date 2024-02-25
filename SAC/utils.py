import os

import numpy as np
import matplotlib.pyplot as plt
import argparse


def plot_learning_curve(x, scores, figure_file, n_episodes):
    running_avg = np.zeros(len(scores))

    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100):(i + 1)])

    plt.plot(x, running_avg)
    plt.title('Running average of previous ' + str(n_episodes) + ' scores')

    plt.savefig(figure_file)
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--env-name', default='RFactor2-v0',
                        help='Mujoco Gym environment (default: RFactor2-v0)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--beta', type=float, default=0.001, metavar='G',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                        help='number of start steps (default: 40000)')
    parser.add_argument('--n_episodes', type=int, default=250, metavar='N',
                        help='number of episodes (default: 250)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--replay_size', type=int, default=int(1e6), metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--epsilon', type=float, default=0.10, help='epsilon for epsilon greedy (default: 0.10')
    parser.add_argument('--input_file', default=os.path.abspath('environment/common/inputs.csv'), help='file name with\
                                                                                         initial inputs')
    parser.add_argument('--states_file', default=os.path.abspath('environment/common/transitions.csv'),
                        help='file name with state transitions of initial inputs')
    parser.add_argument('--skip_initial', type=bool, default=False, help='skip initial transitions training')
    parser.add_argument('--training_file', default=os.path.abspath('environment/common/initial_training.pkl'),
                        help='file name with training inputs')
    parser.add_argument('--coefficients', nargs=3, type=float, default=None, help='to be used coefficients')

    return parser.parse_args()
