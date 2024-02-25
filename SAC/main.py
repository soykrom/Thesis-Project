import os

import numpy as np
import pandas

import environment.utils.fidgrovePluginUtils as utils
from agent import Agent
from environment.rFactor2Environment import RFactor2Environment
from utils import parse_args
from utils import plot_learning_curve

if __name__ == '__main__':
    args = parse_args()

    print(f"Creating Environment {args.env_name}")
    # env = gym.make(args.env_name)
    env = RFactor2Environment()

    agent = Agent(alpha=args.alpha, gamma=args.gamma, tau=args.tau, beta=args.beta,
                  input_dims=env.observation_space.shape[0],
                  env=env, n_actions=env.action_space.shape[0])
    n_episodes = args.n_episodes

    filename = 'inverted_pendulum.png'
    figure_file = os.path.join(os.path.abspath('SAC\\plots'), filename)
    print(figure_file)

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    if args.skip_initial:
        print("Performing initial start steps")

        steps = 0

        while steps < args.start_steps:
            print("Current step: ", steps)
            observation = env.reset()[0]
            done = False

            while not done:
                action = env.action_space.sample()
                observation_, reward, done, _, _ = env.step(action)

                agent.remember(observation, action, reward, observation_, done)

                if not load_checkpoint:
                    agent.learn()

                observation = observation_

                steps += 1

        # updates = 0
        # agent.load_models()
        # states_df = pandas.read_csv(args.states_file)
        # utils.plot(states_df['Previous State'].apply(lambda y: y.strip('[]').split(',')), agent)
    else:
        states_df = pandas.read_csv(args.states_file)
        updates = utils.process_transitions(pandas.read_csv(args.input_file, header=1),
                                            states_df,
                                            agent)
        agent.save_models()
        # utils.save_initial(args.training_file, agent)
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
        plot_learning_curve(x, score_history, figure_file, n_episodes)
