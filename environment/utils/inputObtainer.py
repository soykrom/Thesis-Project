import argparse
import os.path
import time

import numpy as np
import pandas
import pygame

import fidgrovePluginUtils as utils


def remove_useless_commands(actions, states):
    start_index = 0
    end_index = len(actions) - 1

    while start_index < len(actions) and all(val == 0 for val in actions[start_index]):
        start_index += 1

    while end_index >= 0 and all(val == 0 for val in actions[end_index]):
        end_index -= 1

    return actions[start_index:end_index + 1], states[start_index:end_index + 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='w')

    pygame.init()
    pygame.joystick.init()

    actions = []
    state_transitions = []

    if pygame.joystick.get_count() == 0:
        print("No controllers found.")
        return

    controller = pygame.joystick.Joystick(0)
    controller.init()

    print(f"Controller: {controller.get_name()}\n")

    try:
        prev_state = utils.obtain_state()

        while True:
            pygame.event.get()

            joystick_val = controller.get_axis(0) + controller.get_axis(2)

            brake_flag = controller.get_button(8)
            accel_flag = controller.get_button(9)

            action = [-np.clip(joystick_val, -1, 1).round(3), accel_flag - brake_flag]
            actions.append(action)
            print(f"Action: {action}")

            new_state = utils.obtain_state()
            state_transitions.append([prev_state, new_state])

            prev_state = new_state

            utils.reset_events()

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("Finishing...")

        controller.quit()
        pygame.quit()

        actions, state_transitions = remove_useless_commands(actions, state_transitions)

        print(len(actions))
        print(len(state_transitions))

        mode = input("Select mode ('a' - Append or 'w' - Overwrite): ")
        actions_df = pandas.DataFrame(actions, columns=['Steering', 'Throttle'])
        actions_df.to_csv(os.path.abspath('common/inputs.csv'), mode=mode, index=False)

        states_df = pandas.DataFrame(state_transitions, columns=['Previous State', 'New State'])
        states_df.to_csv(os.path.abspath('common/transitions.csv'), mode=mode, index=False)


if __name__ == "__main__":
    main()
