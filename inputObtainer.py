import numpy as np
import pygame
import csv
import fidgrovePluginUtils as utils


def remove_useless_commands(actions):
    start_index = 0
    end_index = len(actions) - 1

    while start_index < len(actions) and all(val == 0 for val in actions[start_index]):
        start_index += 1

    while end_index >= 0 and all(val == 0 for val in actions[end_index]):
        end_index -= 1

    return actions[start_index:end_index + 1]


def main():
    pygame.init()
    pygame.joystick.init()

    actions = []

    if pygame.joystick.get_count() == 0:
        print("No controllers found.")
        return

    controller = pygame.joystick.Joystick(0)
    controller.init()

    print(f"Controller: {controller.get_name()}\n")

    mode = input("Select mode (a - Append ; w - New): ")

    try:
        while True:
            pygame.event.get()

            joystick_val = controller.get_axis(0) + controller.get_axis(2)

            brake_flag = controller.get_button(8)
            accel_flag = controller.get_button(9)

            action = [-np.clip(joystick_val, -1, 1).round(3), accel_flag - brake_flag]
            actions.append(action)

            print(f"Action: {action}")
            state = utils.obtain_state()

    except KeyboardInterrupt:
        controller.quit()
        pygame.quit()

        actions = remove_useless_commands(actions)

        with open("inputs.csv", mode, newline='') as file:
            csvwriter = csv.writer(file)

            csvwriter.writerows(actions)

            file.close()


if __name__ == "__main__":
    main()
