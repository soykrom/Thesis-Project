import numpy as np
import pygame
import csv


def main():
    pygame.init()
    pygame.joystick.init()

    accel_flag = 0
    brake_flag = 0
    joystick_val = 0.0
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
            for event in pygame.event.get():
                if event.axis % 2 != 0 == pygame.JOYAXISMOTION:
                    # Handle analog stick input
                    axis = event.axis
                    if axis % 2 != 0:
                        continue

                    joystick_val = event.value
                elif event.type == pygame.JOYBUTTONDOWN:
                    # Handle button press
                    button = event.button
                    if button == 8:  # L2
                        brake_flag = 1
                    elif button == 9:  # R2
                        accel_flag = 1
                elif event.type == pygame.JOYBUTTONUP:
                    # Handle button release
                    button = event.button
                    if button == 8:
                        brake_flag = 0
                    elif button == 9:
                        accel_flag = 0

                action = [accel_flag - brake_flag, np.clip(joystick_val, -1, 1).round(3)]

                actions.append(action)
                print(f"Action: {action}")

    except KeyboardInterrupt:
        controller.quit()
        pygame.quit()

        with open("inputs.csv", mode, newline='') as file:
            csvwriter = csv.writer(file)

            csvwriter.writerows(actions)

            file.close()


if __name__ == "__main__":
    main()
