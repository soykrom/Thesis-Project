import pygame


def main():
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("No controllers found.")
        return

    controller = pygame.joystick.Joystick(0)
    controller.init()

    print(f"Controller: {controller.get_name()}")

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.JOYAXISMOTION:
                    # Handle analog stick input
                    axis = event.axis
                    if axis % 2 != 0:
                        continue

                    value = event.value
                    print(f"Axis {axis}: {value}")
                elif event.type == pygame.JOYBUTTONDOWN:
                    # Handle button press
                    button = event.button
                    print(f"Button {button} pressed")
                elif event.type == pygame.JOYBUTTONUP:
                    # Handle button release
                    button = event.button
                    print(f"Button {button} released")
    except KeyboardInterrupt:
        controller.quit()
        pygame.quit()


if __name__ == "__main__":
    main()
