import sys
import time
import pygame
from sphero_sdk import SpheroRvrObserver
from sphero_sdk import DriveFlagsBitmask
import pigpio

pwm = pigpio.pi()
rvr = SpheroRvrObserver()

SERVO_PINS = [17, 22]

for pin in SERVO_PINS:
    pwm.set_mode(pin, pigpio.OUTPUT)
    pwm.set_PWM_frequency(pin, 50)

pygame.init()
pygame.joystick.init()

DEADZONE = 0.2
MAX_SPEED = 255
DRIVE_SPEED = 192
TURN_SPEED = 20

direction = 0
joystick_connected = False
last_servo_time = 0
SERVO_DELAY = 1

while not joystick_connected:
    pygame.joystick.quit()
    pygame.joystick.init()

    if pygame.joystick.get_count() > 0:
        joystick_connected = True
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print(f"Joystick connected: {joystick.get_name()}")
    else:
        print("No joystick detected, checking again in 1 second...")
        time.sleep(1)

try:
    rvr.wake()
    time.sleep(2)
    print("RVR has woken up!")

    rvr.drive_control.reset_heading()

    while True:
        try:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt

            try:
                left_stick_y = -joystick.get_axis(1)
                right_stick_y = joystick.get_axis(2)
            except pygame.error as e:
                print(f"Joystick read error: {e}")
                left_stick_y = 0
                right_stick_y = 0

            print(f"Left Stick Y: {left_stick_y}, Right Stick Y: {right_stick_y}")

            if abs(left_stick_y) < DEADZONE:
                left_stick_y = 0
            if abs(right_stick_y) < DEADZONE:
                right_stick_y = 0

            speed_mod = 1
            left_speed = int(left_stick_y * DRIVE_SPEED * speed_mod)
            right_speed = int(right_stick_y * TURN_SPEED * speed_mod)

            direction += right_speed
            if direction > 360:
                direction = 0

            print(f"Left Speed: {left_speed}, Right Speed: {right_speed}, Direction: {direction}")

            current_time = time.time()

            if joystick.get_button(0) and (current_time - last_servo_time) > SERVO_DELAY:
                print("Button A pressed: Moving Servos to 0 degrees")
                
                pwm.set_servo_pulsewidth(SERVO_PINS[0], 500)
                pwm.set_servo_pulsewidth(SERVO_PINS[1], 500)
                
                last_servo_time = current_time

            if joystick.get_button(1) and (current_time - last_servo_time) > SERVO_DELAY:
                print("Button B pressed: Moving Servos to 90 degrees")
                
                pwm.set_servo_pulsewidth(SERVO_PINS[0], 1500)
                pwm.set_servo_pulsewidth(SERVO_PINS[1], 1500)
                
                last_servo_time = current_time

            rvr.drive_control.roll_start(
                speed=left_speed,
                heading=direction
            )

            time.sleep(0.1)

        except Exception as e:
            print(f"Error in main loop: {e}")
            time.sleep(0.1)

except KeyboardInterrupt:
    print("Exiting program due to KeyboardInterrupt...")

except Exception as e:
    print(f"Critical error: {e}")

finally:
    try:
        rvr.close()
    except Exception as e:
        print(f"Error while closing RVR connection: {e}")

    for pin in SERVO_PINS:
        pwm.set_PWM_dutycycle(pin, 0)
        pwm.set_PWM_frequency(pin, 0)

    pygame.quit()
    sys.exit()
