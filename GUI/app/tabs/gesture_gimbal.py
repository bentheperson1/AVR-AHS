from __future__ import annotations


from bell.avr.mqtt.payloads import * # type: ignore

from PySide6 import QtGui, QtCore, QtWidgets
import cv2
from enum import Enum
import threading
from app.lib.gesture_module import GestureRecognizer
from app.tabs.base import BaseTabWidget

import numpy as np
import time

import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils # type: ignore
mp_selfie_segmentation = mp.solutions.selfie_segmentation # type: ignore

class GestureGunneryWidget(BaseTabWidget):
	def __init__(self, parent: QtWidgets.QWidget) -> None:
		super().__init__(parent)

		self.setWindowTitle("Gestures")

	def build(self) -> None:
		"""
		Build the GUI layout
		"""
		layout = QtWidgets.QGridLayout(self)
		self.setLayout(layout)

		gesture_groupbox = QtWidgets.QGroupBox("Gesture Control")
		gesture_layout = QtWidgets.QVBoxLayout()
		gesture_groupbox.setLayout(gesture_layout)

		self.camera_activate_button = QtWidgets.QPushButton("Turn On/Off Camera")
		self.camera_activate_button.clicked.connect(self.camera_on_off)

		self.image_frame = QtWidgets.QLabel()
		gesture_layout.addWidget(self.camera_activate_button)
		gesture_layout.addWidget(self.image_frame)

		layout.addWidget(gesture_groupbox, 0, 0, 3, 1)

		cv2.setUseOptimized(True)

		self.gesture = GestureRecognizer()

		self.ControlMode = Enum("ControlMode", ["INACTIVE", "SERVO", "GIMBAL"])

		self.ControlNames = {
			self.ControlMode.INACTIVE: "Deactivated",
			self.ControlMode.SERVO: "Individual Servo Control",
			self.ControlMode.GIMBAL: "Gimbal Joystick"
		}

		self.gimbal_pos_set = False

		self.deadzone = 48
		self.mode = self.ControlMode.SERVO
		self.laser_on = False
		self.laser_activated_once = True
		self.has_activated = False

		self.max_servos = 4

		self.window_width = 1280
		self.window_height = 720

		self.cap = cv2.VideoCapture(0)
		self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')) # type: ignore
		self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.window_width)
		self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.window_height)

		self.yaw_port = 2
		self.pitch_port = 3

		self.servo_ports = {
			2: "Conex",
			1: "Water Drop 1",
			0: "Water Drop 2",
			5: "Vechicle Drop"
		}

		self.servos_active = [False, False, False, False]
		self.servo_activated = [False, False, False, False]

		self.touch_active_thres = 160

		self.depth_vary = 6000
		self.current_palm_area = -1
		self.hold_palm_area = -1

		self.x_axis = 700
		self.y_axis = 700

		self.move_bounds = [700, 2200]

		self.camera_turned_on = False

		self.frame_thread = threading.Thread(target=self.update_frame,daemon=True)
		self.frame_thread.start()
		self.frame_updated.connect(self.update_frame_display)

	def camera_on_off(self):
		self.camera_turned_on = not self.camera_turned_on

		detail_str = "CAMERA ON" if self.camera_turned_on else "CAMERA OFF"
		print(detail_str)

	def is_point_inside_radius(self, point: tuple, center: tuple, radius: int) -> bool:
		distance = np.sqrt((point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2)

		return distance <= radius

	def clip(self, value: float, lower: float, upper: float) -> int:
		return int(lower if value < lower else upper if value > upper else value)

	def update_frame(self):
		BG_COLOR = (40, 40, 40)

		with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
			bg_image = None

			while True:
				if self.camera_turned_on:
					ret, image = self.cap.read()
					if not ret:
						print("Error: Could not read frame.")
						break

					image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
	
					image.flags.writeable = False
					results = selfie_segmentation.process(image)

					image.flags.writeable = True
					image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

					condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.15

					if bg_image is None:
						bg_image = np.zeros(image.shape, dtype=np.uint8)
						bg_image[:] = BG_COLOR
					frame = np.where(condition, image, bg_image)

					frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
					h, w, ch = frame.shape

					frame, hand_gesture, found_landmarks, bounding_boxes = self.gesture.run(frame, False) # type: ignore

					left_hand = hand_gesture["Right"]
					right_hand = hand_gesture["Left"]

					if left_hand == "Thumb_Up":
						self.mode = self.ControlMode.SERVO
					# elif left_hand == "Victory":
					# 	self.mode = self.ControlMode.GIMBAL
					elif left_hand == "Pointing_Up":
						self.mode = self.ControlMode.INACTIVE

					if self.mode == self.ControlMode.INACTIVE:
						if self.gimbal_pos_set:
							# self.send_message(
							# 	"avr/pcm/servo/absolute",
							# 	AVRPCMServoAbsolute(servo=self.yaw_port, position=int(self.move_bounds[0])),
							# )
							# self.send_message(
							# 	"avr/pcm/servo/absolute",
							# 	AVRPCMServoAbsolute(servo=self.pitch_port, position=int(self.move_bounds[0])),
							# )
							pass

						self.gimbal_pos_set = False

					elif self.mode == self.ControlMode.SERVO:
						self.gimbal_pos_set = False

						for i in range(self.max_servos):
							is_active = False

							if found_landmarks and self.gesture.check_landmark_handedness(found_landmarks, "Right") and bounding_boxes:
								_, landmarks_touching = self.gesture.length_between_landmarks(frame, 0, (i * 4) + 8, length_threshold=self.touch_active_thres)

								if landmarks_touching:
									if not self.servo_activated[i]:
										self.servos_active[i] = not self.servos_active[i]
										self.servo_activated[i] = True
										is_active = True
								else:
									self.servo_activated[i] = False

								active_text = "ON" if self.servos_active[i] else "OFF"
								servo_index = list(self.servo_ports.keys())[i]
								servo_text = list(self.servo_ports.values())[i]
								
								text_pos = (bounding_boxes[0][1], bounding_boxes[0][2] - (i * 32) - 32)

								cv2.putText(frame, f"{servo_text}: {active_text}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 8, cv2.LINE_AA, False)
								
								cv2.putText(frame, f"{servo_text}: {active_text}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA, False)

								if is_active:
									action_text = "open" if self.servos_active[i] else "close"
									self.send_message("avr/pcm/set_servo_open_close", AvrPcmSetServoOpenClosePayload(servo=servo_index, action=action_text))

					elif self.mode == self.ControlMode.GIMBAL:
						landmark_1, landmark_2, landmark_3 = 8, 5, 17

						if self.gesture.landmark_list and self.gesture.check_landmark_handedness([landmark_1], "Right"):
							x1, y1 = self.gesture.landmark_list[landmark_1][1], self.gesture.landmark_list[landmark_1][2]

							cv2.circle(frame, (x1, y1), 8, (255, 0, 0), cv2.FILLED)

							if not self.gimbal_pos_set and x1 > self.window_width // 2:
								self.gimbal_pos_set = True

								center_x = x1
								center_y = y1

							cv2.circle(frame, (center_x, center_y), self.deadzone, (255, 0, 255), 4)

							cv2.line(frame, (center_x, center_y), (x1, y1), (255, 0, 0), 4)
							cv2.line(frame, (center_x, center_y), (x1, center_y), (255, 255, 0), 4)
							cv2.line(frame, (x1, center_y), (x1, y1), (0, 255, 0), 4)

							x_comp = x1 - center_x
							y_comp = y1 - center_y

							if not self.is_point_inside_radius((x1, y1), (center_x, center_y), self.deadzone):
								yaw_pct = round((x_comp / center_x) * 100)
								pitch_pct = round((y_comp / center_y) * 100)

								self.x_axis += yaw_pct
								self.y_axis += pitch_pct

								self.x_axis = self.clip(self.x_axis, self.move_bounds[0], self.move_bounds[1])
								self.y_axis = self.clip(self.y_axis, self.move_bounds[0], self.move_bounds[1])

								# self.send_message(
								# 	"avr/pcm/servo/absolute",
								# 	AVRPCMServoAbsolute(servo=self.yaw_port, position=int(self.x_axis)),
								# )
								# self.send_message(
								# 	"avr/pcm/servo/absolute",
								# 	AVRPCMServoAbsolute(servo=self.pitch_port, position=int(self.y_axis)),
								# )

							_, laser_active = self.gesture.length_between_landmarks(frame, 4, 5, "Right", True, 30)
							active_text = "ON" if laser_active else "OFF"

							if found_landmarks and self.gesture.check_landmark_handedness(found_landmarks, "Right"):
								cv2.putText(frame, f"Laser: {active_text}", (found_landmarks[4][1], found_landmarks[4][2] + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 8, cv2.LINE_AA)
								cv2.putText(frame, f"Laser: {active_text}", (found_landmarks[4][1], found_landmarks[4][2] + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

							if laser_active:
								if not self.has_activated:
									self.has_activated = True
									self.laser_on = not self.laser_on
									self.laser_activated_once = False
							else:
								if self.has_activated:
									self.has_activated = False
									self.laser_activated_once = True

							if self.laser_activated_once:
								self.laser_activated_once = False

								if self.laser_on:
									self.send_message("avr/pcm/laser/on")
								else:
									self.send_message("avr/pcm/laser/off")

					cv2.putText(frame, f"Mode: {self.ControlNames[self.mode]}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

					bytes_per_line = ch * w

					q_image = QtGui.QImage(frame, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888) # type: ignore
					pixmap = QtGui.QPixmap.fromImage(q_image)
					self.image_frame.setPixmap(pixmap)
					self.image_frame.setScaledContents(False)

					self.frame_updated.emit(pixmap)
				else:
					self.image_frame.clear()
					self.frame_updated.emit(QtGui.QPixmap())

				time.sleep(0.002)

	frame_updated = QtCore.Signal(QtGui.QPixmap)

	def update_frame_display(self, pixmap: QtGui.QPixmap):
		self.image_frame.setPixmap(pixmap)
		self.image_frame.setScaledContents(False)

	def closeEvent(self, event: QtGui.QCloseEvent):
		self.cap.release()
		event.accept()
