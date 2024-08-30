import cv2
import mediapipe as mp
import threading
import math
from mediapipe.tasks import python
from google.protobuf.json_format import MessageToDict

"""
Unknown
Closed_Fist
Open_Palm
Pointing_Up
Thumb_Down
Thumb_Up
Victory
ILoveYou
"""

class GestureRecognizer:
	def __init__(self):
		try:
			self.num_hands = 2
			self.tracking_confidence = 0.5
			self.detection_confidence = 0.5

			self.hand_gestures_dict = {
				"Left": "None",
				"Right": "None"
			}

			model_path = "app/lib/models/gesture_recognizer.task"
			GestureRecognizer = mp.tasks.vision.GestureRecognizer
			GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
			VisionRunningMode = mp.tasks.vision.RunningMode

			self.lock = threading.Lock()
			options = GestureRecognizerOptions(
				base_options=python.BaseOptions(model_asset_path=model_path),
				running_mode=VisionRunningMode.LIVE_STREAM,
				num_hands=self.num_hands,
				result_callback=self.results_callback)
			self.recognizer = GestureRecognizer.create_from_options(options)

			self.landmark_list = []
			self.bounding_box_list = []

			self.timestamp = 0
			self.mp_drawing = mp.solutions.drawing_utils # type: ignore
			self.mp_hands = mp.solutions.hands # type: ignore
			self.hands = self.mp_hands.Hands(
					static_image_mode=False,
					max_num_hands=self.num_hands,
					min_detection_confidence=self.detection_confidence,
					min_tracking_confidence=self.tracking_confidence,
					model_complexity=0)
			
			self.cap = cv2.VideoCapture(0)

			if not self.cap.isOpened():
				print("Error: Could not open video capture.")
		except Exception as e:
			print(f"Error initializing GestureRecognizer: {e}")

	def find_radius(self, xc: float, yc: float, x2: float, y2: float):
		return int(math.sqrt((x2 - xc) ** 2 + (y2 - yc) ** 2))

	def check_landmark_handedness(self, landmarks: list, label: str):
		return all(self.landmark_list[i][3].lower() == label.lower() for i in range(len(landmarks)))

	def length_between_landmarks(self, img: cv2.typing.MatLike, landmark_1: int, landmark_2: int, hand_label: str = "Right", direct_line: bool = True, length_threshold: int = 120, draw: bool = True, idle_color: tuple = (15, 50, 255), active_color: tuple = (0, 255, 0), circ_radius: int = 5, line_width: int = 2):
		if not self.landmark_list:
			return -1, False

		if not self.check_landmark_handedness([landmark_1, landmark_2], hand_label):
			return -1, False

		x1, y1 = self.landmark_list[landmark_1][1], self.landmark_list[landmark_1][2]
		x2, y2 = self.landmark_list[landmark_2][1], self.landmark_list[landmark_2][2]
		cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

		length = math.hypot(x2 - x1, y2 - y1)

		is_active = length <= length_threshold

		current_color = active_color if is_active else idle_color

		if draw:
			if direct_line:
				cv2.circle(img, (x1, y1), circ_radius, current_color, cv2.FILLED)
				cv2.circle(img, (x2, y2), circ_radius, current_color, cv2.FILLED)
				cv2.circle(img, (cx, cy), circ_radius, current_color, cv2.FILLED)

				cv2.line(img, (x1, y1), (x2, y2), current_color, line_width)
			else:
				rad = self.find_radius(cx, cy, x2, y2)

				cv2.circle(img, (cx, cy), rad, current_color, line_width)
				cv2.circle(img, (cx, cy), circ_radius, current_color, cv2.FILLED)

		return round(length), is_active

	def run(self, frame: cv2.typing.MatLike, draw_gestures: bool = True):
		try:
			self.landmark_list.clear()
			self.bounding_box_list.clear()
			frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			self.results = self.hands.process(frame)

			if self.results.multi_hand_landmarks:
				for hand_landmarks in self.results.multi_hand_landmarks:
					for i in self.results.multi_handedness:
						label = MessageToDict(i)["classification"][0]["label"]

						x_min, y_min = float('inf'), float('inf')
						x_max, y_max = float('-inf'), float('-inf')

						for id, lm in enumerate(hand_landmarks.landmark):
							h, w, c = frame.shape
							cx, cy = int(lm.x * w), int(lm.y * h)

							self.landmark_list.append([id, cx, cy, label])

							border = 16

							x_min = min(x_min, cx - border)
							y_min = min(y_min, cy - border)
							x_max = max(x_max, cx + border)
							y_max = max(y_max, cy + border)

						self.bounding_box_list.append([i, x_min, y_min, x_max, y_max])

					cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2) # type: ignore

				mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
				self.recognizer.recognize_async(mp_image, self.timestamp)
				self.timestamp += 1
			else:
				self.hand_gestures_dict["Left"] = "None"
				self.hand_gestures_dict["Right"] = "None"

			if draw_gestures:
				left = self.hand_gestures_dict["Left"]
				right = self.hand_gestures_dict["Right"]

				cv2.putText(frame, f"Left Hand: {left}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
				cv2.putText(frame, f"Right Hand: {right}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

			return frame, self.hand_gestures_dict, self.landmark_list, self.bounding_box_list
		except Exception as e:
			print(f"Error in run: {e}")

	def results_callback(self, result, output_image, timestamp_ms): # type: ignore
		try:
			with self.lock:
				if result and any(result.gestures):
					for index, hand in enumerate(result.handedness):
						hand_name = hand[0].category_name
						current_hand_gesture = result.gestures[index][0].category_name
						self.hand_gestures_dict[hand_name] = current_hand_gesture
		except Exception as e:
			print(f"Error in results_callback: {e}")

	def cleanup(self):
		try:
			self.cap.release()
			cv2.destroyAllWindows()
		except Exception as e:
			print(f"Error in cleanup: {e}")

if __name__ == "__main__":
	try:
		gesture = GestureRecognizer()

		while True:
			ret, frame = gesture.cap.read()
			if not ret:
				print("Error: Could not read frame.")
				break

			frame, hand_gestures_dict, _, _ = gesture.run(frame) # type: ignore

			cv2.imshow("Gesture Recognizer", frame)

			print(hand_gestures_dict["Left"])

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

		gesture.cleanup()
	except Exception as e:
		print(f"Error in main loop: {e}")
