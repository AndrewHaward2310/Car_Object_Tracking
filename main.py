import sys
from PyQt5 import QtGui
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
from ui import Ui_MainWindow
import cv2
import numpy as np
import torch
import requests
from deep_sort_realtime.deepsort_tracker import DeepSort
from yolov9.models.common import DetectMultiBackend, AutoShape

# Check for CUDA
print("Number of GPUs available: ", torch.cuda.device_count())
print("GPU name: ", torch.cuda.get_device_name(0))

conf_threshold = 0.5
tracking_class = None  # Change this to the class ID of the bottle

# Initialize DeepSORT
tracker = DeepSort(max_age=5, n_init=3, nms_max_overlap=1.0, nn_budget=100)

# Initialize YOLOv9
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DetectMultiBackend(weights="yolov9/weights/yolov9-m-converted.pt", device=device, fuse=True)
model = AutoShape(model)

# Load class names
with open("yolov9/data_ext/classes.names") as f:
    class_names = f.read().strip().split('\n')

colors = np.random.randint(0, 255, size=(len(class_names), 3))

# HTTP URLs
ip = "192.168.1.1"
led_val = 0
car_id = 0  # 0: stop, 1: up, 2: down, 3: left, 4: right
httpCameraUrl = f"http://{ip}:81/stream"

def send_data(ip, key, value):
    urls = {
        "MoveCar": f"http://{ip}/move_car?move={value}",
        "Speed": f"http://{ip}/control?var=speed&val={value}",
        "Light": f"http://{ip}/control?var=led_intensity&val={value}",
        "ServoX": f"http://{ip}/control?var=servo_x&val={value}",
        "ServoY": f"http://{ip}/control?var=servo_y&val={value}"
    }
    try:
        requests.get(urls[key])
    except requests.RequestException as e:
        print(f"Failed to send {key} command: {e}")

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.uic = Ui_MainWindow()
        self.uic.setupUi(self)

        self.uic.pushButton_2.clicked.connect(self.start_camera)
        self.uic.engine_forward.clicked.connect(lambda: send_data(ip, "MoveCar", "1"))
        self.uic.engine_backward.clicked.connect(lambda: send_data(ip, "MoveCar", "2"))
        self.uic.engine_turn_left.clicked.connect(lambda: send_data(ip, "MoveCar", "3"))
        self.uic.engine_turn_right.clicked.connect(lambda: send_data(ip, "MoveCar", "4"))
        self.uic.track_btn.clicked.connect(self.handle_track_button_click)
        self.uic.horizontalSlider.valueChanged.connect(self.update_led_intensity)

        self.camera_thread = CaptureVideo(self)
        self.camera_thread.signal.connect(self.show_webcam)

    def update_led_intensity(self, value):
        send_data(ip, "Light", value)

    def start_camera(self):
        if not self.camera_thread.isRunning():
            self.camera_thread.start()

    def show_webcam(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.uic.label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(800, 600, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def detect_and_track_objects(self, frame):
        result = model(frame)
        detect = []

        for detect_object in result.pred[0]:
            label, confidence, bbox = detect_object[5], detect_object[4], detect_object[:4]
            x1, y1, x2, y2 = map(int, bbox)
            class_id = int(label)

            if tracking_class is None:
                if confidence < conf_threshold:
                    continue
            else:
                if class_id != tracking_class or confidence < conf_threshold:
                    continue

            detect.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])

        tracks = tracker.update_tracks(detect, frame=frame)

        for track in tracks:
            if track.is_confirmed():
                track_id = track.track_id
                ltrb = track.to_ltrb()
                class_id = track.get_det_class()
                x1, y1, x2, y2 = map(int, ltrb)
                color = colors[class_id]
                B, G, R = map(int, color)

                label = "{}-{}".format(class_names[class_id], track_id)

                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
                frame = cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(label) * 12, y1), (B, G, R), -1)
                frame = cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                if class_id == 39:
                    self.control_car_to_center_bottle(frame, x1, y1, x2, y2)

        return frame

    def handle_track_button_click(self):
        print("Track button clicked. Implement behavior if needed.")

    def control_car_to_center_bottle(self, frame, x1, y1, x2, y2):
        frame_height, frame_width, _ = frame.shape
        bottle_center_x = (x1 + x2) // 2
        bottle_center_y = (y1 + y2) // 2

        center_x = frame_width // 2
        center_y = frame_height // 2

        threshold = 10

        print(f"Bottle Center: ({bottle_center_x}, {bottle_center_y})")
        print(f"Frame Center: ({center_x}, {center_y})")

        if bottle_center_y < center_y - threshold:
            print("Moving forward")
            send_data(ip, "MoveCar", "1")  # Move forward
            QThread.msleep(50)  # Delay after movement
        elif bottle_center_y > center_y + threshold:
            print("Moving backward")
            send_data(ip, "MoveCar", "2")  # Move backward
            QThread.msleep(50)  # Delay after movement
        elif bottle_center_x < center_x - threshold:
            print("Turning left")
            send_data(ip, "MoveCar", "3")  # Turn left
            QThread.msleep(50)  # Delay after movement
        elif bottle_center_x > center_x + threshold:
            print("Turning right")
            send_data(ip, "MoveCar", "4")  # Turn right
            QThread.msleep(50)  # Delay after movement
        else:
            print("Bottle is centered")


class CaptureVideo(QThread):
    signal = pyqtSignal(np.ndarray)

    def __init__(self, parent):
        super(CaptureVideo, self).__init__(parent)
        self.parent = parent
        self.cap = cv2.VideoCapture(httpCameraUrl)

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                frame = self.parent.detect_and_track_objects(frame)
                self.signal.emit(frame)
            else:
                print("Failed to capture frame")
                break

    def __del__(self):
        self.cap.release()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())