import sys
from PyQt5.uic import loadUi
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import pyqtSlot, QTimer, QDate
from PyQt5.QtWidgets import QApplication, QDialog
import resource
import cv2
import face_recognition as fr
import numpy as np
import datetime
import pickle
import pandas as pd
from datetime import date


class Ui_OutputDialog(QDialog):
    def __init__(self):
        super(Ui_OutputDialog, self).__init__()
        self.encode_list = None
        self.class_names = None
        self.timer = None
        self.capture = None
        loadUi("./outputwindow.ui", self)

        # Update Time and Date
        now = QDate.currentDate()
        current_date = now.toString("ddd dd MMMM")
        current_time = datetime.datetime.now().strftime("%I:%M %p")

        self.Date_Label.setText(current_date)
        self.Time_Label.setText(current_time)
        self.image = None
        self.attendance = []
        self.marked = []

    @pyqtSlot()
    def start_video(self, camera_name):
        """
        :param camera_name: link of camera or usb camera
        :return:
        """
        if len(camera_name) == 1:
            self.capture = cv2.VideoCapture(int(camera_name))
        else:
            self.capture = cv2.VideoCapture(camera_name)
        self.timer = QTimer(self)  # Create Timer

        # known face encoding and known face name list
        # Import Saved_Encodings
        with open('Year_11.dat', 'rb') as f:
            data = pickle.load(f)
        self.class_names = list(data.keys())
        self.encode_list = np.array(list(data.values()))

        self.timer.timeout.connect(self.update_frame)  # Connect timeout to the output function
        self.timer.start(10)  # emit the timeout() signal at x=40ms

    def face_rec_(self, frame, encode_list_known, class_names):
        """
        :param frame: frame from camera
        :param encode_list_known: known face encoding
        :param class_names: known face names
        :return:
        """
        present_time = datetime.time(23, 50, 0)
        late_time = datetime.time(23, 59, 0)

        def mark_attendance(student, opt):
            if opt == 1:
                if cur_time <= present_time:
                    # Present
                    self.attendance.append({"Name": student, "Time": cur_time.strftime("%X"), "Status": "Present"})
                elif present_time < cur_time <= late_time:
                    # Late
                    self.attendance.append({"Name": student, "Time": cur_time.strftime("%X"), "Status": "Late"})
            elif opt == 0:
                # Absent
                self.attendance.append({"Name": student, "Time": None, "Status": "Absent"})
            else:
                pass
            self.marked.append(student)

        def show_face(student, opt):
            if opt == 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, "UNKNOWN", (x1 + 6, y2 - 6), (cv2.FONT_HERSHEY_SIMPLEX), 1, (255, 255, 255), 2)
            elif opt == 1:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (255, 255, 0), cv2.FILLED)
                cv2.putText(frame, student, (x1 + 6, y2 - 6), (cv2.FONT_HERSHEY_SIMPLEX), 1, (255, 255, 255), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, student, (x1 + 6, y2 - 6), (cv2.FONT_HERSHEY_SIMPLEX), 1, (0, 0, 0), 2)

        # face recognition
        frameS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
        faces_cur_frame = fr.face_locations(frameS)
        encodings_cur_frame = fr.face_encodings(frameS, faces_cur_frame)

        cur_time = datetime.datetime.now().time()

        if cur_time <= late_time:
            # Comparing Faces Located with Faces Already Recognized
            for encodeFace, faceLoc in zip(encodings_cur_frame, faces_cur_frame):
                matches_cv = fr.compare_faces(encode_list_known, encodeFace, 0.4)
                matches_attendance = fr.compare_faces(encode_list_known, encodeFace, 0.35)
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                if True in matches_attendance:
                    face_distance = fr.face_distance(encode_list_known, encodeFace)
                    match_index = np.argmin(face_distance)
                    name = class_names[match_index].upper()
                    # Attendance
                    if name not in self.marked:
                        mark_attendance(name, 1)
                        show_face(name, 1)
                        print(self.marked)
                    else:
                        show_face("MARKED", 2)
                        self.MarkedNames.setText(name)
                elif True in matches_cv:
                    face_distance = fr.face_distance(self.encode_list, encodeFace)
                    match_index = np.argmin(face_distance)
                    name = self.class_names[match_index].upper()
                    if name not in self.marked:
                        show_face(name, 1)
                    else:
                        show_face("MARKED", 2)
                        self.MarkedNames.setText(name)
                else:
                    show_face("UNKNOWN", 0)
        elif cur_time > late_time:
            for late in class_names:
                if late not in self.marked:
                    mark_attendance(late, 0)
                else:
                    pass
        output = pd.DataFrame(self.attendance)
        output.sort_values("Name")
        output.to_excel(f"{date.today()}.xlsx")
        return frame

    def update_frame(self):
        ret, self.image = self.capture.read()
        self.display_image(self.image, self.encode_list, self.class_names, 1)

    def display_image(self, image, encode_list, class_names, window=1):
        """
        :param image: frame from camera
        :param encode_list: known face encoding list
        :param class_names: known face names
        :param window: number of window
        :return:
        """
        image = cv2.resize(image, (1920, 1080))
        try:
            image = self.face_rec_(image, encode_list, class_names)
        except Exception as e:
            print(e)
        qformat = QImage.Format_Indexed8
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        out_image = QImage(image, image.shape[1], image.shape[0], image.strides[0], qformat)
        out_image = out_image.rgbSwapped()

        if window == 1:
            self.imgLabel.setPixmap(QPixmap.fromImage(out_image))
            self.imgLabel.setScaledContents(True)


class Ui_Dialog(QDialog):
    def __init__(self):
        super(Ui_Dialog, self).__init__()
        loadUi("./mainwindow.ui", self)

        self.runButton.clicked.connect(self.run_slot)

        self._new_window = None
        self.Videocapture_ = None

    def refresh_all(self):
        """
        Set the text of lineEdit once it's valid
        """
        self.Videocapture_ = "0"

    @pyqtSlot()
    def run_slot(self):
        """
        Called when the user presses the Run button
        """
        print("Clicked Run")
        self.refresh_all()
        print(self.Videocapture_)
        ui.hide()  # hide the main window
        self.output_window_()  # Create and open new output window

    def output_window_(self):
        """
        Created new window for vidual output of the video in GUI
        """
        self._new_window = Ui_OutputDialog()
        self._new_window.show()
        self._new_window.start_video(self.Videocapture_)
        print("Video Played")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = Ui_Dialog()
    ui.show()
    sys.exit(app.exec_())