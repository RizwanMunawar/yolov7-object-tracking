from custom_logic.distance_measurer import DistanceMeasurer
from custom_logic.models.point import Point

import cv2
import copy
import numpy as np


def get_image(video_path: str):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return None

    ret, image = cap.read()

    if not ret:
        print("Error: Could not read the first frame.")
        return None

    return image


class DistanceMeter:
    def __init__(self, video_path: str):
        video_title = video_path.split("/")[-1]
        self.Distance_measurer = DistanceMeasurer(video_title)
        self.points: list[Point] = []
        self.initial_image = get_image(video_path)
        self.work_image = self.get_work_image()

    def calculate_distance(self, p1: Point, p2: Point):
        distance = self.Distance_measurer.get_distance(p1, p2)
        return distance

    def click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            point = Point(x, y)
            self.points.append(point)
            self.display_point(point)

            if len(self.points) == 2:
                distance = self.calculate_distance(self.points[0], self.points[1])
                self.display_text(f"Distance: {distance:.2f}", (10, 50))
                self.display_line(self.points[0], self.points[1])
                self.points = []  # Reset points after calculating distance
            else:
                self.work_image = self.get_work_image()
                self.display_point(point)

    def display_text(self, text, position):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        font_thickness = 3
        color = (0, 0, 255)

        cv2.putText(self.work_image, text, position, font, font_scale, color, font_thickness, cv2.LINE_AA)

    def display_point(self, point: Point):
        cv2.circle(self.work_image, (point.X, point.Y), 5, (0, 0, 184), -1)

    def display_line(self, p1: Point, p2: Point):
        cv2.line(self.work_image, (p1.X, p1.Y), (p2.X, p2.Y), (0, 152, 255), 2)

    def get_work_image(self):
        image = copy.copy(self.initial_image)
        y85 = self.Distance_measurer.get_degree_coordinate(85)
        y90 = self.Distance_measurer.get_degree_coordinate(90)

        image = self.draw_top_rectangle(image, y90, 0.2, 90)
        image = self.draw_top_rectangle(image, y85, 0.2, 85)

        return image

    def draw_top_rectangle(self, image, y: int, opacity: float, degree: float = None):
        width = self.Distance_measurer.Config.Resolution_width

        mask = np.zeros_like(image)
        cv2.rectangle(mask, (0, 0), (width, y), (0, 0, 255), -1)
        result_image = cv2.addWeighted(image, 1, mask, opacity, 0)

        if degree is not None:
            text = f"{str(degree)} degrees"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_size = 1
            font_thickness = 3
            text_color = (0, 0, 0)  # Black color for the text

            # Calculate the position for the text (left bottom corner of the rectangle)
            text_position = (10, y - 20)  # Adjust the y-coordinate for the text position

            cv2.putText(result_image, text, text_position, font, font_size, text_color, font_thickness)
        return result_image

    def start(self):
        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", self.click_event)

        while True:
            # Display the image
            cv2.imshow("Image", self.work_image)

            # Wait for a key press and check if it's the 'Esc' key
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # 'Esc' key
                break

        # Close all OpenCV windows
        cv2.destroyAllWindows()
