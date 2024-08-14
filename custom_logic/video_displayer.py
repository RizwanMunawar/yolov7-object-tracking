import cv2
import random

from typing import List

from custom_logic.models.point import Point
from custom_logic.models.tracking_object import TrackingObject
from custom_logic.services.tracking_service import get_tracking_objects_for_video, group_tracking_objects
from custom_logic.helpers.helper import get_video_info
from datetime import datetime


OUTPUT_PATH = "./video_output"


def get_formatted_speed(speed: float):
    return round(speed, 2)


def save_video(frames, video_path):
    _, fps, _, width, height = get_video_info(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec (H264 for .mp4 format)

    datetime_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    name = video_path.split("/")[-1]
    out = cv2.VideoWriter(f'{OUTPUT_PATH}/{datetime_string}-{name}', fourcc, fps,
                          (width, height))  # Filename, codec, fps, frame size

    for frame in frames:
        out.write(frame)


class VideoDisplayer:
    def __init__(self, video_path: str, tracking_run_id: int, is_save_video: bool, is_draw_track: bool):
        self.video_path = video_path
        self.tracking_run_id = tracking_run_id
        self.is_save_video = is_save_video
        self.is_draw_track = is_draw_track
        self.tracks = dict()
        self.colors = dict()

    objects_speed_dictionary = dict()

    def display_video(self):
        capture = cv2.VideoCapture(self.video_path)

        _, fps, _, _, _ = get_video_info(self.video_path)

        tracking_objects = get_tracking_objects_for_video(self.video_path, self.tracking_run_id)
        tracking_objects_by_frame = group_tracking_objects(tracking_objects)
        frame = 1

        frame_images = list()

        while capture.isOpened():
            _, img_1 = capture.read()
            if img_1 is None:
                capture.release()
                continue

            frame_tracking_objects = tracking_objects_by_frame.get(frame)
            self.drawFrameBoxes(img_1, frame_tracking_objects if frame_tracking_objects is not None else [])
            frame += 1

            if self.is_draw_track:
                self.drawTracks(img_1, frame_tracking_objects if frame_tracking_objects is not None else [])

            cv2.imshow("Detecting Motion...", img_1)
            if self.is_save_video:
                frame_images.append(img_1)
            if cv2.waitKey(int(1000 / fps)) == 13:
                exit()

        if self.is_save_video:
            save_video(frame_images, self.video_path)

    def drawFrameBoxes(self, img, trackingObjects: List[TrackingObject]):
        if any(map(lambda obj: obj.speed is not None, trackingObjects)):
            self.objects_speed_dictionary = dict()
        for item in trackingObjects:
            self.drawBox(img, item)

    def drawTracks(self, img, trackingObjects: List[TrackingObject]):
        line_thickness = 2

        for tracking_object in trackingObjects:
            if tracking_object.tracking_object_id in self.tracks:
                self.tracks[tracking_object.tracking_object_id].append(Point(tracking_object.center_x, tracking_object.bottom_y))
            else:
                self.tracks[tracking_object.tracking_object_id] = [Point(tracking_object.center_x, tracking_object.bottom_y)]
                self.colors[tracking_object.tracking_object_id] = self.generate_random_color(tracking_object.tracking_object_id)

        for key, points in self.tracks.items():
            for i in range(len(points) - 1):
                pt1 = (points[i].X, points[i].Y)
                pt2 = (points[i+1].X, points[i+1].Y)
                cv2.line(img, pt1, pt2, self.colors[key], line_thickness)

    @staticmethod
    def generate_random_color(n):
        colors = [
            (255, 0, 0),
            (0, 0, 255),
            (0, 255, 0)
        ]

        return colors[n % len(colors)]

    def drawBox(self, img, tracking_object: TrackingObject):
        x1 = int(tracking_object.center_x - tracking_object.box_width / 2)
        x2 = int(tracking_object.center_x + tracking_object.box_width / 2)
        y1 = int(tracking_object.center_y - tracking_object.box_height / 2)
        y2 = int(tracking_object.center_y + tracking_object.box_height / 2)
        label = str(tracking_object.tracking_object_id)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, [0, 0, 255], 2)

        self.draw_speed(img, tracking_object, x1, y1)

    def draw_speed(self, img, tracking_object: TrackingObject, left_x: int, top_y: int):
        speed: int = None

        if tracking_object.speed is not None:
            speed = tracking_object.speed
            self.objects_speed_dictionary[tracking_object.tracking_object_id] = speed
        elif self.objects_speed_dictionary.get(tracking_object.tracking_object_id) is not None:
            speed = self.objects_speed_dictionary.get(tracking_object.tracking_object_id)

        if speed is not None:
            cv2.putText(img, f"{get_formatted_speed(speed)} m/s", (left_x, top_y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        1, [0, 0, 255], 2)
