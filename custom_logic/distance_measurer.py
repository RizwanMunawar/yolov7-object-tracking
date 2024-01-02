from custom_logic.models.speed_estimation_config import SpeedEstimationConfig
from custom_logic.models.point import Point

import custom_logic.repositories.video_repository as video_repository
import custom_logic.repositories.camera_repository as camera_repository
import math


def get_distance_measuring_config(video_title: str):
    video = video_repository.get_by_title(video_title)
    if video is None:
        return None

    camera = camera_repository.get(video.Camera_id)
    if camera is None:
        return None

    return SpeedEstimationConfig(video.Camera_height, video.Tilt_angle, camera.Focal_length,
                                 camera.Horizontal_viewing_angle,
                                 camera.Vertical_viewing_angle, camera.Resolution_width,
                                 camera.Resolution_height, camera.Sensor_size)


class DistanceMeasurer:
    def __init__(self, video_title: str):
        self.Config = get_distance_measuring_config(video_title)

    def get_distance(self, p1: Point, p2: Point) -> float:
        # dx and dy in millimeters
        dx = self.get_trapeze_big_base(p1, p2)
        dy = self.get_dy(p1, p2)

        return math.sqrt(dx ** 2 + dy ** 2) / 1000

    def get_trapeze_big_base(self, p1: Point, p2: Point):
        # physical pixel size
        k = self.get_pixel_size()

        du = abs(p1.X - p2.X)
        high_object, _ = self.get_high_and_low_objects(p1, p2)

        # distance between high object and center of the image vertically
        delta_high_object = int(abs(high_object.Y - self.Config.Resolution_height / 2))

        # Theta in radians
        theta = math.radians(self.Config.Theta)

        # deviation from the center of the image vertically in radians
        theta_i = math.atan(delta_high_object * k/ self.Config.f)

        # Angle from high object ot vertical
        if high_object.Y < self.Config.Resolution_height / 2:
            theta_full = theta + math.atan(theta_i)
        else:
            theta_full = theta - math.atan(theta_i)

        # In millimeters
        trapeze_base = (self.Config.H * du * k * math.cos(theta_i)) / (self.Config.f * math.cos(theta_full))

        return trapeze_base

    def get_dy(self, p1: Point, p2: Point):
        high_object: Point
        low_object: Point

        high_object, low_object = self.get_high_and_low_objects(p1, p2)

        image_center = self.Config.Resolution_height / 2
        k = self.get_pixel_size()

        theta_radians = math.radians(self.Config.Theta)
        theta_high = math.atan((abs(high_object.Y - image_center) * k) / self.Config.f)
        theta_low = math.atan((abs(low_object.Y - image_center) * k) / self.Config.f)

        if low_object.Y <= image_center:
            return self.Config.H * (math.tan(theta_radians + theta_high) -
                                    math.tan(theta_radians + theta_low))
        elif high_object.Y > image_center:
            return self.Config.H * (math.tan(theta_radians - theta_high) -
                                    math.tan(theta_radians - theta_low))
        else:
            return self.Config.H * (math.tan(theta_radians + theta_high) -
                                    math.tan(theta_radians - theta_low))

    def get_high_and_low_objects(self, p1: Point, p2: Point):
        if p1.Y < p2.Y:
            high_object = p1
            low_object = p2
        else:
            high_object = p2
            low_object = p1

        return high_object, low_object

    def get_pixel_size(self):
        return self.Config.Sensor_size / (
            math.sqrt(self.Config.Resolution_width ** 2 + self.Config.Resolution_height ** 2))

    def get_degree_coordinate(self, degree: int):
        pixel_size = self.get_pixel_size()
        if self.Config.Theta + self.Config.Vertical_viewing_angle / 2 < degree:
            return self.Config.Resolution_height
        elif self.Config.Theta - self.Config.Vertical_viewing_angle / 2 > degree:
            return 0

        if degree > self.Config.Theta:
            return int(self.Config.Resolution_height / 2 - self.Config.f * math.tan(
                math.radians(degree - self.Config.Theta)) / pixel_size)
        else:
            return int(self.Config.Resolution_height / 2 + self.Config.f * math.tan(
                math.radians(self.Config.Theta - degree)) / pixel_size)
