from custom_logic.models.speed_estimation_config import SpeedEstimationConfig
from custom_logic.models.point import Point

import custom_logic.repositories.video_repository as video_repository
import custom_logic.repositories.camera_repository as camera_repository
import math

MAX_DEGREE_OF_MEASURING = 88


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
        # trapeze big base
        a = self.get_dx_by_high_point(p1, p2)
        # trapeze height
        h = self.get_dy(p1, p2)
        if h == 0:
            return a / 1000

        # angle near small base where diagonal starts
        alpha = self.get_trapeze_angle(p1, p2)

        return h / (math.sin(math.atan(h / (a + h / math.tan(alpha))))) / 1000

    # to get trapeze big base
    def get_dx_by_high_point(self, p1: Point, p2: Point):
        # physical pixel size
        k = self.get_pixel_size()

        du = abs(p1.X - p2.X)
        high_object, _ = self.get_high_and_low_points(p1, p2)

        # distance between high object and center of the image vertically
        delta_high_object = int(abs(high_object.Y - self.Config.Resolution_height / 2))

        # Theta in radians
        theta = math.radians(self.Config.Theta)

        # deviation from the center of the image vertically in radians
        theta_i = math.atan(delta_high_object * k / self.Config.f)

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

        high_object, low_object = self.get_high_and_low_points(p1, p2)

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

    def get_trapeze_angle(self, p1: Point, p2: Point):
        theta_max = min(MAX_DEGREE_OF_MEASURING - self.Config.Theta, self.Config.Vertical_viewing_angle / 2)
        theta_max_radians = math.radians(theta_max)
        theta_radians = math.radians(self.Config.Theta)

        M = Point(int(self.Config.Resolution_width / 2), self.get_degree_coordinate(self.Config.Theta + theta_max))
        high_point, low_point = self.get_high_and_low_points(p1, p2)
        FM = self.get_dx_by_high_point(M, low_point)

        if FM == 0:
            return math.pi / 2

        LM = self.Config.H * (
            math.tan(theta_radians + theta_max_radians) + 1 / math.tan(theta_radians))
        alpha = math.atan(LM / FM)

        # there is a difference whether vector moves from the center to the sides or opposite
        # if abs(high_point.X - M.X) < abs(low_point.X - M.X):
        #     alpha = math.pi - alpha
        return alpha

    def get_high_and_low_points(self, p1: Point, p2: Point):
        if p1.Y < p2.Y:
            high_object = p1
            low_object = p2
        else:
            high_object = p2
            low_object = p1

        return high_object, low_object

    def get_point_closer_to_center_horizontally(self, p1: Point, p2: Point):
        center = int(self.Config.Resolution_width / 2)
        if abs(p1.X - center) < abs(p2.X - center):
            return p1
        else:
            return p2

    def get_pixel_size(self):
        return self.Config.Sensor_size / (
            math.sqrt(self.Config.Resolution_width ** 2 + self.Config.Resolution_height ** 2))

    def get_degree_coordinate(self, degree: int):
        pixel_size = self.get_pixel_size()
        if self.Config.Theta + self.Config.Vertical_viewing_angle / 2 < degree:
            return self.Config.Resolution_height-1
        elif self.Config.Theta - self.Config.Vertical_viewing_angle / 2 > degree:
            return 0

        if degree > self.Config.Theta:
            return int(self.Config.Resolution_height / 2 - self.Config.f * math.tan(
                math.radians(degree - self.Config.Theta)) / pixel_size)
        else:
            return int(self.Config.Resolution_height / 2 + self.Config.f * math.tan(
                math.radians(self.Config.Theta - degree)) / pixel_size)
